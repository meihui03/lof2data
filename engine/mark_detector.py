"""Template-free "pick one of N" choice-field reader.

Replaces bbox-ocr-extractor's per-template calibrated boxes
(NATIVE_CHOICE_FIELDS / SCAN_CHOICE_FIELDS) with a two-stage approach that
needs no knowledge of which template it is looking at:

  1. LOCATE - the schema already lists a field's candidate options, so find
     the text line where the most DISTINCT candidates co-occur. A real
     choice line has nearly all of them on one line ("NOT / PARTLY /
     FULLY"); body prose that happens to contain the word "not" has one.
     That distinct-count is what separates them, and it needs no layout
     knowledge. (In production an LLM can supply/confirm the line via the
     source_index it is already asked for - see hybrid/extractor.py's
     format_ocr_text; this module keeps locating standalone so the
     geometry can be tested without an LLM call.)

  2. TEST - for each candidate's own box, decide whether it is struck
     through or circled:
       - born-digital PDF: the strike is real vector geometry
         (page.get_drawings()), so this is exact
       - scan: crop the line, upscale it (a strike is only 2-3px tall at
         page scale), re-OCR to get per-word boxes, then test each word's
         ink profile

  3. DECIDE - exactly one unstruck candidate wins; exactly one circled
     candidate wins; nothing marked returns None (abstain) rather than
     guessing, which is what an unmarked form needs.
"""

import re

import cv2
import fitz
import numpy as np

# Strike geometry, as fractions of a candidate word's own box.
STRIKE_Y_LO, STRIKE_Y_HI = 0.25, 0.80   # mid-glyph band; below this is an underline
STRIKE_MAX_WIDTH_RATIO = 2.5            # reject full-row rules / table borders
STRIKE_MIN_X_OVERLAP = 0.45             # of the word's width
RASTER_MIN_RUN = 0.62                   # ink must span this much of the word's width
RASTER_HOUGH_MIN_RUN = 0.35             # below this, a Hough "line" is just glyph strokes


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _fuzzy_eq(token, candidate, min_prefix=4, min_coverage=0.7):
    """OCR mangles choice words ('FULLY' -> 'FULL', 'Semi-D' -> 'SemiD'), so
    allow a prefix match - but only when both sides are long enough to be
    distinctive AND the shorter covers most of the longer. A naive
    startswith() matched the word "of" to the candidate "Office" (giving it
    a 5.9pt box instead of 18.5pt, so its strikethrough was missed), and
    would equally match "Shop" to "Shop Apt"."""
    a, b = _norm(token), _norm(candidate)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < min_prefix:
        return False
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    return longer.startswith(shorter) and len(shorter) / len(longer) >= min_coverage


# ------------------------------------------------------------------ locating

def _cluster_rows(items, tol_frac=0.6):
    """Groups items into text rows by vertical overlap. items are
    (text, x0, y0, x1, y1)."""
    rows = []
    for item in sorted(items, key=lambda i: i[2]):
        height = max(item[4] - item[2], 1)
        for row in rows:
            if abs(row[0][2] - item[2]) <= height * tol_frac:
                row.append(item)
                break
        else:
            rows.append([item])
    return rows


def locate_native(page, candidates, min_distinct=2):
    """Finds the choice row on a born-digital page and returns
    {candidate: fitz.Rect}. Multi-word candidates ('Shop Apt') are matched
    across consecutive words."""
    words = [(w[4], w[0], w[1], w[2], w[3]) for w in page.get_text("words")]
    best, best_score = {}, 0

    for row in _cluster_rows(words):
        row.sort(key=lambda w: w[1])
        found = {}
        for candidate in candidates:
            parts = candidate.split()
            for i in range(len(row) - len(parts) + 1):
                if all(_fuzzy_eq(row[i + j][0], parts[j]) for j in range(len(parts))):
                    seq = row[i:i + len(parts)]
                    found[candidate] = fitz.Rect(
                        seq[0][1], min(s[2] for s in seq),
                        seq[-1][3], max(s[4] for s in seq))
                    break
        if len(found) > best_score:
            best, best_score = found, len(found)

    return best if best_score >= min_distinct else {}


def locate_ocr_line(ocr_df, candidates, min_distinct=2):
    """Finds the OCR line where the most distinct candidates co-occur.
    Returns (row, n_distinct) or (None, 0). Works on a scan, where only
    line-level boxes exist - per-word boxes come later, from the upscaled
    crop."""
    best, best_score = None, 0
    for row in ocr_df.itertuples():
        text = str(row.text)
        tokens = re.split(r"[\s/,*]+", text)
        hits = {c for c in candidates
                if any(_fuzzy_eq(t, c) for t in tokens)}
        if len(hits) > best_score:
            best, best_score = row, len(hits)
    return (best, best_score) if best_score >= min_distinct else (None, 0)


# ------------------------------------------------ born-digital strike detect

def detect_struck_native(page, candidate_boxes):
    """Marks a candidate struck when a thin drawing crosses its mid-glyph
    band. No row_box needed - the candidate's own box supplies the geometry,
    and the width ratio rejects full-row rules (the ~400pt field underline)
    that would otherwise look like a strike over every option."""
    drawings = [d["rect"] for d in page.get_drawings() if d.get("rect")]
    struck = {}
    for candidate, box in candidate_boxes.items():
        width = max(box.width, 1e-6)
        hit = False
        for rect in drawings:
            if rect.height > box.height * 0.5:
                continue                                    # not a line
            if rect.width > width * STRIKE_MAX_WIDTH_RATIO:
                continue                                    # row rule, not a strike
            y_mid = (rect.y0 + rect.y1) / 2
            depth = (y_mid - box.y0) / max(box.height, 1e-6)
            if not (STRIKE_Y_LO <= depth <= STRIKE_Y_HI):
                continue                                    # underline or unrelated
            overlap = min(rect.x1, box.x1) - max(rect.x0, box.x0)
            if overlap / width >= STRIKE_MIN_X_OVERLAP:
                hit = True
                break
        struck[candidate] = hit
    return struck


# ------------------------------------------------ born-digital circle detect

# Annotation types a person uses to ring or box an option. An annotation is
# always added on top of the generated form, never part of the template, so
# even a rectangle one is a deliberate mark - unlike a printed box.
RING_ANNOT_TYPES = {"Circle", "Square", "Ink", "Polygon", "PolyLine"}
RING_MIN_COVER = 0.6          # of the option's box that the ring must cover


def _area(rect):
    """Area of a rect, 0 for an empty one (a non-overlapping intersection)."""
    return max(rect.width, 0) * max(rect.height, 0)


def _is_drawn_ring(drawing):
    """Is this vector path shaped like a hand ring rather than template
    furniture? A curve (an oval, a Circle annotation's appearance), or a
    freehand polyline - many segments, some diagonal. A printed box is four
    axis-aligned lines or a 're', and stays out."""
    items = drawing.get("items") or []
    if any(item[0] == "c" for item in items):
        return True
    lines = [item for item in items if item[0] == "l"]
    diagonal = [l for l in lines if abs(l[1].x - l[2].x) > 0.5 and abs(l[1].y - l[2].y) > 0.5]
    return len(lines) >= 5 and len(diagonal) >= 2


def _ring_rects(page):
    """Every ring-shaped mark on the page. get_drawings() already includes
    annotation appearances on current PyMuPDF; annots() is read too, so an
    annotation without a rendered appearance is not missed. A ring counted
    twice is harmless - it still encloses the same single option."""
    rects = [d["rect"] for d in page.get_drawings() if d.get("rect") and _is_drawn_ring(d)]
    rects += [annot.rect for annot in page.annots() if annot.type[1] in RING_ANNOT_TYPES]
    return rects


def detect_circled_native(page, candidate_boxes):
    """Marks a candidate circled when a ring covers most of its box and the
    centre of no other candidate. The second rule is what rejects a ring (or
    an oval border) drawn around the whole option line."""
    rings = _ring_rects(page)
    centres = {c: fitz.Point((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2)
               for c, b in candidate_boxes.items()}
    circled = {}
    for candidate, box in candidate_boxes.items():
        hit = False
        for ring in rings:
            cover = _area(ring & box) / max(_area(box), 1e-6)
            if cover < RING_MIN_COVER or not ring.contains(centres[candidate]):
                continue
            if any(ring.contains(p) for other, p in centres.items() if other != candidate):
                continue
            hit = True
            break
        circled[candidate] = hit
    return circled


# ------------------------------------------------------- raster (scan) path

def crop_line(img, box, pad_frac=0.9, target_line_h=110):
    """Crops a text line with vertical padding and upscales it so the LINE
    itself reaches target_line_h. Scaling on the padded crop height instead
    (the first version's mistake) left form2 at scale 1.0 - already 56px
    tall before padding - so nothing was magnified at all.

    Returns (crop, scale, band) where band is the (y0, y1) of the text line
    within the returned crop, in crop pixels."""
    x0, y0, x1, y1 = (int(v) for v in box)
    line_h = max(y1 - y0, 1)
    pad_y, pad_x = int(line_h * pad_frac), int(line_h * 0.3)
    cx0, cy0 = max(x0 - pad_x, 0), max(y0 - pad_y, 0)
    cx1, cy1 = min(x1 + pad_x, img.shape[1]), min(y1 + pad_y, img.shape[0])
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None, 1.0, None

    scale = max(1.0, target_line_h / line_h)
    if scale > 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    band = (int((y0 - cy0) * scale), int((y1 - cy0) * scale))
    # where the LINE itself sits inside the padded, scaled crop - callers
    # need this to map a character offset to a crop x coordinate
    line_x = (int((x0 - cx0) * scale), int((x1 - cx0) * scale))
    return crop, scale, band, line_x


def _ink_segments(ink, band, min_gap_frac=0.18):
    """Splits a text band into horizontal ink segments (words) by finding
    gap columns with no ink. Needed because PaddleOCR returns one box per
    LINE, not per word - so per-option boxes have to come from pixel
    geometry, not from OCR."""
    y0, y1 = band
    strip = ink[max(y0, 0):min(y1, ink.shape[0]), :]
    if strip.size == 0:
        return []
    columns = strip.sum(axis=0) > 0
    min_gap = max(int((y1 - y0) * min_gap_frac), 2)

    segments, start, gap = [], None, 0
    for x, has_ink in enumerate(columns):
        if has_ink:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                segments.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None:
        segments.append((start, len(columns)))
    return [(a, b) for a, b in segments if b - a >= 2]


def word_boxes_by_segment_ocr(crop, ink, band, candidates, ocr_engine,
                              pad_frac=0.35, max_span=2):
    """Locates each candidate's own box by segmenting the line into visual
    words (ink gaps) and OCRing EACH SEGMENT separately.

    This replaces character-proportional estimation, which was off by a
    whole word on form2: the lowercase lead-in ("9) This Premise shall be
    rented ") is far narrower per character than the bold uppercase options
    that follow, so a linear char->pixel map put FULLY on top of FURNISHED.
    Reading each segment tells us what it actually says, so no guessing.

    max_span lets a multi-word candidate ("Shop Apt") claim adjacent
    segments. A circled option's ring usually merges into its own segment,
    which is harmless - the box just includes the ring.
    """
    segments = _ink_segments(ink, band)
    if not segments:
        return {}, segments

    y0, y1 = band
    pad = int((y1 - y0) * pad_frac)
    top, bottom = max(y0 - pad, 0), min(y1 + pad, crop.shape[0])

    texts = []
    for a, b in segments:
        piece = crop[top:bottom, max(a - 2, 0):min(b + 2, crop.shape[1])]
        if piece.size == 0 or piece.shape[1] < 4:
            texts.append("")
            continue
        try:
            result = ocr_engine.predict(piece)[0]
            texts.append(" ".join(result["rec_texts"]))
        except Exception:
            texts.append("")

    boxes, taken = {}, set()
    for candidate in candidates:
        parts = [p for p in re.split(r"[\s\-]+", candidate) if p]
        for span in range(1, max_span + 1):
            if candidate in boxes:
                break
            for i in range(len(segments) - span + 1):
                if any(j in taken for j in range(i, i + span)):
                    continue
                joined = " ".join(texts[i:i + span])
                tokens = [t for t in re.split(r"[\s/,*()]+", joined) if t]
                if span == 1 and len(parts) == 1:
                    matched = any(_fuzzy_eq(t, candidate) for t in tokens)
                else:
                    matched = (len(tokens) >= len(parts) and
                               all(any(_fuzzy_eq(t, p) for t in tokens) for p in parts))
                if matched:
                    boxes[candidate] = (segments[i][0], y0, segments[i + span - 1][1], y1)
                    taken.update(range(i, i + span))
                    break
    return boxes, list(zip(segments, texts))


def word_boxes_from_line(ink, band, line_x, line_text, candidates):
    """Locates each candidate's own box inside the crop.

    Estimates a candidate's x-window from its character offset in the OCR
    line text (proportional across the line's pixel span, which crop_line
    reports as line_x), then snaps that window to the ink segment it
    overlaps most - so the final box hugs real pixels rather than a linear
    guess, which matters because the fonts are proportional.

    Each segment is claimed by at most one candidate: without that, the
    proportional estimates all collapsed onto whichever segment happened to
    overlap most and every option got the same box."""
    segments = _ink_segments(ink, band)
    if not segments:
        return {}, segments

    tokens = [(m.group(0), m.start(), m.end())
              for m in re.finditer(r"[A-Za-z][A-Za-z\-]*", line_text)]
    total_chars = max(len(line_text), 1)
    lx0, lx1 = line_x
    line_w = max(lx1 - lx0, 1)

    estimates = {}
    for candidate in candidates:
        parts = [p for p in re.split(r"[\s\-]+", candidate) if p]
        for i in range(len(tokens) - len(parts) + 1):
            if all(_fuzzy_eq(tokens[i + j][0], parts[j]) for j in range(len(parts))):
                span = (tokens[i][1], tokens[i + len(parts) - 1][2])
                estimates[candidate] = (
                    lx0 + (span[0] / total_chars) * line_w,
                    lx0 + (span[1] / total_chars) * line_w,
                )
                break

    # assign best-fitting segment first, so a confident match is not stolen
    boxes, taken = {}, set()
    ranked = []
    for candidate, (est0, est1) in estimates.items():
        for idx, (a, b) in enumerate(segments):
            overlap = min(b, est1) - max(a, est0)
            if overlap > 0:
                ranked.append((overlap, candidate, idx))
    for overlap, candidate, idx in sorted(ranked, reverse=True):
        if candidate in boxes or idx in taken:
            continue
        a, b = segments[idx]
        boxes[candidate] = (int(a), band[0], int(b), band[1])
        taken.add(idx)

    # anything with no free segment falls back to its linear estimate
    for candidate, (est0, est1) in estimates.items():
        if candidate not in boxes:
            boxes[candidate] = (int(est0), band[0], int(est1), band[1])
    return boxes, segments


def _binary_ink(crop):
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 10)


def word_boxes_in_crop(crop, candidates, ocr_engine):
    """Re-OCRs the upscaled crop so each option gets its OWN box. At page
    scale PaddleOCR returns the whole choice list as a single line box
    (form2's 9 options came back as one 2269px-wide region), which is too
    coarse to test per-option."""
    result = ocr_engine.predict(crop)[0]
    boxes = {}
    for text, box in zip(result["rec_texts"], result["rec_boxes"]):
        for token in re.split(r"[\s/,*]+", str(text)):
            for candidate in candidates:
                if candidate not in boxes and _fuzzy_eq(token, candidate):
                    boxes[candidate] = tuple(int(v) for v in box)
    return boxes


def detect_struck_raster(crop, candidate_boxes):
    """A struck word has a near-full-width horizontal ink run inside its own
    box, in the mid-glyph band - glyphs alone do not produce that. Falls
    back to a Hough line test with angle tolerance so a slanted hand strike
    is not missed."""
    ink = _binary_ink(crop)
    struck, evidence = {}, {}
    for candidate, (x0, y0, x1, y1) in candidate_boxes.items():
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, ink.shape[1]), min(y1, ink.shape[0])
        cell = ink[y0:y1, x0:x1]
        if cell.size == 0:
            struck[candidate] = False
            continue
        width, height = cell.shape[1], cell.shape[0]
        best_run, best_depth = 0.0, None
        for row in range(height):
            depth = row / max(height - 1, 1)
            if not (STRIKE_Y_LO <= depth <= STRIKE_Y_HI):
                continue
            run = _longest_run(cell[row] > 0) / max(width, 1)
            if run > best_run:
                best_run, best_depth = run, round(depth, 2)

        hit = best_run >= RASTER_MIN_RUN
        # Hough only CORROBORATES a partial run (a slanted hand strike breaks
        # the horizontal run into pieces). Ungated it fired on the glyph
        # strokes of every option and reported the whole line struck.
        if not hit and best_run >= RASTER_HOUGH_MIN_RUN:
            hit = _hough_strike(cell)
        struck[candidate] = hit
        evidence[candidate] = {"run": round(best_run, 2), "depth": best_depth}
    return struck, evidence


def _longest_run(mask):
    best = run = 0
    for value in mask:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def _hough_strike(cell, angle_tol=20):
    """Catches a diagonal/wavy hand strike that a horizontal run test
    misses."""
    height, width = cell.shape
    lines = cv2.HoughLinesP(cell, 1, np.pi / 180,
                            threshold=max(int(width * 0.4), 12),
                            minLineLength=int(width * 0.55), maxLineGap=int(width * 0.15))
    if lines is None:
        return False
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        angle = min(angle, 180 - angle)
        mid_depth = ((y1 + y2) / 2) / max(height - 1, 1)
        if angle <= angle_tol and STRIKE_Y_LO <= mid_depth <= STRIKE_Y_HI:
            return True
    return False


def find_ring_and_read(crop, ink, band, candidates, ocr_engine,
                       inset=0.14, min_hole_frac=0.12):
    """Finds a hand-drawn ring in the line band, then reads what is INSIDE
    it to identify which option was circled.

    This ordering matters. Testing each option's box for "is there a ring
    around me" fails twice over: the box is derived from ink so it already
    contains the ring (nothing looks bigger than the word), and a ring often
    bridges the gap to its neighbour so the two words share one segment
    (form2's furnished line segmented as a single 'PARTLYFULLY'). Locating
    the ring first and reading its interior sidesteps both - the same
    strategy bbox-ocr-extractor's detect_circled_choice uses, but with the
    search region coming from OCR rather than a calibrated y0/y1.

    Returns (candidate, ring_rect) or (None, None).
    """
    y0, y1 = band
    band_h = max(y1 - y0, 1)
    ey0, ey1 = max(int(y0 - band_h * 0.6), 0), min(int(y1 + band_h * 0.6), ink.shape[0])
    region = ink[ey0:ey1, :]
    if region.size == 0:
        return None, None

    # close small breaks so a slightly open hand-drawn ring still forms a loop
    k = max(int(band_h * 0.05), 2)
    closed = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None, None
    hierarchy = hierarchy[0]

    # a ring encloses a large hole; glyph counters (o, d, e) are far smaller
    holes = {}
    for idx, (_, _, child, parent) in enumerate(hierarchy):
        if parent == -1:
            continue
        area = cv2.contourArea(contours[idx])
        holes[parent] = max(holes.get(parent, 0), area)

    best, best_area = None, 0
    for idx, contour in enumerate(contours):
        if hierarchy[idx][3] != -1:
            continue
        cx, cy, cw, ch = cv2.boundingRect(contour)
        if not (band_h * 0.5 <= cw <= band_h * 8):
            continue
        if not (band_h * 0.5 <= ch <= band_h * 2.6):
            continue
        if holes.get(idx, 0) < min_hole_frac * cw * ch:
            continue
        fill = cv2.contourArea(contour) / max(cw * ch, 1)
        if not (0.10 <= fill <= 0.95):
            continue
        if cw * ch > best_area:
            best, best_area = (cx, cy, cw, ch), cw * ch

    if best is None:
        return None, None

    cx, cy, cw, ch = best
    ix, iy = int(cw * inset), int(ch * inset)
    inner = crop[max(ey0 + cy + iy, 0):ey0 + cy + ch - iy,
                 max(cx + ix, 0):cx + cw - ix]
    if inner.size == 0 or inner.shape[0] < 4 or inner.shape[1] < 4:
        return None, best
    try:
        text = " ".join(ocr_engine.predict(inner)[0]["rec_texts"])
    except Exception:
        return None, best

    tokens = [t for t in re.split(r"[\s/,*()]+", text) if t]
    for candidate in candidates:
        parts = [p for p in re.split(r"[\s\-]+", candidate) if p]
        if all(any(_fuzzy_eq(t, p) for t in tokens) for p in parts):
            return candidate, best
    return None, best


def detect_circled_raster(crop, candidate_boxes):
    """A hand-drawn ring shows up as a hollow contour enclosing one option:
    larger than the word, mostly empty inside, roughly elliptical."""
    ink = _binary_ink(crop)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    circled = {}
    for candidate, (x0, y0, x1, y1) in candidate_boxes.items():
        wbox = (x0, y0, x1, y1)
        found = False
        for contour in contours:
            cx, cy, cw, ch = cv2.boundingRect(contour)
            if cw * ch < (x1 - x0) * (y1 - y0) * 1.15:
                continue                                     # not enclosing
            if not (cx <= x0 + 3 and cy <= y0 + 3 and cx + cw >= x1 - 3 and cy + ch >= y1 - 3):
                continue
            if not (0.25 <= cw / max(ch, 1) <= 6.0):
                continue
            fill = cv2.contourArea(contour) / max(cw * ch, 1)
            if 0.15 <= fill <= 0.92:                         # hollow ring, not solid print
                found = True
                break
        circled[candidate] = found
    return circled


# ------------------------------------------------------------------ decision

def resolve(struck=None, circled=None):
    """Convention: exactly one circled option wins; otherwise exactly one
    un-struck option wins. Anything ambiguous returns None so the caller
    abstains instead of guessing - the behaviour an unmarked form needs."""
    if circled:
        picked = [c for c, hit in circled.items() if hit]
        if len(picked) == 1:
            return picked[0], "circled"
    if struck:
        unstruck = [c for c, hit in struck.items() if not hit]
        if len(unstruck) == 1 and any(struck.values()):
            return unstruck[0], "strikethrough"
    return None, "abstain"
