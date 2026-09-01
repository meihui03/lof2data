"""Turning an LLM's "source_index" citation into OCR boxes to draw.

Split out of app.py so it can be tested without executing the Streamlit UI.

The prompt asks each extracted field to cite the single `[N]` OCR line that
backs it, but the citation comes straight out of parsed model JSON and
nothing enforces that shape. Models do emit others - a list of indexes when
a value spans more than one OCR line is the common one, and passing that
straight to `ocr_by_index.get()` raised "unhashable type: 'list'" and took
out the whole Source Document tab.

A multi-line citation is real information rather than a malformed answer, so
a list draws a box per cited line instead of being collapsed to one. Shapes
that carry no usable index are dropped, which leaves the caller to fall back
to matching the extracted text against the OCR result.

That fallback (find_box_by_text / find_choice_line) lives here too, since it
is what the two choice fields ALWAYS use - their value comes from a circled
or struck mark, which the model has no OCR line to cite for.
"""

import re


def normalize_source_indexes(source_index):
    """Coerces whatever a model put in "source_index" into a list of int OCR
    indexes, in citation order and without repeats. Anything that isn't a
    usable index is dropped rather than raising - a bad citation should cost
    one box, not the page."""
    candidates = source_index if isinstance(source_index, (list, tuple)) else [source_index]

    indexes, seen = [], set()
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            # Some models nest the array one level deeper than asked.
            nested = normalize_source_indexes(candidate)
        elif isinstance(candidate, bool):
            # bool is an int subclass, and True would otherwise read as
            # index 1 and draw a box over an unrelated OCR line.
            continue
        elif isinstance(candidate, int):
            nested = [candidate]
        elif isinstance(candidate, str) and candidate.strip().lstrip("-").isdigit():
            nested = [int(candidate)]
        else:
            # None, floats, dicts, "[12]", prose - no usable index.
            continue

        for index in nested:
            if index not in seen:
                seen.add(index)
                indexes.append(index)
    return indexes


def normalize_leaf(leaf):
    """A field's parsed value as a {value, confidence, source_index} dict,
    whatever shape the model actually used.

    The prompt asks for that envelope and nothing enforces it. Seen on a real
    run: every party in every group came back as a bare string -

        landlord: [{"Name": "TAN SIEW KIAU", "NRIC/Passport No.": "6406..."}]

    extractor._safe_leaf already absorbs this on the flattening side, which is
    why the comparison table filled in normally while the box collector - which
    skipped anything that wasn't a dict - drew nothing for the entire signature
    block. Both sides need the same tolerance or they disagree about what was
    extracted, and the disagreement shows up as silently missing boxes."""
    if isinstance(leaf, dict):
        return leaf
    return {"value": leaf, "confidence": None, "source_index": None}


# How extractor.join_occurrences joins a repeatable role into one field.
# Spaces on both sides matter: "A/L" and "S/O" in Malaysian names, and
# "NRIC/Passport No.", contain a bare slash that is not a separator.
OCCURRENCE_SEPARATOR = " / "


def split_occurrences(value):
    """A flattened repeatable field back into its individual values.

    "Landlord Name(s)" can hold two co-landlords joined with " / ", and no
    single OCR line contains both - so a box lookup has to search for each
    one separately or it finds nothing at all."""
    if not value:
        return []
    return [part.strip() for part in str(value).split(OCCURRENCE_SEPARATOR) if part.strip()]


def _alphanumeric(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def citation_is_plausible(ocr_text, value):
    """Does `ocr_text` - the line a model cited - actually back `value`?

    A citation is just a number in model JSON and nothing makes it correct.
    An index that merely resolves gets a confidently-drawn box on the wrong
    part of the page, which is worse than no box: on a real run the whole
    signature block's boxes landed in the upper half of the form, leaving
    that block looking unread.

    Meaningful overlap rather than equality, because values are routinely
    reformatted - an added "RM", different punctuation, a label stripped off
    the front - and a long value may be cited across two wrapped lines, so
    either side containing the other counts. The 4-character run is the same
    rule extractor.verify_source_index documents for this job; that function
    is engine-side and needs a DataFrame, so the rule lives here too, where
    the UI can apply it before drawing."""
    value_norm, ocr_norm = _alphanumeric(value), _alphanumeric(ocr_text)
    if not value_norm or not ocr_norm:
        return False
    if value_norm in ocr_norm or ocr_norm in value_norm:
        return True
    return len(value_norm) >= 4 and any(
        value_norm[i:i + 4] in ocr_norm for i in range(len(value_norm) - 3)
    )


def cited_boxes(ocr_by_index, source_index, value=None):
    """The OCR boxes a field's citation points at, in citation order.

    Returns [] when the citation is missing, malformed, or names lines that
    aren't in this document's OCR result - all of which mean "no usable
    citation", so the caller can fall back to locating the value by text.

    Pass `value` to also drop cited lines whose text doesn't back it (see
    citation_is_plausible). A citation that survives is worth trusting over a
    text match; one that doesn't is worse than nothing, so it goes and the
    caller falls back. Omitting `value` keeps every resolvable citation, for
    callers with nothing to validate against."""
    found = [
        ocr_by_index[index]
        for index in normalize_source_indexes(source_index)
        if index in ocr_by_index
    ]
    if value is None:
        return found
    return [b for b in found if citation_is_plausible(b.get("text", ""), value)]


# Shortest value that can be looked up as a bare substring. Below this, a
# whole-word hit is required: the choice options are short words ("NOT",
# "Land", "Shop") that occur inside ordinary form prose all the time.
_MIN_SUBSTRING_LEN = 5

# Minimum DISTINCT options on a line before it counts as a printed choice
# list. Matches mark_detector.locate_ocr_line's own min_distinct.
_MIN_DISTINCT_CANDIDATES = 2


def _collapse_whitespace(text):
    """Runs of whitespace to one space.

    Forms pad with spaces to push a value up against a printed rule - the text
    layer holds ": RM     4,400.00" while the model returns "RM 4,400.00" - so
    matching literally loses a box for a value sitting right there. Collapsing
    both sides makes the padding invisible without joining separate words."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _contains_word(haystack, needle):
    """Is `needle` present in `haystack` as a whole word? Lookarounds rather
    than \\b so a needle that ends in punctuation ("SDN. BHD.") still anchors
    correctly - \\b after a '.' never matches."""
    pattern = r"(?<![A-Za-z0-9])" + re.escape(_collapse_whitespace(needle)) + r"(?![A-Za-z0-9])"
    return re.search(pattern, _collapse_whitespace(haystack), re.IGNORECASE) is not None


def find_box_by_text(ocr_boxes, value):
    """The OCR box a value was most likely read from, for fields the model
    cited no source_index for.

    Tried in descending order of confidence: a whole-word match, then a bare
    substring, then a 4-character prefix. The two weaker tiers are gated on
    length because an unanchored search for a short value lands on the first
    line that merely CONTAINS those letters - "NOT" matching "NOTICE TO QUIT"
    put the Furnished Status box on the notice-period clause. The prefix tier
    stays because PaddleOCR routinely clips the last letter of a word a mark
    is drawn through ("FULLY" recognized as "FULL").

    Returns None when nothing matches, or when the value is too short to
    search for at all - no box beats a confidently wrong one.
    """
    needle = _collapse_whitespace(value)
    if len(needle) < 2:
        return None

    for box in ocr_boxes:
        if _contains_word(box.get("text", ""), needle):
            return box

    if len(needle) < _MIN_SUBSTRING_LEN:
        return None

    upper = _collapse_whitespace(needle).upper()
    for box in ocr_boxes:
        if upper in _collapse_whitespace(box.get("text", "")).upper():
            return box

    # A multi-word value is often returned whole by the model while OCR split
    # it across lines, so no single line contains all of it. Fall back to the
    # line sharing the most words with it. NOT the prefix tier below: four
    # characters of a long value identify nothing - an address beginning
    # "NO. 50, JALAN ..." matched "Business Reg No. ..." on the prefix "NO. "
    # and boxed the letterhead.
    if " " in needle:
        return _best_word_overlap(ocr_boxes, needle)

    # One word whose tail OCR clipped ("FULLY" recognized as "FULL"). Only a
    # single word can be mangled that way.
    prefix = upper[:4]
    for box in ocr_boxes:
        if prefix in _collapse_whitespace(box.get("text", "")).upper():
            return box
    return None


# Words shorter than this are skipped when comparing a wrapped value against a
# line - "no", "of", "7" coincide constantly and carry no locating power.
_MIN_OVERLAP_WORD_LEN = 3

# A single word in common is coincidence. Two is a location.
_MIN_OVERLAP_WORDS = 2


def _best_word_overlap(ocr_boxes, value):
    """The OCR line sharing the most distinct words with `value`, or None if
    no line shares enough to be more than coincidence."""
    wanted = {w for w in re.split(r"[^a-z0-9]+", value.lower())
              if len(w) >= _MIN_OVERLAP_WORD_LEN}
    if not wanted:
        return None

    best, best_count = None, 0
    for box in ocr_boxes:
        words = set(re.split(r"[^a-z0-9]+", box.get("text", "").lower()))
        count = len(wanted & words)
        if count > best_count:
            best, best_count = box, count
    return best if best_count >= _MIN_OVERLAP_WORDS else None


def find_choice_line(ocr_boxes, candidates):
    """The OCR box holding a choice field's printed option list, found the
    same way mark_detector.locate_ocr_line finds it: the line where the most
    DISTINCT options co-occur.

    This is what a choice field's box should use in preference to
    find_box_by_text. The selected option on its own is a short, common word,
    but the LINE it sits on carries nearly the whole option list ("NOT /
    PARTLY / FULLY"), while prose that happens to use one of those words
    carries exactly one. Counting distinct options separates the two without
    knowing anything about the form's layout.

    Returns None when no line carries at least _MIN_DISTINCT_CANDIDATES of
    them - i.e. this form states the field as free text rather than a printed
    list - so the caller falls back to find_box_by_text.
    """
    best, best_count = None, 0
    for box in ocr_boxes:
        text = box.get("text", "")
        count = sum(1 for c in candidates if c and _contains_word(text, c))
        if count > best_count:
            best, best_count = box, count
    return best if best_count >= _MIN_DISTINCT_CANDIDATES else None
