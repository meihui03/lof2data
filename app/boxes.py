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


def cited_boxes(ocr_by_index, source_index):
    """The OCR boxes a field's citation points at, in citation order.

    Returns [] when the citation is missing, malformed, or names lines that
    aren't in this document's OCR result - all of which mean "no usable
    citation", so the caller can fall back to locating the value by text."""
    return [
        ocr_by_index[index]
        for index in normalize_source_indexes(source_index)
        if index in ocr_by_index
    ]


# Shortest value that can be looked up as a bare substring. Below this, a
# whole-word hit is required: the choice options are short words ("NOT",
# "Land", "Shop") that occur inside ordinary form prose all the time.
_MIN_SUBSTRING_LEN = 5

# Minimum DISTINCT options on a line before it counts as a printed choice
# list. Matches mark_detector.locate_ocr_line's own min_distinct.
_MIN_DISTINCT_CANDIDATES = 2


def _contains_word(haystack, needle):
    """Is `needle` present in `haystack` as a whole word? Lookarounds rather
    than \\b so a needle that ends in punctuation ("SDN. BHD.") still anchors
    correctly - \\b after a '.' never matches."""
    pattern = r"(?<![A-Za-z0-9])" + re.escape(needle) + r"(?![A-Za-z0-9])"
    return re.search(pattern, haystack, re.IGNORECASE) is not None


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
    needle = str(value).strip()
    if len(needle) < 2:
        return None

    for box in ocr_boxes:
        if _contains_word(box.get("text", ""), needle):
            return box

    if len(needle) < _MIN_SUBSTRING_LEN:
        return None

    upper = needle.upper()
    for box in ocr_boxes:
        if upper in box.get("text", "").upper():
            return box

    prefix = upper[:4]
    for box in ocr_boxes:
        if prefix in box.get("text", "").upper():
            return box
    return None


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
