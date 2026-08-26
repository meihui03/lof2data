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
"""


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
