"""Tests for boxes.py - turning an LLM's "source_index" citation into the
OCR boxes to draw on the source document.

Stdlib unittest, run with:

    .venv/bin/python3 test_boxes.py

This exists because `ocr_by_index.get(idx)` crashed with "unhashable type:
'list'" the first time a model cited more than one OCR line for a value.
The prompt asks for a single [N] index, but nothing makes the model comply,
and the citation is drawn straight from parsed model JSON - so every shape a
model might plausibly emit has to land somewhere sane rather than take out
the whole Source Document tab.
"""

import unittest

import boxes

OCR = [
    {"index": 0, "page": 0, "text": "LETTER OF OFFER", "x0": 0.1, "y0": 0.1, "x1": 0.5, "y1": 0.2},
    {"index": 1, "page": 0, "text": "ALICE TAN", "x0": 0.1, "y0": 0.3, "x1": 0.5, "y1": 0.4},
    {"index": 2, "page": 1, "text": "BRIAN LIM", "x0": 0.1, "y0": 0.5, "x1": 0.5, "y1": 0.6},
]
BY_INDEX = {b["index"]: b for b in OCR}


class TestNormalizeSourceIndexes(unittest.TestCase):
    def test_scalar_int(self):
        self.assertEqual(boxes.normalize_source_indexes(2), [2])

    def test_list_of_ints(self):
        self.assertEqual(boxes.normalize_source_indexes([1, 2]), [1, 2])

    def test_none_is_empty(self):
        self.assertEqual(boxes.normalize_source_indexes(None), [])

    def test_empty_list_is_empty(self):
        self.assertEqual(boxes.normalize_source_indexes([]), [])

    def test_numeric_string_is_coerced(self):
        # OCR box indexes are ints, so "1" has to become 1 or it never matches.
        self.assertEqual(boxes.normalize_source_indexes("1"), [1])
        self.assertEqual(boxes.normalize_source_indexes(["1", "2"]), [1, 2])

    def test_non_numeric_entries_are_dropped(self):
        self.assertEqual(boxes.normalize_source_indexes(["1", "abc", None, 2]), [1, 2])

    def test_nested_list_is_flattened(self):
        self.assertEqual(boxes.normalize_source_indexes([[1], [2, 3]]), [1, 2, 3])

    def test_duplicates_removed_preserving_order(self):
        self.assertEqual(boxes.normalize_source_indexes([2, 1, 2]), [2, 1])

    def test_dict_is_dropped_not_raised(self):
        # A model returning an object here must not take down the page.
        self.assertEqual(boxes.normalize_source_indexes({"n": 1}), [])

    def test_bool_is_not_an_index(self):
        # True == 1 in Python; treating it as index 1 would draw a wrong box.
        self.assertEqual(boxes.normalize_source_indexes(True), [])

    def test_float_index_is_dropped(self):
        self.assertEqual(boxes.normalize_source_indexes(1.5), [])


class TestCitedBoxes(unittest.TestCase):
    def test_single_index_returns_one_box(self):
        result = boxes.cited_boxes(BY_INDEX, 1)
        self.assertEqual([b["text"] for b in result], ["ALICE TAN"])

    def test_list_returns_every_cited_box(self):
        result = boxes.cited_boxes(BY_INDEX, [1, 2])
        self.assertEqual([b["text"] for b in result], ["ALICE TAN", "BRIAN LIM"])

    def test_boxes_can_span_pages(self):
        self.assertEqual([b["page"] for b in boxes.cited_boxes(BY_INDEX, [1, 2])], [0, 1])

    def test_unknown_index_is_skipped(self):
        self.assertEqual(boxes.cited_boxes(BY_INDEX, [1, 99]),
                         [BY_INDEX[1]])

    def test_all_unknown_returns_empty(self):
        # Empty means "no citation" so the caller falls back to text matching.
        self.assertEqual(boxes.cited_boxes(BY_INDEX, [99]), [])

    def test_none_returns_empty(self):
        self.assertEqual(boxes.cited_boxes(BY_INDEX, None), [])

    def test_list_does_not_raise(self):
        # The original crash: dict.get() on an unhashable list.
        try:
            boxes.cited_boxes(BY_INDEX, [1, 2])
        except TypeError as e:
            self.fail(f"cited_boxes raised on a list source_index: {e}")

    def test_index_zero_is_a_valid_citation(self):
        # 0 is falsy - a naive `if idx:` guard would silently drop it.
        self.assertEqual([b["text"] for b in boxes.cited_boxes(BY_INDEX, 0)],
                         ["LETTER OF OFFER"])


class TestFindBoxByText(unittest.TestCase):
    """The fallback used when a model cites no source_index - which is ALWAYS
    the case for the two choice fields, whose value comes from a circled or
    struck mark rather than printed text. An unanchored substring search puts
    those boxes on the wrong line entirely."""

    # A realistic page: the Notice-to-Quit clause is printed above the
    # furnished-status choice line, and contains "NOT" inside "NOTICE".
    CHOICE_PAGE = [
        {"index": 0, "page": 0, "text": "14. NOTICE TO QUIT: TWO (2) MONTHS"},
        {"index": 1, "page": 0, "text": "Furnished Status: NOT / PARTLY / FULLY FURNISHED"},
    ]

    def test_short_value_does_not_match_inside_a_longer_word(self):
        # "NOT" must not land on "NOTICE TO QUIT".
        box = boxes.find_box_by_text(self.CHOICE_PAGE, "NOT")
        self.assertEqual(box["index"], 1)

    def test_whole_word_match_still_found(self):
        box = boxes.find_box_by_text(self.CHOICE_PAGE, "FULLY")
        self.assertEqual(box["index"], 1)

    def test_prefers_whole_word_over_an_earlier_substring(self):
        ocr = [
            {"index": 0, "page": 0, "text": "SUBLETTING IS NOT PERMITTED"},
            {"index": 1, "page": 0, "text": "PARTLY"},
        ]
        self.assertEqual(boxes.find_box_by_text(ocr, "PARTLY")["index"], 1)

    def test_value_absent_returns_none(self):
        self.assertIsNone(boxes.find_box_by_text(self.CHOICE_PAGE, "BUNGALOW"))

    def test_single_character_is_too_short_to_search(self):
        self.assertIsNone(boxes.find_box_by_text(self.CHOICE_PAGE, "N"))

    def test_prefix_fallback_absorbs_a_dropped_trailing_character(self):
        # PaddleOCR commonly clips the last letter of a word a mark is drawn
        # through - "FULLY" recognized as "FULL". The prefix path must survive.
        ocr = [{"index": 0, "page": 0, "text": "NOT / PARTLY / FULL FURNISHED"}]
        self.assertEqual(boxes.find_box_by_text(ocr, "FULLY")["index"], 0)

    def test_empty_value_returns_none(self):
        self.assertIsNone(boxes.find_box_by_text(self.CHOICE_PAGE, ""))


class TestFindChoiceLine(unittest.TestCase):
    """Locating a choice field's printed option line the way mark_detector
    does it - the line where the most DISTINCT candidates co-occur. Prose that
    happens to contain one option word has exactly one; the real choice line
    has nearly all of them."""

    def test_picks_the_line_where_most_candidates_co_occur(self):
        ocr = [
            {"index": 0, "page": 0, "text": "Land Office reference: LAND TITLE"},
            {"index": 1, "page": 0, "text": "House / Semi-D / Bungalow / Condo / Land"},
        ]
        candidates = ["House", "Semi-D", "Bungalow", "Condo", "Land"]
        self.assertEqual(boxes.find_choice_line(ocr, candidates)["index"], 1)

    def test_single_candidate_hit_is_not_a_choice_line(self):
        # One match is prose, not an option list - abstain so the caller can
        # fall back rather than box a random sentence.
        ocr = [{"index": 0, "page": 0, "text": "Land Office reference only"}]
        self.assertIsNone(boxes.find_choice_line(ocr, ["House", "Condo", "Land"]))

    def test_no_candidates_returns_none(self):
        ocr = [{"index": 0, "page": 0, "text": "NOT / PARTLY / FULLY"}]
        self.assertIsNone(boxes.find_choice_line(ocr, []))

    def test_candidate_inside_a_longer_word_does_not_count(self):
        # "NOTICE" must not count as the candidate "NOT".
        ocr = [
            {"index": 0, "page": 0, "text": "NOTICE TO QUIT AND PARTLY VACATE"},
            {"index": 1, "page": 0, "text": "NOT / PARTLY / FULLY"},
        ]
        self.assertEqual(boxes.find_choice_line(ocr, ["NOT", "PARTLY", "FULLY"])["index"], 1)

    def test_empty_ocr_returns_none(self):
        self.assertIsNone(boxes.find_choice_line([], ["NOT", "PARTLY"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
