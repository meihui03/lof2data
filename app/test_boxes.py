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


class TestNormalizeLeaf(unittest.TestCase):
    """The prompt asks for {value, confidence, source_index} per field, and
    nothing makes the model comply. Observed on a real run: every party in
    every group came back as a BARE STRING -

        landlord: [{"Name": "TAN SIEW KIAU", "NRIC/Passport No.": "6406..."}]

    The flattener already tolerates that (extractor._safe_leaf), so the
    comparison table filled in normally, while the box collector skipped every
    non-dict leaf and the whole signature block went unboxed. Same tolerance
    has to exist on both sides or they disagree about what was extracted."""

    def test_a_bare_string_becomes_a_leaf(self):
        self.assertEqual(boxes.normalize_leaf("TAN SIEW KIAU"),
                         {"value": "TAN SIEW KIAU", "confidence": None, "source_index": None})

    def test_a_proper_leaf_is_unchanged(self):
        leaf = {"value": "x", "confidence": 0.9, "source_index": 3}
        self.assertEqual(boxes.normalize_leaf(leaf), leaf)

    def test_a_number_becomes_a_leaf(self):
        self.assertEqual(boxes.normalize_leaf(2200)["value"], 2200)

    def test_none_becomes_an_empty_leaf(self):
        self.assertIsNone(boxes.normalize_leaf(None)["value"])

    def test_a_partial_dict_keeps_its_keys(self):
        # Some models return {"value": ...} with no confidence/source_index.
        self.assertEqual(boxes.normalize_leaf({"value": "x"})["value"], "x")
        self.assertIsNone(boxes.normalize_leaf({"value": "x"}).get("source_index"))


class TestSplitOccurrences(unittest.TestCase):
    """A repeatable role is flattened into one field joined with " / "
    (extractor.join_occurrences), so "Landlord Name(s)" can hold two people.
    No single OCR line contains both, so the joined string has to be split
    back apart before anything tries to locate it."""

    def test_two_parties_split(self):
        self.assertEqual(boxes.split_occurrences("ALICE TAN / BRIAN LIM"),
                         ["ALICE TAN", "BRIAN LIM"])

    def test_single_value_is_left_alone(self):
        self.assertEqual(boxes.split_occurrences("ALICE TAN"), ["ALICE TAN"])

    def test_blank_is_empty(self):
        self.assertEqual(boxes.split_occurrences(""), [])
        self.assertEqual(boxes.split_occurrences(None), [])

    def test_a_slash_inside_a_value_is_not_a_separator(self):
        # The separator is " / " with spaces; "A/L" and "S/O" in Malaysian
        # names, and "NRIC/Passport", must survive.
        self.assertEqual(boxes.split_occurrences("SATHISWARAN A/L THAMBYRAJAH"),
                         ["SATHISWARAN A/L THAMBYRAJAH"])

    def test_blank_segments_are_dropped(self):
        self.assertEqual(boxes.split_occurrences("ALICE TAN /  / BRIAN LIM"),
                         ["ALICE TAN", "BRIAN LIM"])


class TestCitationIsPlausible(unittest.TestCase):
    """Does the OCR line a model cited actually back the value it extracted?

    A citation is just a number in model JSON, and nothing makes it correct.
    An index that happens to exist gets a confidently-drawn box on the wrong
    part of the page - seen on a real run, where the whole signature block's
    boxes landed in the upper half of the form and that block looked bare.
    Values are routinely reformatted (an added RM, different punctuation), so
    this checks for meaningful overlap rather than an exact match."""

    def test_exact_line_backs_the_value(self):
        self.assertTrue(boxes.citation_is_plausible(": LEM SIEW YUEN", "LEM SIEW YUEN"))

    def test_reformatted_amount_still_backs_it(self):
        # The model adds "RM" and the form prints only the digits.
        self.assertTrue(boxes.citation_is_plausible(": 2,200.00", "RM 2,200.00"))

    def test_label_prefix_still_backs_it(self):
        self.assertTrue(boxes.citation_is_plausible("Date:   2026-03-31", "2026-03-31"))

    def test_value_spanning_two_lines_backs_each(self):
        # An address the PDF wraps; each cited line is a slice of the value.
        value = "5, LORONG MARKISAH 23, TAMAN MARKISAH, 14000 BUKIT MERTAJAM, PULAU PINANG"
        self.assertTrue(boxes.citation_is_plausible(": 5, LORONG MARKISAH 23, TAMAN MARKISAH, 14000", value))
        self.assertTrue(boxes.citation_is_plausible("BUKIT MERTAJAM, PULAU PINANG", value))

    def test_unrelated_line_does_not_back_it(self):
        # The actual bug: a tenant name cited against the property-type line.
        self.assertFalse(boxes.citation_is_plausible(
            "Description of Property: House / Semi-D / Bungalow / Condo", "LEM SIEW YUEN"))

    def test_wrong_nric_does_not_back_it(self):
        self.assertFalse(boxes.citation_is_plausible(": 931130-14-6848", "960314105840"))

    def test_blank_value_is_not_checkable(self):
        self.assertFalse(boxes.citation_is_plausible("anything", ""))


class TestCitedBoxesValidation(unittest.TestCase):
    OCR = [
        {"index": 0, "page": 0, "text": "Description of Property: House / Condo"},
        {"index": 1, "page": 0, "text": ": LEM SIEW YUEN"},
        {"index": 2, "page": 0, "text": ": 5, LORONG MARKISAH 23, TAMAN MARKISAH, 14000"},
        {"index": 3, "page": 0, "text": "BUKIT MERTAJAM, PULAU PINANG"},
    ]
    BY_INDEX = {b["index"]: b for b in OCR}

    def test_implausible_citation_is_dropped(self):
        # Caller gets [] and falls back to locating the value by text.
        self.assertEqual(boxes.cited_boxes(self.BY_INDEX, 0, value="LEM SIEW YUEN"), [])

    def test_plausible_citation_is_kept(self):
        result = boxes.cited_boxes(self.BY_INDEX, 1, value="LEM SIEW YUEN")
        self.assertEqual([b["index"] for b in result], [1])

    def test_multi_line_citation_keeps_every_backing_line(self):
        value = "5, LORONG MARKISAH 23, TAMAN MARKISAH, 14000 BUKIT MERTAJAM, PULAU PINANG"
        result = boxes.cited_boxes(self.BY_INDEX, [2, 3], value=value)
        self.assertEqual([b["index"] for b in result], [2, 3])

    def test_mixed_citation_keeps_only_the_backing_line(self):
        result = boxes.cited_boxes(self.BY_INDEX, [0, 1], value="LEM SIEW YUEN")
        self.assertEqual([b["index"] for b in result], [1])

    def test_without_a_value_every_resolvable_citation_is_kept(self):
        # Unchanged behaviour when there is nothing to validate against.
        self.assertEqual([b["index"] for b in boxes.cited_boxes(self.BY_INDEX, [0, 1])], [0, 1])


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

    def test_prefix_fallback_does_not_apply_to_a_multi_word_value(self):
        # Seen on a real scan: an address starting "NO. 50, JALAN ..." matched
        # "Business Reg No. 201801002938" on its 4-char prefix "NO. " and boxed
        # the letterhead. The clipped-character case is about ONE word, so a
        # value with spaces must not use this path - no box beats a wrong one.
        ocr = [
            {"index": 0, "page": 0, "text": "Business Reg No. 201801002938 (1264951-W)"},
            {"index": 1, "page": 0, "text": "somewhere else entirely"},
        ]
        address = "NO. 50, JALAN IMPIAN EMAS 7, TAMAN IMPIAN EMAS, 81300 SKUDAI, JOHOR."
        self.assertIsNone(boxes.find_box_by_text(ocr, address))

    def test_multi_word_value_still_found_when_it_genuinely_appears(self):
        ocr = [{"index": 0, "page": 0, "text": ": NO. 50, JALAN IMPIAN EMAS 7"}]
        self.assertEqual(boxes.find_box_by_text(ocr, "NO. 50, JALAN IMPIAN EMAS 7")["index"], 0)

    def test_runs_of_whitespace_are_not_a_difference(self):
        # Forms pad with spaces to align a value against a printed rule:
        #   ': RM     4,400.00'   (five spaces)
        # while the model returns "RM 4,400.00" with one. Matching literally
        # loses the box for a value sitting right there.
        ocr = [{"index": 0, "page": 0, "text": ": RM     4,400.00"}]
        self.assertEqual(boxes.find_box_by_text(ocr, "RM 4,400.00")["index"], 0)

    def test_leading_and_trailing_space_is_ignored(self):
        ocr = [{"index": 0, "page": 0, "text": "   TAN SIEW KIAU   "}]
        self.assertEqual(boxes.find_box_by_text(ocr, "TAN SIEW KIAU")["index"], 0)

    def test_a_tab_counts_as_whitespace(self):
        ocr = [{"index": 0, "page": 0, "text": "Date:\t\t2026-03-19"}]
        self.assertEqual(boxes.find_box_by_text(ocr, "Date: 2026-03-19")["index"], 0)

    def test_whitespace_collapsing_does_not_join_separate_words(self):
        # "RM 4,400.00" must not match a line that merely has RM and 4,400.00
        # in unrelated places.
        ocr = [{"index": 0, "page": 0, "text": "RM is the currency; unrelated 9,999.00"}]
        self.assertIsNone(boxes.find_box_by_text(ocr, "RM 4,400.00"))

    def test_wrapped_value_matches_the_line_sharing_most_words(self):
        # A long address the model returns whole but OCR split across lines:
        # no single line contains all of it, so fall back to the line that
        # shares the most words rather than giving up.
        ocr = [
            {"index": 0, "page": 0, "text": "Business Reg No. 201801002938 (1264951-W)"},
            {"index": 1, "page": 0, "text": "NO. 50, JALAN IMPIAN EMAS 7, TAMAN IMPIAN EMAS,"},
            {"index": 2, "page": 0, "text": "81300 SKUDAI, JOHOR."},
        ]
        address = "NO. 50, JALAN IMPIAN EMAS 7, TAMAN IMPIAN EMAS, 81300 SKUDAI, JOHOR."
        self.assertEqual(boxes.find_box_by_text(ocr, address)["index"], 1)

    def test_one_shared_word_is_not_enough(self):
        # The letterhead shares only "NO" with the address - a single common
        # word is coincidence, not a location.
        ocr = [{"index": 0, "page": 0, "text": "Business Reg No. 201801002938 (1264951-W)"}]
        address = "NO. 50, JALAN IMPIAN EMAS 7, TAMAN IMPIAN EMAS, 81300 SKUDAI, JOHOR."
        self.assertIsNone(boxes.find_box_by_text(ocr, address))

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


class TestFlatNameByGroupPath(unittest.TestCase):
    """The legend used to print one line twice - "agent.Name · Negotiator/
    Agent Name" - because the raw reply nests a party's name under its group
    while the table names it flat, and both passes box the same line."""

    SOURCES = {
        "Reference No.": ["field", "Reference No."],
        "Property Address": ["group", "property", "Address"],
        "Negotiator/Agent Name": ["group", "agent", "Name"],
        "Landlord NRIC/Passport No.": ["group", "landlord", "NRIC/Passport No."],
    }

    def test_a_group_path_maps_to_its_flat_name(self):
        names = boxes.flat_name_by_group_path(self.SOURCES)
        self.assertEqual(names["agent.Name"], "Negotiator/Agent Name")
        self.assertEqual(names["property.Address"], "Property Address")

    def test_a_sub_field_name_with_a_slash_still_keys_correctly(self):
        names = boxes.flat_name_by_group_path(self.SOURCES)
        self.assertEqual(names["landlord.NRIC/Passport No."],
                         "Landlord NRIC/Passport No.")

    def test_a_top_level_field_needs_no_translation(self):
        # It is already the name the table shows.
        self.assertNotIn("Reference No.", boxes.flat_name_by_group_path(self.SOURCES))

    def test_nothing_published_is_an_empty_map(self):
        # An older engine, or a run whose result predates the field_sources
        # key: the legend falls back to readable_group_path rather than
        # breaking.
        self.assertEqual(boxes.flat_name_by_group_path(None), {})
        self.assertEqual(boxes.flat_name_by_group_path({}), {})

    def test_a_malformed_source_is_skipped(self):
        self.assertEqual(boxes.flat_name_by_group_path({"X": "group", "Y": ["group"]}), {})

    def test_tuples_work_as_well_as_lists(self):
        # field_sources() returns tuples; JSON turns them into lists.
        names = boxes.flat_name_by_group_path({"Property Address": ("group", "property", "Address")})
        self.assertEqual(names["property.Address"], "Property Address")


class TestGroupDisplayName(unittest.TestCase):
    """A group sub-field that reaches no checkbox and no table row must not be
    highlighted - it was numbered and legended for something the reader never
    asked for and cannot switch off."""

    FLAT = {
        "landlord.Name": "Landlord Name(s)",
        "landlord.Address": "Landlord Address",
        "witness.Name": "Witnessed by",
    }

    def test_a_sub_field_that_feeds_a_row_is_shown_by_that_name(self):
        self.assertEqual(boxes.group_display_name("landlord", "Name", self.FLAT),
                         "Landlord Name(s)")

    def test_a_sub_field_that_feeds_nothing_is_not_shown(self):
        # A witness's NRIC: extracted to tell one signature block from
        # another, offered by no picker, shown in no row.
        self.assertIsNone(
            boxes.group_display_name("witness", "NRIC/Passport No.", self.FLAT))

    def test_a_sub_field_of_an_unknown_group_is_not_shown_either(self):
        self.assertIsNone(boxes.group_display_name("guarantor", "Name", self.FLAT))

    def test_no_mapping_at_all_falls_back_to_showing_everything(self):
        # An older result published no field_sources; with no mapping there is
        # no basis to call anything unoffered, so nothing is hidden.
        self.assertEqual(boxes.group_display_name("witness", "NRIC/Passport No.", {}),
                         "Witness NRIC/Passport No.")
        self.assertEqual(boxes.group_display_name("witness", "NRIC/Passport No.", None),
                         "Witness NRIC/Passport No.")


class TestReadableGroupPath(unittest.TestCase):
    def test_a_sub_field_with_no_flat_name_reads_as_words(self):
        # Extracted, but never in the table, so nothing renames it.
        self.assertEqual(boxes.readable_group_path("witness", "Signature Date"),
                         "Witness Signature Date")

    def test_the_sub_fields_own_capitalisation_is_kept(self):
        self.assertEqual(boxes.readable_group_path("tenant", "NRIC/Passport No."),
                         "Tenant NRIC/Passport No.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
