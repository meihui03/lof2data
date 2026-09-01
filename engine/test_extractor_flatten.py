"""Tests for join_occurrences(), the flattening of a repeatable role
(landlord/tenant/agent/witness) into one flat field.

Stdlib unittest so it runs on the extraction venv with no extra dependency:

    .venv/bin/python3 test_extractor_flatten.py

This exists because the previous behaviour - keep occurrences[0], drop the
rest - silently lost the second of two co-landlords on a real form, and the
comparison table's substring matcher scored the truncated answer as correct,
so nothing surfaced it. The join has to survive the shapes models actually
return, not just the shape the prompt asks for.
"""

import unittest

import extractor as E


def leaf(value):
    """A sub-field in the documented shape."""
    return {"value": value, "confidence": 1.0, "source_index": 3}


class TestFieldLine(unittest.TestCase):
    """A regression guard, not a feature test.

    Putting the schema's declared type into the prompt line looks like free
    accuracy and is not. Measured on the local model, vision-only, file.pdf:
    26/28 without the annotation, 19/28 with "(number)" - every numeric field
    came back as an empty string. The model answered with silence rather than
    a value it could not render as a bare number.

    These tests exist so that gets re-derived from the measurement instead of
    from the idea, which is appealing enough to try twice."""

    def test_type_is_not_put_in_the_prompt_line(self):
        line = E.format_field_line({"field": "Monthly Rental", "type": "number",
                                    "description": "The monthly rental amount."})
        self.assertEqual(line, "- Monthly Rental: The monthly rental amount.")
        self.assertNotIn("(number)", line)

    def test_categorical_type_is_not_stated_either(self):
        line = E.format_field_line({"field": "Furnished Status", "type": "categorical",
                                    "description": "Not / Partly / Fully."})
        self.assertNotIn("(categorical)", line)

    def test_missing_type_is_harmless(self):
        self.assertEqual(E.format_field_line({"field": "X", "description": "d"}), "- X: d")


class TestPromptsCarryNoTypes(unittest.TestCase):
    def test_vision_grounded_prompt_has_no_type_annotation(self):
        import pandas as pd
        ocr = pd.DataFrame([{"index": 0, "page": 0, "text": "x", "confidence": 1.0}])
        prompt = E.build_vision_grounded_prompt(
            E.get_field_schema(), E.get_group_schema(), ocr, 1)
        self.assertIn("Monthly Rental:", prompt)
        self.assertNotIn("(number)", prompt)

    def test_vision_only_prompt_has_no_type_annotation(self):
        prompt = E.build_vision_only_prompt(
            E.get_field_schema(), E.get_group_schema(), 1)
        self.assertIn("Monthly Rental:", prompt)
        self.assertNotIn("(number)", prompt)


class TestChatPayload(unittest.TestCase):
    """num_ctx was set only when images were attached, so a text-only call ran
    at Ollama's default context. With the document's text in the prompt that
    overflows and the reply is truncated mid-JSON - deterministically, at the
    same character every retry."""

    def test_text_only_call_still_raises_the_context_window(self):
        payload = E.build_chat_payload("prompt", model="m", images=None, json_mode=True)
        self.assertEqual(payload["options"]["num_ctx"], 16384)

    def test_vision_call_raises_it_too(self):
        payload = E.build_chat_payload("p", model="m", images=["b64"], json_mode=True)
        self.assertEqual(payload["options"]["num_ctx"], 16384)

    def test_images_are_attached_to_the_message(self):
        payload = E.build_chat_payload("p", model="m", images=["b64"], json_mode=True)
        self.assertEqual(payload["messages"][0]["images"], ["b64"])

    def test_no_images_key_when_there_are_none(self):
        payload = E.build_chat_payload("p", model="m", images=None, json_mode=True)
        self.assertNotIn("images", payload["messages"][0])

    def test_json_mode_is_optional(self):
        self.assertEqual(
            E.build_chat_payload("p", model="m", images=None, json_mode=True)["format"], "json")
        self.assertNotIn(
            "format", E.build_chat_payload("p", model="m", images=None, json_mode=False))


class TestVisionOnlyPrompt(unittest.TestCase):
    """Arm A's prompt: the same schema and output contract as the grounded
    one, minus the OCR text block and the source_index citation (which means
    nothing when there is no OCR index to cite). Measured at 92.0% on cloud
    vs 94.6% grounded, for a third of the wall clock."""

    def build(self, num_pages=1):
        return E.build_vision_only_prompt(
            E.get_field_schema(), E.get_group_schema(), num_pages)

    def test_asks_for_no_source_index(self):
        self.assertNotIn("source_index", self.build())

    def test_says_the_image_is_the_only_source(self):
        self.assertIn("only source of truth", self.build())

    def test_covers_every_top_level_field(self):
        prompt = self.build()
        for f in E.get_field_schema():
            self.assertIn(f["field"], prompt)

    def test_covers_every_group(self):
        prompt = self.build()
        for name in E.get_group_schema():
            self.assertIn(name, prompt)

    def test_multi_page_note_only_when_multi_page(self):
        self.assertNotIn("pages of the same form", self.build(num_pages=1))
        self.assertIn("all 3 pages", self.build(num_pages=3))


class TestUngroundedBoxLines(unittest.TestCase):
    """Turning grounding off should not cost the source boxes on a document
    where the text is free to obtain.

    The toggle exists to skip the thousands of prompt tokens the text costs -
    not to skip the text itself. A born-digital PDF's text layer is read in
    milliseconds and carries exact coordinates, so the boxes can be kept at no
    price. A scan would need a full OCR pass, which is the cost being avoided,
    so there the boxes genuinely go."""

    DIGITAL = "/Users/meihui/propnex_ai/hybrid/input/72979.pdf"
    SCAN = "/Users/meihui/propnex_ai/hybrid/input/72834.pdf"

    def test_digital_pdf_still_gets_box_lines(self):
        lines = E.ungrounded_box_lines(self.DIGITAL)
        self.assertTrue(lines, "a born-digital PDF should still yield box lines")
        first = lines[0]
        self.assertEqual(set(first) & {"index", "page", "text", "x0", "y0", "x1", "y1"},
                         {"index", "page", "text", "x0", "y0", "x1", "y1"})

    def test_scan_yields_nothing_and_never_runs_ocr(self):
        # Patched to explode: reaching PaddleOCR here would mean paying the
        # exact cost the toggle was turned off to avoid.
        original = E.run_ocr
        E.run_ocr = lambda *a, **k: self.fail("ran OCR on a scan with grounding off")
        try:
            self.assertEqual(E.ungrounded_box_lines(self.SCAN), [])
        finally:
            E.run_ocr = original


class TestGroundingIsCheap(unittest.TestCase):
    """Grounding costs milliseconds on a born-digital PDF (its own text layer)
    but 100s+ on a scan (a full PaddleOCR pass). The toggle only has anything
    to save in the second case."""

    def test_native_text_layer_is_cheap(self):
        self.assertTrue(E.grounding_is_cheap(has_text_layer=True))

    def test_scan_is_not_cheap(self):
        self.assertFalse(E.grounding_is_cheap(has_text_layer=False))


class TestChoiceCandidates(unittest.TestCase):
    """The option lists the UI needs to locate a choice field's printed line
    on the page. Published from the schema rather than restated in the UI, so
    the prompt, the geometric detector and the box lookup can never disagree
    about what the legal options are."""

    def test_publishes_both_choice_fields(self):
        self.assertEqual(set(E.choice_candidates()), set(E.CHOICE_FIELD_TARGETS))

    def test_furnished_status_options(self):
        self.assertEqual(E.choice_candidates()["Furnished Status"],
                         ["NOT", "PARTLY", "FULLY"])

    def test_property_type_options_come_from_the_schema(self):
        schema = {f["field"]: f for spec in E.get_group_schema().values()
                  for f in spec["fields"]}
        self.assertEqual(E.choice_candidates()["Description/Property Type"],
                         schema["Description/Property Type"]["candidates"])

    def test_every_published_field_has_a_non_empty_list(self):
        for field, candidates in E.choice_candidates().items():
            self.assertTrue(candidates, f"{field} published an empty option list")


class TestJoinOccurrences(unittest.TestCase):
    def test_joins_two_co_landlords(self):
        occurrences = [{"Name": leaf("ALICE TAN")}, {"Name": leaf("BRIAN LIM")}]
        self.assertEqual(E.join_occurrences(occurrences, "Name"),
                         "ALICE TAN / BRIAN LIM")

    def test_single_occurrence_has_no_separator(self):
        self.assertEqual(E.join_occurrences([{"Name": leaf("ALICE TAN")}], "Name"),
                         "ALICE TAN")

    def test_preserves_document_order(self):
        occurrences = [{"Name": leaf("B")}, {"Name": leaf("A")}]
        self.assertEqual(E.join_occurrences(occurrences, "Name"), "B / A")

    def test_deduplicates_repeated_values(self):
        # One witness signing both the tenant's and the landlord's block is
        # two genuine occurrences of the same person - show them once.
        occurrences = [{"Name": leaf("CARA WONG")}, {"Name": leaf("CARA WONG")}]
        self.assertEqual(E.join_occurrences(occurrences, "Name"), "CARA WONG")

    def test_dedup_ignores_case_and_surrounding_space(self):
        occurrences = [{"Name": leaf("Cara Wong")}, {"Name": leaf("  CARA WONG ")}]
        self.assertEqual(E.join_occurrences(occurrences, "Name"), "Cara Wong")

    def test_skips_blank_and_missing_values(self):
        occurrences = [
            {"Name": leaf("ALICE TAN")},
            {"Name": leaf("")},
            {"Name": leaf(None)},
            {},  # sub-field absent entirely
            {"Name": leaf("   ")},
            {"Name": leaf("BRIAN LIM")},
        ]
        self.assertEqual(E.join_occurrences(occurrences, "Name"),
                         "ALICE TAN / BRIAN LIM")

    def test_empty_group_is_empty_string(self):
        # An absent role (e.g. a form with no "Attn:" line) must stay blank,
        # not become "None" or a stray separator.
        self.assertEqual(E.join_occurrences([], "Name"), "")

    def test_all_blank_is_empty_string(self):
        self.assertEqual(E.join_occurrences([{"Name": leaf(None)}], "Name"), "")

    def test_bare_string_leaves(self):
        # Observed on a real minimax-m3 run: the model returned each
        # sub-field as a bare string rather than the {value: ...} object the
        # prompt asks for. _safe_leaf absorbs that, and so must this.
        occurrences = [{"Name": "ALICE TAN"}, {"Name": "BRIAN LIM"}]
        self.assertEqual(E.join_occurrences(occurrences, "Name"),
                         "ALICE TAN / BRIAN LIM")

    def test_non_string_values_are_stringified(self):
        self.assertEqual(E.join_occurrences([{"No.": leaf(600710)}], "No."), "600710")

    def test_non_dict_occurrence_is_skipped(self):
        occurrences = [{"Name": leaf("ALICE TAN")}, None, "junk"]
        self.assertEqual(E.join_occurrences(occurrences, "Name"), "ALICE TAN")

    def test_none_group_is_empty_string(self):
        # raw["groups"].get(name) is None when the model omitted the key.
        self.assertEqual(E.join_occurrences(None, "Name"), "")

    def test_separator_matches_the_ground_truth_api(self):
        # api_ground_truth.py builds "Landlord Name" as
        #   " / ".join(p["name"] for p in _landlord_parties(s))
        # so this side has to use the same separator for the rows to compare.
        self.assertEqual(E.OCCURRENCE_SEPARATOR, " / ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
