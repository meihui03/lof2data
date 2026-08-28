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
