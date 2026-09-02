"""Tests for narrowing the schema to the fields the user picked.

    .venv/bin/python3 test_field_selection.py

The point of excluding a field is that it never reaches the model - not that
its answer is thrown away afterwards. An unasked field costs no prompt tokens,
no generation time, and (for the two choice fields) no geometric detector pass.
So most of what is pinned here is ABSENCE: from the prompt, from the
constrained schema, from the flat result.

The other half is the contract with the UI. field_sources() is the engine's
answer to "what can be extracted"; app/fields.py builds the picker from it and
hides table rows by the same names. If those two lists ever disagree, the
failure is silent - a picker entry that switches nothing off, or a row that
can never be filled - so the first test below ties the published names to the
keys flatten_fields actually produces.
"""

import unittest

import extractor as E


def leaf(value):
    return {"value": value, "confidence": 1.0, "source_index": 3}


def full_raw():
    """A reply covering every field in the schema, in the documented shape."""
    fields = {f["field"]: leaf(f"v:{f['field']}") for f in E.get_field_schema()}
    groups = {}
    for name, spec in E.get_group_schema().items():
        occurrence = {f["field"]: leaf(f"v:{name}.{f['field']}") for f in spec["fields"]}
        groups[name] = [occurrence] if spec["multiple"] else occurrence
    return {"fields": fields, "groups": groups}


class TestFieldSources(unittest.TestCase):
    def test_published_names_are_exactly_what_a_run_produces(self):
        # The invariant the picker depends on. A name here that flatten_fields
        # never emits is a checkbox that switches nothing off; a key it emits
        # that isn't here is a field with no way to switch it off at all.
        self.assertEqual(sorted(E.field_sources()),
                         sorted(E.flatten_fields(full_raw())))

    def test_a_top_level_field_reports_itself_as_its_source(self):
        self.assertEqual(E.field_sources()["Monthly Rental"], ("field", "Monthly Rental"))

    def test_a_lifted_field_reports_the_group_it_came_from(self):
        self.assertEqual(E.field_sources()["Property Address"],
                         ("group", "property", "Address"))

    def test_sub_fields_that_never_reach_the_flat_output_are_not_offered(self):
        # A landlord's designation and signature date stay in _raw for the
        # box overlay; no table shows them, so no picker should either.
        self.assertNotIn("Designation (if any)", E.field_sources())
        self.assertNotIn("Signature Date", E.field_sources())

    def test_selectable_fields_is_the_published_order(self):
        self.assertEqual(E.selectable_fields(), list(E.field_sources()))


class TestFieldSchemaFor(unittest.TestCase):
    def test_nothing_excluded_is_the_whole_schema(self):
        self.assertEqual(E.field_schema_for(), E.get_field_schema())

    def test_an_excluded_field_is_gone(self):
        kept = [f["field"] for f in E.field_schema_for(["Monthly Rental"])]
        self.assertNotIn("Monthly Rental", kept)
        self.assertIn("Total Payable", kept)

    def test_a_name_the_schema_doesnt_have_is_ignored(self):
        # A stale name in a caller's list must narrow nothing, not raise -
        # the run is worth more than the tidiness of the list.
        self.assertEqual(E.field_schema_for(["Nonexistent Field"]), E.get_field_schema())

    def test_a_group_field_name_doesnt_touch_the_flat_schema(self):
        self.assertEqual(E.field_schema_for(["Property Address"]), E.get_field_schema())


class TestGroupSchemaFor(unittest.TestCase):
    def test_nothing_excluded_is_the_whole_schema(self):
        self.assertEqual(E.group_schema_for(), E.get_group_schema())

    def test_a_sub_field_is_dropped_by_its_flattened_name(self):
        # The picker offers "Property Address"; the schema calls it "Address".
        property_fields = [f["field"] for f in E.group_schema_for(["Property Address"])["property"]["fields"]]
        self.assertNotIn("Address", property_fields)
        self.assertIn("Title Details", property_fields)

    def test_the_group_survives_while_one_offered_field_remains(self):
        kept = E.group_schema_for(["Landlord Name(s)"])
        self.assertIn("landlord", kept)

    def test_the_group_goes_when_every_offered_field_is_excluded(self):
        # Asking for a landlord object whose name and NRIC are both switched
        # off would spend prompt tokens on an answer nothing reads.
        kept = E.group_schema_for(["Landlord Name(s)", "Landlord NRIC/Passport No."])
        self.assertNotIn("landlord", kept)
        self.assertIn("tenant", kept)

    def test_unoffered_sub_fields_ride_along_with_a_surviving_group(self):
        fields = [f["field"] for f in E.group_schema_for(["Landlord Name(s)"])["landlord"]["fields"]]
        self.assertNotIn("Name", fields)
        self.assertIn("Designation (if any)", fields)

    def test_the_group_keeps_its_description_and_multiple_flag(self):
        kept = E.group_schema_for(["Tenant Address"])["tenant"]
        self.assertEqual(kept["multiple"], E.get_group_schema()["tenant"]["multiple"])
        self.assertEqual(kept["description"], E.get_group_schema()["tenant"]["description"])

    def test_excluding_everything_leaves_no_groups(self):
        self.assertEqual(E.group_schema_for(E.selectable_fields()), {})


class TestExcludedFieldsLeaveThePrompt(unittest.TestCase):
    """The reason for all of the above: an excluded field is not asked for."""

    def _grounded(self, exclude):
        import pandas as pd
        ocr_df = pd.DataFrame([{"index": 1, "text": "SOMETHING", "page": 0,
                                "confidence": 0.99,
                                "x0": 0, "y0": 0, "x1": 1, "y1": 1}])
        return E.build_vision_grounded_prompt(
            E.field_schema_for(exclude), E.group_schema_for(exclude), ocr_df, 1)

    def test_a_top_level_field_is_absent_from_the_grounded_prompt(self):
        prompt = self._grounded(["Monthly Rental"])
        self.assertNotIn("Monthly Rental", prompt)
        self.assertIn("Total Payable", prompt)

    def test_a_group_sub_field_is_absent_too(self):
        prompt = self._grounded(["Title Details"])
        self.assertNotIn("Title Details", prompt)

    def test_a_fully_excluded_group_leaves_the_key_list(self):
        # The prompt's closing line enumerates the group keys; a group that
        # isn't described above must not still be demanded there.
        prompt = self._grounded(["Witnessed by"])
        self.assertNotIn('"witness"', prompt)

    def test_the_vision_only_prompt_drops_it_as_well(self):
        prompt = E.build_vision_only_prompt(
            E.field_schema_for(["Earnest Deposit Amount"]),
            E.group_schema_for(["Earnest Deposit Amount"]), 1)
        self.assertNotIn("Earnest Deposit Amount", prompt)


class TestFlattenHonoursTheExclusion(unittest.TestCase):
    def test_an_excluded_field_has_no_key(self):
        flat = E.flatten_fields(full_raw(), exclude=["Monthly Rental"])
        self.assertNotIn("Monthly Rental", flat)
        self.assertIn("Total Payable", flat)

    def test_an_excluded_group_field_has_no_key(self):
        flat = E.flatten_fields(full_raw(), exclude=["Landlord Name(s)"])
        self.assertNotIn("Landlord Name(s)", flat)
        self.assertIn("Landlord NRIC/Passport No.", flat)

    def test_a_key_the_model_echoed_anyway_is_still_dropped(self):
        # Nothing asked for it, but a model can return a key it was never
        # given - and one that slipped through would render a row the user
        # switched off.
        raw = full_raw()
        raw["fields"]["Monthly Rental"] = leaf("RM 2,800")
        self.assertNotIn("Monthly Rental",
                         E.flatten_fields(raw, exclude=["Monthly Rental"]))

    def test_excluding_nothing_is_the_old_behaviour(self):
        self.assertEqual(E.flatten_fields(full_raw()),
                         E.flatten_fields(full_raw(), exclude=[]))


class TestChoiceFieldsRespectTheExclusion(unittest.TestCase):
    def test_an_excluded_choice_field_is_not_published(self):
        self.assertNotIn("Furnished Status", E.choice_candidates(["Furnished Status"]))

    def test_the_other_choice_field_still_is(self):
        published = E.choice_candidates(["Furnished Status"])
        self.assertIn("Description/Property Type", published)

    def test_both_are_published_when_nothing_is_excluded(self):
        self.assertEqual(sorted(E.choice_candidates()),
                         sorted(E.CHOICE_FIELD_TARGETS))


class TestConstrainedSchemaMatchesThePrompt(unittest.TestCase):
    """Constrained decoding has to be narrowed with everything else: a JSON
    Schema still listing the excluded field would oblige the model to emit a
    key the prompt never described."""

    def test_an_excluded_field_is_not_in_the_json_schema(self):
        import json
        import schema_models
        schema = schema_models.extraction_json_schema(
            fields=E.field_schema_for(["Monthly Rental"]),
            groups=E.group_schema_for(["Monthly Rental"]))
        self.assertNotIn("Monthly Rental", json.dumps(schema))

    def test_the_default_schema_is_unchanged(self):
        import schema_models
        self.assertEqual(schema_models.extraction_json_schema(),
                         schema_models.extraction_json_schema(fields=None, groups=None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
