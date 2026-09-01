"""Tests for schema_models.py - the Pydantic mirror of the extraction schema.

    .venv/bin/python3 test_schema_models.py

The models are GENERATED from get_field_schema()/get_group_schema() rather
than hand-written beside them, so there is one source of truth. A second
hand-maintained copy would drift, and a drifted field name breaks silently:
the CRM comparison and the ground-truth scorer both key on the exact display
name, so a rename produces empty rows rather than an error.

The display names are not Python identifiers ("Reference No.",
"Other Deposit (if any)", "NRIC/Passport No."), so every field needs an
alias, and the JSON schema handed to the model must use the display name -
that is the key the rest of the pipeline reads.
"""

import unittest

import extractor as E
import schema_models as S


class TestSafeName(unittest.TestCase):
    def test_makes_a_valid_python_identifier(self):
        for display in ["Reference No.", "Other Deposit (if any)",
                        "NRIC/Passport No.", "Legal Cost / Stamp Duty Allocation",
                        "Description/Property Type", "Notice to Quit (months)"]:
            name = S.safe_name(display)
            self.assertTrue(name.isidentifier(), f"{display!r} -> {name!r} is not an identifier")

    def test_does_not_start_with_a_digit(self):
        self.assertTrue(S.safe_name("2nd Landlord").isidentifier())

    def test_is_stable(self):
        self.assertEqual(S.safe_name("Reference No."), S.safe_name("Reference No."))

    def test_every_schema_field_gets_a_unique_name(self):
        displays = [f["field"] for f in E.get_field_schema()]
        names = [S.safe_name(d) for d in displays]
        self.assertEqual(len(set(names)), len(names), "safe_name collided across the schema")


class TestExtractionSchema(unittest.TestCase):
    def setUp(self):
        self.schema = S.extraction_json_schema()

    def test_top_level_has_fields_and_groups(self):
        self.assertEqual(set(self.schema["properties"]), {"fields", "groups"})

    def test_every_flat_field_appears_under_its_display_name(self):
        props = S.resolve(self.schema, self.schema["properties"]["fields"])["properties"]
        for f in E.get_field_schema():
            self.assertIn(f["field"], props, f"{f['field']!r} missing from the schema")

    def test_every_group_appears(self):
        props = S.resolve(self.schema, self.schema["properties"]["groups"])["properties"]
        self.assertEqual(set(props), set(E.get_group_schema()))

    def test_multiple_groups_are_arrays_and_property_is_an_object(self):
        groups = S.resolve(self.schema, self.schema["properties"]["groups"])["properties"]
        for name, spec in E.get_group_schema().items():
            resolved = S.resolve(self.schema, groups[name])
            kind = resolved.get("type")
            if spec["multiple"]:
                self.assertEqual(kind, "array", f"{name} should be an array")
            else:
                self.assertEqual(kind, "object", f"{name} should be an object")

    def test_descriptions_are_carried_through(self):
        props = S.resolve(self.schema, self.schema["properties"]["fields"])["properties"]
        first = E.get_field_schema()[0]
        self.assertIn("description", S.resolve(self.schema, props[first["field"]]))

    def test_a_leaf_has_value_confidence_and_source_index(self):
        props = S.resolve(self.schema, self.schema["properties"]["fields"])["properties"]
        leaf = S.resolve(self.schema, props["Date"])
        self.assertEqual(set(leaf["properties"]), {"value", "confidence", "source_index"})


class TestValidation(unittest.TestCase):
    def test_a_well_formed_payload_validates(self):
        model = S.build_extraction_model()
        payload = {
            "fields": {f["field"]: {"value": "x", "confidence": 1.0, "source_index": 0}
                       for f in E.get_field_schema()},
            "groups": {
                name: ([] if spec["multiple"] else
                       {f["field"]: {"value": "x", "confidence": 1.0, "source_index": None}
                        for f in spec["fields"]})
                for name, spec in E.get_group_schema().items()
            },
        }
        parsed = model.model_validate(payload)
        dumped = parsed.model_dump(by_alias=True)
        self.assertEqual(dumped["fields"]["Reference No."]["value"], "x")

    def test_round_trip_keeps_display_names(self):
        # The flattener and the scorer both key on the display name, so the
        # dumped shape has to match what the non-Pydantic path produces.
        model = S.build_extraction_model()
        payload = {
            "fields": {f["field"]: {"value": None, "confidence": None, "source_index": None}
                       for f in E.get_field_schema()},
            "groups": {name: ([] if spec["multiple"] else
                              {f["field"]: {"value": None, "confidence": None, "source_index": None}
                               for f in spec["fields"]})
                       for name, spec in E.get_group_schema().items()},
        }
        dumped = model.model_validate(payload).model_dump(by_alias=True)
        self.assertEqual(set(dumped["fields"]), {f["field"] for f in E.get_field_schema()})
        self.assertEqual(set(dumped["groups"]), set(E.get_group_schema()))


class TestPayloadCarriesSchema(unittest.TestCase):
    def test_schema_is_passed_as_format_when_given(self):
        schema = {"type": "object"}
        payload = E.build_chat_payload("p", model="m", images=None, json_mode=True, schema=schema)
        self.assertEqual(payload["format"], schema)

    def test_plain_json_mode_when_no_schema(self):
        payload = E.build_chat_payload("p", model="m", images=None, json_mode=True)
        self.assertEqual(payload["format"], "json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
