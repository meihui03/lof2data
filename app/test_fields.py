"""Tests for fields.py - laying the engine's field list out for the picker,
and keeping removed fields out of the comparison table.

    .venv/bin/python3 test_fields.py

Pure logic, deliberately kept out of app.py: importing app.py executes the
whole Streamlit script, so anything living there can only be tested through
AppTest, which is far heavier than these rules deserve.
"""

import unittest

import fields as F

ALL = [
    "Reference No.", "Date", "Tenancy Commencement Date",
    "Tenancy Expiration Date", "Period of Tenancy", "Option to Renew",
    "Monthly Rental", "Security Deposit", "Utilities Deposit",
    "Advance Rental", "Other Deposit (if any)", "Total Payable",
    "Earnest Deposit Amount", "Legal Cost / Stamp Duty Allocation",
    "Special/Other Condition(s)", "Notice to Quit (months)",
    "Description of Property/Property Type", "Property Address",
    "Title Details", "Premise Usage", "Furnished Status",
    "Landlord Name(s)", "Landlord NRIC/Passport No.",
    "Tenant Name(s)/Company", "Tenant NRIC/Passport No.", "Tenant Address",
    "Negotiator/Agent Name", "Witnessed by",
]


class TestSectionedFields(unittest.TestCase):
    def test_every_field_is_placed_exactly_once(self):
        placed = [name for _, names in F.sectioned_fields(ALL) for name in names]
        self.assertEqual(sorted(placed), sorted(ALL))
        self.assertEqual(len(placed), len(set(placed)))

    def test_sections_come_out_in_declared_order(self):
        titles = [title for title, _ in F.sectioned_fields(ALL)]
        self.assertEqual(titles, [t for t, _ in F.FIELD_SECTIONS])

    def test_a_field_the_sections_dont_know_lands_under_other(self):
        # A field added to the engine's schema must still be pickable, or the
        # UI would silently offer no way to remove it.
        sections = dict(F.sectioned_fields(ALL + ["Brand New Field"]))
        self.assertEqual(sections["Other"], ["Brand New Field"])

    def test_no_other_section_when_everything_is_known(self):
        self.assertNotIn("Other", dict(F.sectioned_fields(ALL)))

    def test_a_section_with_nothing_present_is_dropped(self):
        sections = dict(F.sectioned_fields(["Monthly Rental"]))
        self.assertEqual(sections, {"Financial": ["Monthly Rental"]})


class TestRowIsHidden(unittest.TestCase):
    """The comparison table walks CANONICAL_FIELDS, not the extracted result,
    so without this a removed field still renders a row - with nothing in it,
    scored as a miss. That would make removing a field LOOK like breaking it,
    and drag the accuracy figure down with it."""

    def test_nothing_is_hidden_when_nothing_was_removed(self):
        self.assertFalse(F.row_is_hidden(["Monthly Rental"], excluded=set()))

    def test_a_row_whose_only_field_was_removed_is_hidden(self):
        self.assertTrue(F.row_is_hidden(["Monthly Rental"],
                                        excluded={"Monthly Rental"}))

    def test_a_row_keeps_showing_while_one_alias_survives(self):
        # A canonical field can map to several engine fields; the row is still
        # answerable while any of them is being extracted.
        self.assertFalse(F.row_is_hidden(["Total Payable", "Monthly Rental"],
                                         excluded={"Monthly Rental"}))

    def test_a_row_with_no_aliases_is_never_hidden(self):
        self.assertFalse(F.row_is_hidden([], excluded={"Monthly Rental"}))


class TestExtractableAliases(unittest.TestCase):
    """Some alias lists carry names from a sibling extractor's schema, which
    this engine can never produce. Left in the list they pin a row open
    forever: row_is_hidden waits for every alias to be excluded, and an alias
    the picker never offers can never be excluded."""

    def test_an_alias_the_engine_doesnt_have_is_dropped(self):
        self.assertEqual(
            F.extractable_aliases(["Tenant Name(s)/Company", "Purchaser Name(s)"], ALL),
            ["Tenant Name(s)/Company"])

    def test_so_that_row_hides_when_its_real_field_is_unticked(self):
        aliases = F.extractable_aliases(["Tenant Name(s)/Company", "Purchaser Name(s)"], ALL)
        self.assertTrue(F.row_is_hidden(aliases, excluded={"Tenant Name(s)/Company"}))

    def test_a_row_with_no_recognized_alias_is_left_alone(self):
        # Nothing can be reasoned about it, and showing it is the safe way to
        # be wrong.
        self.assertEqual(F.extractable_aliases(["Purchaser Name(s)"], ALL),
                         ["Purchaser Name(s)"])

    def test_all_recognized_aliases_survive(self):
        self.assertEqual(F.extractable_aliases(["Total Payable", "Monthly Rental"], ALL),
                         ["Total Payable", "Monthly Rental"])

    def test_an_empty_field_list_changes_nothing(self):
        # Engine unreachable: the picker isn't shown, so nothing is excluded
        # and every row should still render.
        self.assertEqual(F.extractable_aliases(["Monthly Rental"], []), ["Monthly Rental"])


class TestExcludedFrom(unittest.TestCase):
    def test_unticked_fields_are_the_excluded_ones(self):
        self.assertEqual(
            F.excluded_from(ALL, {"Monthly Rental": False, "Date": True}),
            ["Monthly Rental"])

    def test_a_field_with_no_recorded_flag_counts_as_kept(self):
        # First render, before any checkbox has been touched.
        self.assertEqual(F.excluded_from(ALL, {}), [])

    def test_order_follows_the_engine_list_not_the_flags(self):
        flags = {"Witnessed by": False, "Date": False}
        self.assertEqual(F.excluded_from(ALL, flags), ["Date", "Witnessed by"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
