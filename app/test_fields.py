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
    "Landlord Name(s)", "Landlord NRIC/Passport No.", "Landlord Address",
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


# app.py's alias map, values only - the shape picker_units is given.
ALIAS_GROUPS = [
    ["Negotiator/Agent Name", "Witnessed by"],
    ["Tenant Name(s)/Company", "Purchaser Name(s)"],   # sibling schema name
    ["Landlord Name(s)", "Vendor Name"],               # sibling schema name
    ["Monthly Rental"],
]


class TestPickerUnits(unittest.TestCase):
    """Two fields that answer ONE comparison row have to be one tick.

    A real document (73009) carries no "Attn:" line at all and names the
    negotiator only under "Witnessed by". The extractor is told to return no
    agent rather than promote a witness into the role, so on that form the
    witness IS the answer to the "Attn: Negotiator(s) / Witness" row. Keeping
    the negotiator while dropping the witness left the row blank and scored it
    a miss, with the answer sitting on the page unread."""

    def units(self, fields=None):
        return F.picker_units(fields if fields is not None else ALL, ALIAS_GROUPS)

    def test_fields_answering_one_row_become_one_unit(self):
        pairs = [u for u in self.units() if len(u[2]) > 1]
        self.assertEqual(len(pairs), 1)
        key, label, members = pairs[0]
        self.assertEqual(members, ["Negotiator/Agent Name", "Witnessed by"])
        self.assertEqual(label, "Negotiator/Agent Name · Witnessed by")
        self.assertEqual(key, "Negotiator/Agent Name")

    def test_every_field_is_offered_exactly_once(self):
        offered = [name for _, _, members in self.units() for name in members]
        self.assertEqual(sorted(offered), sorted(ALL))
        self.assertEqual(len(offered), len(set(offered)))

    def test_an_alias_the_engine_doesnt_offer_never_forms_a_unit(self):
        # "Purchaser Name(s)" belongs to a sibling extractor's schema; it must
        # not appear in the picker, nor pull Tenant Name into a pair.
        tenant = [u for u in self.units() if u[0] == "Tenant Name(s)/Company"]
        self.assertEqual(tenant[0][2], ["Tenant Name(s)/Company"])

    def test_units_follow_the_engines_order(self):
        keys = [key for key, _, _ in self.units()]
        self.assertEqual(keys, sorted(keys, key=ALL.index))

    def test_a_unit_whose_partner_is_missing_stands_alone(self):
        # The engine dropped a field: the survivor is still pickable.
        fields = [n for n in ALL if n != "Witnessed by"]
        unit = [u for u in self.units(fields) if u[0] == "Negotiator/Agent Name"][0]
        self.assertEqual(unit[2], ["Negotiator/Agent Name"])

    def test_no_alias_groups_means_one_unit_per_field(self):
        self.assertEqual(len(F.picker_units(ALL, [])), len(ALL))
        self.assertEqual(len(F.picker_units(ALL, None)), len(ALL))


class TestSectionedUnits(unittest.TestCase):
    def test_a_paired_unit_appears_once_in_its_section(self):
        sections = dict(F.sectioned_units(ALL, ALIAS_GROUPS))
        labels = [label for _, label, _ in sections["Parties"]]
        self.assertIn("Negotiator/Agent Name · Witnessed by", labels)
        self.assertNotIn("Witnessed by", labels)

    def test_every_field_still_reaches_a_section(self):
        sections = F.sectioned_units(ALL, ALIAS_GROUPS)
        offered = [n for _, section in sections for _, _, members in section for n in members]
        self.assertEqual(sorted(offered), sorted(ALL))

    def test_sections_keep_their_declared_order(self):
        titles = [title for title, _ in F.sectioned_units(ALL, ALIAS_GROUPS)]
        self.assertEqual(titles, [t for t, _ in F.FIELD_SECTIONS])


class TestExcludedFromUnits(unittest.TestCase):
    def test_dropping_a_pair_drops_both_its_fields(self):
        units = F.picker_units(ALL, ALIAS_GROUPS)
        excluded = F.excluded_from_units(ALL, units, {"Negotiator/Agent Name": False})
        self.assertEqual(excluded, ["Negotiator/Agent Name", "Witnessed by"])

    def test_keeping_a_pair_keeps_both_its_fields(self):
        # The bug this exists for: the witness must not be droppable while the
        # negotiator is kept, since either can answer the row.
        units = F.picker_units(ALL, ALIAS_GROUPS)
        self.assertEqual(F.excluded_from_units(ALL, units, {}), [])

    def test_a_single_field_unit_behaves_as_before(self):
        units = F.picker_units(ALL, ALIAS_GROUPS)
        self.assertEqual(F.excluded_from_units(ALL, units, {"Monthly Rental": False}),
                         ["Monthly Rental"])

    def test_order_follows_the_engine_list(self):
        units = F.picker_units(ALL, ALIAS_GROUPS)
        flags = {"Witnessed by": False, "Date": False, "Negotiator/Agent Name": False}
        self.assertEqual(F.excluded_from_units(ALL, units, flags),
                         ["Date", "Negotiator/Agent Name", "Witnessed by"])


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
