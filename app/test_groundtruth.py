"""Tests for groundtruth.py's mapping of a CRM sale record onto the flat
field names the extractor produces.

Stdlib unittest, run with:

    .venv/bin/python3 test_groundtruth.py

The party lambdas are the fragile part: they read nested API payloads whose
optional keys are genuinely sometimes absent, and any exception they raise
surfaces to the user as a bare "Extraction failed: 'nric'" after a full
extraction has already been paid for.
"""

import unittest

import groundtruth as G


def sale(**overrides):
    """A minimal sale record in the shape CANONICAL_API_MAP expects."""
    base = {
        "lof_no": "ERHQ00008180",
        "date_of_lof": "2026-03-31",
        "total_deposit": "7750.00",
        "listing_negotiator": {"name": "CHAN KAR MUN"},
        "property": {"type": "condo", "address": "A-34-08, JALAN DUA"},
        "deposits": [{"deposit_date": "2026-03-31", "amount": "2200"}],
        "amla_purpose": {
            "purchasers_or_tenants": [{"name": "LEM SIEW YUEN", "nric": "960314105840"}],
            "vendors_or_landlords": [{"name": "MAH CHEE WENG", "nric": "931014145303"}],
        },
    }
    base.update(overrides)
    return base


class TestLandlordParties(unittest.TestCase):
    """A landlord record missing an optional key must not take down the whole
    comparison - the tenant lambdas next to it already use .get()."""

    def _landlords(self, parties):
        return sale(amla_purpose={"purchasers_or_tenants": [],
                                  "vendors_or_landlords": parties})

    def test_joins_two_co_landlords(self):
        truth = G.build_canonical_ground_truth(self._landlords(
            [{"name": "ALICE TAN", "nric": "1"}, {"name": "BRIAN LIM", "nric": "2"}]))
        self.assertEqual(truth["Landlord Name"], "ALICE TAN / BRIAN LIM")
        self.assertEqual(truth["Landlord IC"], "1 / 2")

    def test_missing_nric_key_does_not_raise(self):
        truth = G.build_canonical_ground_truth(self._landlords([{"name": "ALICE TAN"}]))
        self.assertEqual(truth["Landlord Name"], "ALICE TAN")
        self.assertIsNone(truth["Landlord IC"])

    def test_missing_name_key_does_not_raise(self):
        truth = G.build_canonical_ground_truth(self._landlords([{"nric": "1"}]))
        self.assertIsNone(truth["Landlord Name"])
        self.assertEqual(truth["Landlord IC"], "1")

    def test_blank_values_are_skipped_not_joined_as_empty(self):
        # " / " with nothing either side reads as a value; it isn't one.
        truth = G.build_canonical_ground_truth(self._landlords(
            [{"name": "ALICE TAN", "nric": "1"}, {"name": "", "nric": None}]))
        self.assertEqual(truth["Landlord Name"], "ALICE TAN")

    def test_no_landlords_is_none(self):
        truth = G.build_canonical_ground_truth(self._landlords([]))
        self.assertIsNone(truth["Landlord Name"])


class TestCanonicalMap(unittest.TestCase):
    def test_every_canonical_field_has_a_mapping(self):
        self.assertEqual(set(G.CANONICAL_FIELDS), set(G.CANONICAL_API_MAP))

    def test_unverifiable_fields_resolve_to_none(self):
        # The API carries no itemized deposit breakdown; these must be shown
        # as "not verifiable" rather than as a false mismatch.
        truth = G.build_canonical_ground_truth(sale())
        for field in ["Furnish Status", "Security Deposit", "Utilities Deposit",
                      "Advance Rental", "Others (if any)"]:
            self.assertIsNone(truth[field], f"{field} should be unverifiable")

    def test_reads_the_earliest_deposit(self):
        truth = G.build_canonical_ground_truth(sale(deposits=[
            {"deposit_date": "2026-05-01", "amount": "999"},
            {"deposit_date": "2026-03-31", "amount": "2200"},
        ]))
        self.assertEqual(truth["Earnest Deposit Amount"], "2200")


class TestValuesMatch(unittest.TestCase):
    def test_blank_against_a_value_is_not_a_match(self):
        # "" in "anything" is True in Python - the guard that stops a missing
        # extraction scoring as correct.
        self.assertFalse(G.values_match("", "ALICE TAN"))
        self.assertFalse(G.values_match(None, "ALICE TAN"))

    def test_both_blank_is_a_match(self):
        self.assertTrue(G.values_match("", None))


class TestChoiceMatch(unittest.TestCase):
    def test_requires_exact_equality(self):
        # The unselected option list contains every candidate as a substring,
        # so substring credit would score any answer as correct.
        self.assertFalse(G.choice_match("House / Condo / Land", "Condo"))
        self.assertTrue(G.choice_match("Condo", "condo"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
