"""Tests for the comparison table's markup.

    .venv/bin/python3 test_table.py

Two things matter here and neither is cosmetic. A value must survive intact -
the whole reason this markup is hand-built is that Streamlit's grid clipped a
long address, and a truncated value cannot be checked against the record
beside it. And extracted text goes into a page as HTML, so it has to be
escaped: it comes from a model reading a document, not from anything this app
controls.
"""

import unittest

import table as T

ROWS = [
    {"Field": "Reference No.", "Ground Truth (API)": "206537 (31/03)",
     "Extracted": "206537 (31/03)", "Match": T.MATCH},
    {"Field": "Date of LOF", "Ground Truth (API)": "2026-03-31",
     "Extracted": "28/03/2026", "Match": T.MISS},
    {"Field": "Furnish Status", "Ground Truth (API)": T.NA,
     "Extracted": "Not*", "Match": T.NA},
]

LONG_ADDRESS = "NO. 50, JALAN IMPIAN EMAS 7, TAMAN IMPIAN EMAS, 81300 SKUDAI, JOHOR."


class TestStructure(unittest.TestCase):
    def test_every_column_appears_in_the_header(self):
        html = T.comparison_table_html(ROWS)
        for name in T.COLUMNS:
            self.assertIn(f"<th>{name}</th>", html)

    def test_one_row_per_input(self):
        self.assertEqual(T.comparison_table_html(ROWS).count("<tr>"), len(ROWS) + 1)

    def test_the_field_name_is_a_row_heading(self):
        # It labels the row rather than being one of its values - which is what
        # lets the stylesheet weight it differently from the data.
        self.assertIn('<th scope="row">Reference No.</th>',
                      T.comparison_table_html(ROWS))

    def test_no_rows_still_yields_a_valid_table(self):
        html = T.comparison_table_html([])
        self.assertIn("<tbody></tbody>", html)
        self.assertIn("<thead>", html)

    def test_carries_the_class_the_stylesheet_targets(self):
        # The selectors in app.py's CSS are a contract with this file.
        html = T.comparison_table_html(ROWS)
        self.assertIn('class="comparison-wrap"', html)
        self.assertIn('class="comparison"', html)


class TestValues(unittest.TestCase):
    def test_a_long_value_is_not_truncated(self):
        # The defect that started all this: the grid clipped this address to
        # "...81300, S..." and no CSS could reach inside to widen it.
        rows = [{"Field": "Property known as", "Ground Truth (API)": "-",
                 "Extracted": LONG_ADDRESS, "Match": T.MATCH}]
        self.assertIn(LONG_ADDRESS, T.comparison_table_html(rows))

    def test_html_in_an_extracted_value_is_escaped(self):
        # The value comes from a model reading someone's document.
        rows = [{"Field": "Special/Other Condition(s)", "Ground Truth (API)": "-",
                 "Extracted": "<script>alert(1)</script>", "Match": T.MISS}]
        html = T.comparison_table_html(rows)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)

    def test_ampersands_and_quotes_survive_readably(self):
        rows = [{"Field": "Legal Cost", "Ground Truth (API)": '"Tenant" & Landlord',
                 "Extracted": "-", "Match": T.MISS}]
        html = T.comparison_table_html(rows)
        self.assertIn("&amp;", html)
        self.assertNotIn('"Tenant" & Landlord', html)

    def test_a_blank_value_reads_as_the_tables_own_dash(self):
        # Not an empty cell, which reads as a rendering fault rather than as
        # "nothing was found here".
        rows = [{"Field": "Tenant IC", "Ground Truth (API)": "",
                 "Extracted": None, "Match": T.NA}]
        html = T.comparison_table_html(rows)
        self.assertEqual(html.count(T.NA), 3)  # both values, plus the mark

    def test_a_row_missing_a_key_still_renders(self):
        html = T.comparison_table_html([{"Field": "Total"}])
        self.assertIn('<th scope="row">Total</th>', html)


class TestMatchMarks(unittest.TestCase):
    """Plain glyphs with a colour class, not the emoji the CSV carries: an
    emoji ignores CSS colour on most platforms, so a wrong row would print in
    the same ink as a right one."""

    def test_a_match_is_a_green_tick(self):
        self.assertIn('<span class="mark ok">✓</span>', T.comparison_table_html(ROWS))

    def test_a_miss_is_a_red_cross(self):
        self.assertIn('<span class="mark bad">✗</span>', T.comparison_table_html(ROWS))

    def test_an_unverifiable_row_is_a_muted_dash(self):
        self.assertIn('<span class="mark na">—</span>', T.comparison_table_html(ROWS))

    def test_the_mark_sits_in_its_own_column_class(self):
        self.assertIn('<td class="match-col">', T.comparison_table_html(ROWS))

    def test_an_unrecognised_state_is_shown_rather_than_dropped(self):
        rows = [{"Field": "Total", "Ground Truth (API)": "1", "Extracted": "1",
                 "Match": "?"}]
        self.assertIn('<span class="mark">?</span>', T.comparison_table_html(rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
