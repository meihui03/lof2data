"""Tests for the comparison table's markup.

    .venv/bin/python3 test_table.py

Two things matter here and neither is cosmetic. A value must survive intact -
the whole reason this markup is hand-built is that Streamlit's grid clipped a
long address, and a truncated value cannot be checked against the record
beside it. And extracted text goes into a page as HTML, so it has to be
escaped: it comes from a model reading a document, not from anything this app
controls.

The rows are the dicts app.py builds - the same ones the CSV is written from -
keyed by the CSV's own column names. The table only changes how they are
shown: its headers, the order of its columns, and the extra on-screen "Where"
column linking a row to its numbered highlight on the page.
"""

import csv
import io
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


def body(html):
    return html.split("<tbody>", 1)[1]


class TestStructure(unittest.TestCase):
    def test_headers_are_the_on_screen_names_in_order(self):
        html = T.comparison_table_html(ROWS)
        head = html.split("<tbody>", 1)[0]
        positions = [head.index(f"<th>{name}</th>")
                     for name in ("Field", "Extracted", "Record (API)", "Result")]
        self.assertEqual(positions, sorted(positions))

    def test_csv_names_are_not_used_as_headers(self):
        html = T.comparison_table_html(ROWS)
        self.assertNotIn("<th>Ground Truth (API)</th>", html)
        self.assertNotIn("<th>Match</th>", html)

    def test_one_row_per_input(self):
        self.assertEqual(body(T.comparison_table_html(ROWS)).count("<tr"), len(ROWS))

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
        self.assertEqual(body(T.comparison_table_html(rows)).count(T.NA), 2)

    def test_a_row_missing_a_key_still_renders(self):
        html = T.comparison_table_html([{"Field": "Total"}])
        self.assertIn('<th scope="row">Total</th>', html)


class TestResultPills(unittest.TestCase):
    """Plain glyphs with a colour class, not the emoji the CSV carries: an
    emoji ignores CSS colour on most platforms, so a wrong row would print in
    the same ink as a right one."""

    def test_a_match_is_a_green_pill(self):
        self.assertIn('<span class="st ok">✓</span>', T.comparison_table_html(ROWS))

    def test_a_miss_is_a_red_pill(self):
        self.assertIn('<span class="st bad">✗</span>', T.comparison_table_html(ROWS))

    def test_an_unverifiable_row_says_no_record(self):
        self.assertIn('<span class="st na">No record</span>', T.comparison_table_html(ROWS))

    def test_the_pill_sits_in_its_own_column_class(self):
        self.assertIn('<td class="status">', T.comparison_table_html(ROWS))

    def test_an_unrecognised_state_is_shown_rather_than_dropped(self):
        rows = [{"Field": "Total", "Ground Truth (API)": "1", "Extracted": "1",
                 "Match": "?"}]
        self.assertIn('<span class="st">?</span>', T.comparison_table_html(rows))


class TestMismatchRows(unittest.TestCase):
    ADDRESS = [{"Field": "Property known as",
                "Ground Truth (API)": "B-9-18, JALAN KEMAJUAN, PJ MIDTOWN, 46200",
                "Extracted": "B-9-16, PJ MIDTOWN, Jalan Kemajuan, Seksyen 13, "
                             "46200 Petaling Jaya, Selangor",
                "Match": T.MISS}]

    def test_a_mismatched_row_is_tinted(self):
        html = T.comparison_table_html(self.ADDRESS)
        self.assertIn('<tr class="miss" data-field="Property known as"', html)

    def test_a_matched_row_is_not(self):
        html = T.comparison_table_html(ROWS[:1])
        self.assertNotIn('class="miss"', html)
        self.assertNotIn("<mark", html)

    def test_only_the_differing_characters_are_marked(self):
        # The parts appear in a different order in the two values; a plain
        # character diff then marks "6, PJ MIDTOWN" against "8". Matching by
        # word first leaves the one real difference.
        html = T.comparison_table_html(self.ADDRESS)
        self.assertIn('B-9-1<mark class="diff">6</mark>', html)
        self.assertIn('B-9-1<mark class="diff">8</mark>', html)
        self.assertEqual(html.count("<mark"), 2)

    def test_words_one_side_simply_lacks_are_not_marked(self):
        # The record stops at the postcode; the page goes on to the city.
        # That is a shorter record, not a misread.
        html = T.comparison_table_html(self.ADDRESS)
        self.assertNotIn('<mark class="diff">Petaling', html)

    def test_unrelated_values_are_not_marked_at_all(self):
        # Two formats of one company number: nothing in common to point at.
        rows = [{"Field": "Tenant IC", "Ground Truth (API)": "202501024887",
                 "Extracted": "1626300-V / K5404881N", "Match": T.MISS}]
        self.assertNotIn("<mark", T.comparison_table_html(rows))

    def test_marked_values_are_still_escaped(self):
        rows = [{"Field": "X", "Ground Truth (API)": "<b>A1</b>",
                 "Extracted": "<b>A2</b>", "Match": T.MISS}]
        html = T.comparison_table_html(rows)
        self.assertNotIn("<b>", html)
        self.assertIn("&lt;b&gt;", html)


class TestDiffHtml(unittest.TestCase):
    def test_case_and_spacing_are_not_differences(self):
        a, b = T.diff_html("Jalan  Kemajuan B-9-16", "JALAN KEMAJUAN B-9-18")
        self.assertEqual(a.count("<mark"), 1)
        self.assertEqual(b.count("<mark"), 1)

    def test_identical_values_have_no_marks(self):
        a, b = T.diff_html("RM 2,200.00", "RM 2,200.00")
        self.assertEqual((a, b), ("RM 2,200.00", "RM 2,200.00"))


class TestWhereColumn(unittest.TestCase):
    WHERE_ROWS = [dict(ROWS[0], Where=["4"]), dict(ROWS[1], Where=["p1·2", "p2·1"]),
                  dict(ROWS[2], Where=[])]

    def test_header_is_a_hash(self):
        html = T.comparison_table_html(self.WHERE_ROWS, T.COLUMNS_WITH_WHERE)
        self.assertIn("<th>#</th>", html)

    def test_one_badge_per_label(self):
        html = T.comparison_table_html(self.WHERE_ROWS, T.COLUMNS_WITH_WHERE)
        self.assertIn('<span class="badge">4</span>', html)
        self.assertIn('<span class="badge">p1·2</span><span class="badge">p2·1</span>', html)

    def test_no_label_is_a_quiet_dash(self):
        html = T.comparison_table_html(self.WHERE_ROWS, T.COLUMNS_WITH_WHERE)
        self.assertIn('<span class="nobadge">–</span>', html)

    def test_left_out_by_default(self):
        self.assertNotIn('class="badge"', T.comparison_table_html(self.WHERE_ROWS))


class TestRowLinks(unittest.TestCase):
    """Each row carries the ids of its highlights, so a click on either side
    can find the other - and its field name, so a row with no highlight is
    still selectable."""

    def test_row_carries_its_field_and_links(self):
        rows = [dict(ROWS[0], Where=["1"], Links=["1:1", "1:4"])]
        html = T.comparison_table_html(rows, T.COLUMNS_WITH_WHERE)
        self.assertIn('data-field="Reference No."', html)
        self.assertIn('data-links="1:1 1:4"', html)

    def test_a_row_without_links_is_still_a_row_with_a_field(self):
        html = T.comparison_table_html([dict(ROWS[2], Links=[])], T.COLUMNS)
        self.assertIn('data-field="Furnish Status"', html)
        self.assertIn('data-links=""', html)

    def test_links_never_reach_the_csv(self):
        out = T.csv_bytes([dict(ROWS[0], Links=["1:1"])]).decode()
        self.assertNotIn("Links", out)
        self.assertNotIn("1:1", out)


class TestUnrecordedRows(unittest.TestCase):
    """Fields the CRM record has nothing to compare against - Tenant Address,
    Monthly Rental, the tenancy dates - were extracted and then never shown,
    because the table only listed the comparable ones."""

    ENGINE = ["Reference No.", "Monthly Rental", "Tenant Address", "Title Details",
              "Negotiator/Agent Name", "Witnessed by"]
    ALIASES = {"Reference No.": ["Reference No."],
               "Attn: Negotiator(s) / Witness": ["Negotiator/Agent Name", "Witnessed by"]}
    EXTRACTED = {"Reference No.": "ERHQ1", "Monthly Rental": "RM 2,200.00",
                 "Tenant Address": "8, JALAN CONTOH 3", "Title Details": ""}

    def rows(self, excluded=()):
        return T.unrecorded_rows(self.EXTRACTED, self.ENGINE, self.ALIASES, set(excluded))

    def test_every_field_without_a_record_gets_a_row_in_engine_order(self):
        self.assertEqual([r["Field"] for r in self.rows()],
                         ["Monthly Rental", "Tenant Address", "Title Details"])

    def test_fields_the_record_covers_are_left_to_their_own_rows(self):
        names = [r["Field"] for r in self.rows()]
        self.assertNotIn("Reference No.", names)
        self.assertNotIn("Witnessed by", names)

    def test_they_have_no_record_and_never_count_as_a_miss(self):
        row = self.rows()[1]
        self.assertEqual((row["Extracted"], row["Ground Truth (API)"], row["Match"]),
                         ("8, JALAN CONTOH 3", T.NA, T.NA))

    def test_nothing_read_shows_the_dash(self):
        self.assertEqual(self.rows()[2]["Extracted"], T.NA)

    def test_an_unticked_field_has_no_row(self):
        self.assertNotIn("Tenant Address",
                         [r["Field"] for r in self.rows(excluded={"Tenant Address"})])


class TestArrangeRows(unittest.TestCase):
    COMPARED = [{"Field": "Tenant Name"}, {"Field": "Tenant IC"}, {"Field": "Landlord Name"}]
    EXTRA = [{"Field": "Monthly Rental"}, {"Field": "Tenant Address"}]

    def names(self, rows):
        return [r["Field"] for r in rows]

    def test_an_anchored_row_follows_its_partner(self):
        rows = T.arrange_rows(self.COMPARED, self.EXTRA, {"Tenant Address": "Tenant IC"})
        self.assertEqual(self.names(rows), ["Tenant Name", "Tenant IC", "Tenant Address",
                                            "Landlord Name", "Monthly Rental"])

    def test_without_its_partner_it_stays_at_the_end(self):
        # Tenant IC unticked: there is no row to sit beside, but the address
        # was still ticked and must still show.
        rows = T.arrange_rows(self.COMPARED[:1], self.EXTRA, {"Tenant Address": "Tenant IC"})
        self.assertEqual(self.names(rows), ["Tenant Name", "Monthly Rental", "Tenant Address"])

    def test_no_anchors_keeps_the_old_order(self):
        rows = T.arrange_rows(self.COMPARED, self.EXTRA, {})
        self.assertEqual(self.names(rows), self.names(self.COMPARED + self.EXTRA))

    def test_tenant_address_sits_after_tenant_ic_by_default(self):
        self.assertEqual(T.ROW_AFTER["Tenant Address"], "Tenant IC")


class TestCsv(unittest.TestCase):
    def read(self, rows):
        return list(csv.reader(io.StringIO(T.csv_bytes(rows).decode("utf-8"))))

    def test_where_never_reaches_the_csv(self):
        # An on-screen pointer into the page image; meaningless in a file.
        header = self.read([dict(ROWS[0], Where=["4"])])[0]
        self.assertNotIn("Where", header)

    def test_csv_keeps_its_original_columns_and_values(self):
        rows = self.read(ROWS)
        self.assertEqual(rows[0], ["Field", "Ground Truth (API)", "Extracted", "Match"])
        self.assertEqual(rows[1], ["Reference No.", "206537 (31/03)", "206537 (31/03)", T.MATCH])


class TestFilters(unittest.TestCase):
    ROWS = [
        {"Field": "A", "Extracted": "x", "Match": T.MATCH, "Where": ["1"]},
        {"Field": "B", "Extracted": "x", "Match": T.MISS, "Where": ["2"]},
        {"Field": "C", "Extracted": "x", "Match": T.NA, "Where": []},   # read, but not on the page
        {"Field": "D", "Extracted": T.NA, "Match": T.NA, "Where": []},  # nothing read
    ]

    def test_counts(self):
        self.assertEqual(T.filter_counts(self.ROWS, has_where=True),
                         {T.ALL: 4, T.NEEDS_LOOK: 2, T.MATCHED: 1, T.NO_RECORD: 2})

    def test_needs_a_look_is_a_miss_or_an_unplaced_value(self):
        names = [r["Field"] for r in T.filter_rows(self.ROWS, T.NEEDS_LOOK, has_where=True)]
        self.assertEqual(names, ["B", "C"])

    def test_without_highlights_a_missing_badge_proves_nothing(self):
        # Grounding off: no row has a badge, so "no badge" would flag them all.
        names = [r["Field"] for r in T.filter_rows(self.ROWS, T.NEEDS_LOOK, has_where=False)]
        self.assertEqual(names, ["B"])

    def test_matched_and_no_record(self):
        self.assertEqual([r["Field"] for r in T.filter_rows(self.ROWS, T.MATCHED, True)], ["A"])
        self.assertEqual([r["Field"] for r in T.filter_rows(self.ROWS, T.NO_RECORD, True)],
                         ["C", "D"])

    def test_all_and_none_selected_show_everything(self):
        self.assertEqual(T.filter_rows(self.ROWS, T.ALL, True), self.ROWS)
        self.assertEqual(T.filter_rows(self.ROWS, None, True), self.ROWS)


class TestBadgeLinks(unittest.TestCase):
    PAGES = [
        {"legend": [(1, "Reference No."), (2, "Negotiator/Agent Name · Witnessed by"),
                    (3, "Property Address")]},
        {"legend": [(1, "Witnessed by"), (2, "Tenant Name(s)/Company")]},
    ]
    ALIASES = {
        "Reference No.": ["Reference No."],
        "Attn: Negotiator(s) / Witness": ["Negotiator/Agent Name", "Witnessed by"],
        "Tenant Name": ["Tenant Name(s)/Company", "Purchaser Name(s)"],
        "Total": ["Total Payable"],
    }

    def test_legend_is_inverted_by_name(self):
        index = T.badges_by_field(self.PAGES)
        self.assertEqual(index["Witnessed by"], [(1, 2), (2, 1)])
        self.assertEqual(index["Negotiator/Agent Name"], [(1, 2)])
        self.assertEqual(index["Reference No."], [(1, 1)])

    def test_single_page_labels_are_bare_numbers(self):
        index = T.badges_by_field(self.PAGES[:1])
        self.assertEqual(T.where_for_row("Reference No.", index, 1, self.ALIASES), ["1"])

    def test_multi_page_labels_name_the_page(self):
        index = T.badges_by_field(self.PAGES)
        self.assertEqual(T.where_for_row("Tenant Name", index, 2, self.ALIASES), ["p2·2"])

    def test_every_alias_is_looked_up_once(self):
        # The negotiator and the witness share badge 2 on page 1: one label.
        index = T.badges_by_field(self.PAGES)
        self.assertEqual(T.where_for_row("Attn: Negotiator(s) / Witness", index, 2, self.ALIASES),
                         ["p1·2", "p2·1"])

    def test_link_ids_name_page_and_number_whatever_the_page_count(self):
        index = T.badges_by_field(self.PAGES)
        self.assertEqual(T.links_for_row("Attn: Negotiator(s) / Witness", index, self.ALIASES),
                         ["1:2", "2:1"])
        self.assertEqual(T.links_for_row("Total", index, self.ALIASES), [])

    def test_a_field_with_no_highlight_has_no_label(self):
        index = T.badges_by_field(self.PAGES)
        self.assertEqual(T.where_for_row("Total", index, 2, self.ALIASES), [])
        self.assertEqual(T.where_for_row("Unknown", index, 2, self.ALIASES), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
