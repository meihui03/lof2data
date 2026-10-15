"""Tests for docview.py - the clickable document page.

    .venv/bin/python3 test_docview.py

The page is HTML now rather than one flattened image: the plain page as a
picture, with every highlight laid over it as its own element, so a click can
say WHICH highlight was clicked. These pin down the two things the linking
rests on - the numbering, which must be the same numbering the table's "#"
column shows, and the ids that tie a highlight to its table rows.
"""

import unittest

import docview as D


def box(label, page, x0, y0, x1, y1, display=None):
    return {"label": label, "display": display or label, "page": page,
            "box": (x0, y0, x1, y1), "color": "#66984E"}


FIELD_BOXES = [
    box("Reference No.", 0, 0.70, 0.05, 0.90, 0.07),
    box("Date", 0, 0.60, 0.20, 0.70, 0.22),
    box("Negotiator/Agent Name", 0, 0.10, 0.20, 0.40, 0.22),
    # Two fields read from one line share one region.
    box("Security Deposit", 0, 0.30, 0.50, 0.40, 0.52),
    box("Utilities Deposit", 0, 0.30, 0.50, 0.40, 0.52),
    box("Witnessed by", 1, 0.02, 0.80, 0.30, 0.82),
]


class TestOverlays(unittest.TestCase):
    def test_numbered_in_reading_order_per_page(self):
        pages = D.page_overlays(FIELD_BOXES, page_count=2)
        first = [(o["number"], o["labels"]) for o in pages[0]]
        self.assertEqual(first, [
            (1, ["Reference No."]),
            (2, ["Negotiator/Agent Name"]),
            (3, ["Date"]),
            (4, ["Security Deposit", "Utilities Deposit"]),
        ])
        self.assertEqual([o["number"] for o in pages[1]], [1])

    def test_a_page_with_nothing_found_has_no_overlays(self):
        self.assertEqual(D.page_overlays(FIELD_BOXES, page_count=3)[2], [])

    def test_the_same_field_found_twice_is_listed_once(self):
        twice = FIELD_BOXES[:1] + [box("agent.Name", 0, 0.70, 0.05, 0.90, 0.07,
                                       display="Reference No.")]
        self.assertEqual(D.page_overlays(twice, 1)[0][0]["labels"], ["Reference No."])

    def test_legend_matches_the_overlays(self):
        # badges_by_field reads this legend to number the table's rows, so the
        # page and the table cannot disagree.
        overlays = D.page_overlays(FIELD_BOXES, page_count=2)
        self.assertEqual(D.legend(overlays[0])[3], (4, "Security Deposit · Utilities Deposit"))


class TestHtml(unittest.TestCase):
    PAGES = [{"src": "data:image/jpeg;base64,AAAA", "width": 1000, "height": 1400,
              "overlays": D.page_overlays(FIELD_BOXES, 2)[0]},
             {"src": "data:image/jpeg;base64,BBBB", "width": 1000, "height": 1400,
              "overlays": D.page_overlays(FIELD_BOXES, 2)[1]}]

    def test_every_region_is_a_clickable_element_with_its_link_id(self):
        html = D.doc_view_html(self.PAGES, zoom=100)
        self.assertIn('data-id="1:4"', html)
        self.assertIn('data-id="2:1"', html)
        self.assertEqual(html.count('class="region"'), 5)

    def test_regions_are_placed_by_percentage(self):
        html = D.doc_view_html(self.PAGES, zoom=100)
        self.assertIn("left:70.0000%;top:5.0000%;width:20.0000%;height:2.0000%", html)

    def test_pin_shows_the_number(self):
        html = D.doc_view_html(self.PAGES, zoom=100)
        self.assertIn('<button type="button" class="pin" data-id="1:4"', html)
        self.assertIn('>4</button>', html)

    def test_pin_moves_right_when_the_line_starts_at_the_edge(self):
        # Hard against the left margin there is no room for a pin before it.
        html = D.doc_view_html(self.PAGES, zoom=100)
        witness_pin = html.split('class="pin" data-id="2:1"', 1)[1].split(">", 1)[0]
        self.assertIn("pin-right", witness_pin)

    def test_zoom_sets_the_page_width(self):
        self.assertIn("--zoom:150%", D.doc_view_html(self.PAGES, zoom=150))

    def test_field_names_are_escaped(self):
        pages = [{"src": "x", "width": 1, "height": 1,
                  "overlays": [{"number": 1, "box": (0.1, 0.1, 0.2, 0.2),
                                "labels": ["<b>x</b>"]}]}]
        html = D.doc_view_html(pages, zoom=100)
        self.assertNotIn("<b>x</b>", html)

    def test_knows_which_tab_shows_the_results(self):
        # A click on the page while the Input tab is open switches to the
        # results tab by its label, then scrolls the row into view.
        html = D.doc_view_html(self.PAGES, zoom=100, results_tab="Results")
        self.assertIn('const RESULTS_TAB = "Results";', html)

    def test_tab_label_is_safely_quoted(self):
        html = D.doc_view_html(self.PAGES, zoom=100, results_tab='Re"sults</script>')
        self.assertNotIn('Re"sults</script>', html)

    def test_knows_which_filter_shows_every_row(self):
        # A box whose row the chosen filter hides ("No record" while the box
        # is a matched field) switches the filter to this one, then scrolls.
        html = D.doc_view_html(self.PAGES, zoom=100, all_filter="All")
        self.assertIn('const ALL_FILTER = "All";', html)
        self.assertIn(".st-key-row_filter button", html)

    def test_filter_label_is_safely_quoted(self):
        html = D.doc_view_html(self.PAGES, zoom=100, all_filter='A"ll</script>')
        self.assertNotIn('A"ll</script>', html)

    def test_page_captions(self):
        self.assertIn("Page 2 of 2", D.doc_view_html(self.PAGES, zoom=100))


if __name__ == "__main__":
    unittest.main(verbosity=2)
