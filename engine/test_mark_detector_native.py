"""Tests for the born-digital choice-field reader: strikes AND circles.

    .venv/bin/python3 test_mark_detector_native.py

Every PDF here is generated, so no real LOF (names, NRIC numbers) is needed.
A born-digital form can be marked two ways - the system that generates it
strikes the unselected options out as vector lines, and a person marking it up
afterwards in Acrobat/Preview/a phone app circles or strikes one as an
annotation. Both have to read the same.

Optional real-document regression, skipped unless the path is set:

    LOF2DATA_SAMPLE_72941=~/training_demo/72941.pdf .venv/bin/python3 test_mark_detector_native.py
"""

import math
import os
import tempfile
import unittest

import fitz

import extractor as E
import mark_detector as md

OPTIONS = ["NOT", "PARTLY", "FULLY"]


def form():
    """A one-line choice field, with enough surrounding text to be treated as
    born-digital. Returns (doc, page, {option: Rect})."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "This Premise shall be rented NOT / PARTLY / FULLY FURNISHED.",
                     fontsize=11)
    page.insert_text((72, 130), "Notice To Quit shall be 2 months in advance in writing by "
                     "either party before expiry of the tenancy.", fontsize=11)
    return doc, page, md.locate_native(page, OPTIONS)


def around(rect, pad_x=4, pad_y=3):
    return fitz.Rect(rect.x0 - pad_x, rect.y0 - pad_y, rect.x1 + pad_x, rect.y1 + pad_y)


def strike(page, rect):
    y = (rect.y0 + rect.y1) / 2 + 1
    page.draw_line((rect.x0 - 1, y), (rect.x1 + 1, y))


def ink_ring(page, rect, points=10):
    """A freehand loop, as a pen tool records it: a polyline, not a curve."""
    r = around(rect)
    cx, cy, rx, ry = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, r.width / 2, r.height / 2
    page.add_ink_annot([[(cx + rx * math.cos(2 * math.pi * t / points),
                          cy + ry * math.sin(2 * math.pi * t / points))
                         for t in range(points + 1)]])


def resolve(doc):
    """The production entry point for a born-digital PDF, on a saved copy."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "form.pdf")
        doc.save(path)
        return E._resolve_choice_native(path, OPTIONS)


class TestFixture(unittest.TestCase):
    def test_all_three_options_are_located(self):
        _, _, boxes = form()
        self.assertEqual(sorted(boxes), sorted(OPTIONS))


class TestStrikes(unittest.TestCase):
    def test_vector_strikes_leave_the_selected_option(self):
        # What the PropNex system itself draws - 72941's furnished line.
        doc, page, boxes = form()
        strike(page, boxes["NOT"])
        strike(page, boxes["PARTLY"])
        self.assertEqual(resolve(doc), "FULLY")

    def test_strikeout_annotations_count_as_strikes(self):
        doc, page, boxes = form()
        page.add_strikeout_annot(boxes["NOT"])
        page.add_strikeout_annot(boxes["FULLY"])
        self.assertEqual(resolve(doc), "PARTLY")


class TestCircles(unittest.TestCase):
    def test_vector_oval_around_one_option(self):
        doc, page, boxes = form()
        page.draw_oval(around(boxes["PARTLY"]))
        self.assertEqual(resolve(doc), "PARTLY")

    def test_circle_annotation(self):
        doc, page, boxes = form()
        page.add_circle_annot(around(boxes["FULLY"]))
        self.assertEqual(resolve(doc), "FULLY")

    def test_freehand_ink_ring(self):
        doc, page, boxes = form()
        ink_ring(page, boxes["NOT"])
        self.assertEqual(resolve(doc), "NOT")

    def test_rectangle_annotation_is_a_deliberate_mark(self):
        # An annotation is always added by a person, so a box drawn with
        # Acrobat's rectangle tool selects, unlike a printed box.
        doc, page, boxes = form()
        page.add_rect_annot(around(boxes["PARTLY"]))
        self.assertEqual(resolve(doc), "PARTLY")

    def test_circle_reports_per_option(self):
        doc, page, boxes = form()
        page.draw_oval(around(boxes["PARTLY"]))
        self.assertEqual(md.detect_circled_native(page, boxes),
                         {"NOT": False, "PARTLY": True, "FULLY": False})


class TestAbstains(unittest.TestCase):
    def test_unmarked_line(self):
        doc, _, _ = form()
        self.assertIsNone(resolve(doc))

    def test_printed_box_around_each_option_is_not_a_selection(self):
        # Template furniture: drawn straight-edged boxes, one per option.
        doc, page, boxes = form()
        for rect in boxes.values():
            page.draw_rect(around(rect))
        self.assertIsNone(resolve(doc))

    def test_oval_around_the_whole_line_is_not_a_selection(self):
        doc, page, boxes = form()
        page.draw_oval(around(boxes["NOT"] | boxes["FULLY"], pad_x=8, pad_y=5))
        self.assertIsNone(resolve(doc))

    def test_printed_box_around_one_option_does_not_override_strikes(self):
        # A template that frames one choice. Read as a circle, it would
        # contradict the strikes and throw away a correct answer.
        doc, page, boxes = form()
        strike(page, boxes["NOT"])
        strike(page, boxes["PARTLY"])
        page.draw_rect(around(boxes["PARTLY"]))
        self.assertEqual(resolve(doc), "FULLY")

    def test_rounded_border_round_the_line_does_not_hide_a_real_circle(self):
        # Rounded corners are curves, so the border passes the ring-shape
        # test - it is rejected only because it covers more than one option.
        doc, page, boxes = form()
        page.draw_rect(around(boxes["NOT"] | boxes["FULLY"], pad_x=10, pad_y=6), radius=0.3)
        page.add_circle_annot(around(boxes["FULLY"]))
        self.assertEqual(resolve(doc), "FULLY")

    def test_tiny_ring_inside_a_word_is_not_a_circle(self):
        # A bullet or radio dot: ring-shaped, but it covers a sliver of the
        # word rather than enclosing it.
        doc, page, boxes = form()
        c = boxes["PARTLY"]
        mid = fitz.Point((c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2)
        page.draw_oval(fitz.Rect(mid.x - 2, mid.y - 2, mid.x + 2, mid.y + 2))
        self.assertIsNone(resolve(doc))

    def test_circle_and_strikes_that_disagree(self):
        # Strikes leave FULLY, the circle says PARTLY. Picking either would be
        # a guess; abstaining hands the field back to the LLM.
        doc, page, boxes = form()
        strike(page, boxes["NOT"])
        strike(page, boxes["PARTLY"])
        page.draw_oval(around(boxes["PARTLY"]))
        self.assertIsNone(resolve(doc))

    def test_circle_and_strikes_that_agree(self):
        doc, page, boxes = form()
        strike(page, boxes["NOT"])
        strike(page, boxes["PARTLY"])
        page.add_circle_annot(around(boxes["FULLY"]))
        self.assertEqual(resolve(doc), "FULLY")


@unittest.skipUnless(os.environ.get("LOF2DATA_SAMPLE_72941"),
                     "set LOF2DATA_SAMPLE_72941 to a copy of LOF 72941 to run")
class TestReal72941(unittest.TestCase):
    """Born-digital, system-drawn strikes, no annotations: Condo / FULLY. The
    circle check must not change a document that has no circles."""

    def test_choice_fields_unchanged(self):
        path = os.path.expanduser(os.environ["LOF2DATA_SAMPLE_72941"])
        self.assertEqual(E.resolve_choice_fields_geometrically(path), {
            "Description of Property/Property Type": "Condo",
            "Furnished Status": "FULLY",
        })


if __name__ == "__main__":
    unittest.main()
