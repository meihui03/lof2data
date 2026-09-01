"""Tests for app.py's widget structure, via Streamlit's own AppTest harness.

    .venv/bin/python3 test_app_ui.py

This exists because toggling "Give the model the document's text" briefly
rendered a SECOND Extract button. The cause is structural rather than
cosmetic: the toggle conditionally rendered a caption ABOVE the button, and
Streamlit derives an unkeyed widget's identity partly from its position in
the element tree - so adding or removing that caption made the button look
like a different widget, and the stale one lingered until reconciliation.

WHAT THESE TESTS DO AND DON'T COVER. Verified by reverting the fix and
re-running: only test_extract_button_keeps_a_stable_key failed. The
"exactly one button" tests passed WITHOUT the fix, because AppTest reads the
server-side element tree - which held one button either way - while the
duplicate was a client-side reconciliation artifact in the browser. So the
visual bug itself is NOT reproducible here.

What is pinned is the fix's mechanism: the button keeps an explicit key, so
its identity no longer depends on what is rendered above it. That is the
part a future edit could silently undo. The structural assertions below are
worth keeping as cheap sanity checks, but do not read them as coverage of
the duplicate-button symptom - confirm that in a browser.
"""

import unittest

from streamlit.testing.v1 import AppTest

# Extraction never runs here: no PDF is uploaded and the button is not
# clicked, so no model or network call is involved.
APP = "app.py"
TIMEOUT = 60


def run_app():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    return at


class TestExtractButtonIdentity(unittest.TestCase):
    def test_exactly_one_extract_button_on_first_render(self):
        at = run_app()
        extract = [b for b in at.button if b.label in ("Extract", "Processing...")]
        self.assertEqual(len(extract), 1, f"expected 1 Extract button, got {len(extract)}")

    def test_still_exactly_one_after_turning_grounding_off(self):
        at = run_app()
        at.toggle(key="grounding").set_value(False).run()
        extract = [b for b in at.button if b.label in ("Extract", "Processing...")]
        self.assertEqual(len(extract), 1,
                         f"toggling off produced {len(extract)} Extract buttons")

    def test_still_exactly_one_after_toggling_back_on(self):
        at = run_app()
        at.toggle(key="grounding").set_value(False).run()
        at.toggle(key="grounding").set_value(True).run()
        extract = [b for b in at.button if b.label in ("Extract", "Processing...")]
        self.assertEqual(len(extract), 1,
                         f"toggling back on produced {len(extract)} Extract buttons")

    def test_extract_button_keeps_a_stable_key(self):
        # Position-independent identity is what stops the duplicate coming
        # back if another conditional element is added above it.
        at = run_app()
        self.assertIsNotNone(at.button(key="extract"))


class TestGroundingNote(unittest.TestCase):
    def test_note_is_shown_only_when_grounding_is_off(self):
        at = run_app()
        self.assertFalse(any("expect lower accuracy" in c.value.lower() for c in at.caption))
        at.toggle(key="grounding").set_value(False).run()
        self.assertTrue(any("expect lower accuracy" in c.value.lower() for c in at.caption))

    def test_grounding_defaults_to_on(self):
        self.assertTrue(run_app().toggle(key="grounding").value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
