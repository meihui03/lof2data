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


class TestFieldPicker(unittest.TestCase):
    """Step 2. The list comes from the engine (`--list-fields`), so these
    assert the shape of the picker rather than particular field names - the
    one exception being that it must never offer a way to ADD a field, which
    is the whole constraint the step exists under."""

    def picker(self, at):
        return [c for c in at.checkbox if c.key and c.key.startswith("keep::")]

    def test_every_engine_field_is_offered(self):
        # One checkbox per PICK, not per field: two fields answering the same
        # comparison row are offered together (fields.picker_units).
        at = run_app()
        self.assertEqual(len(self.picker(at)), len(app_module_units()))
        offered = {name for _, _, members in app_module_units() for name in members}
        self.assertEqual(offered, set(app_module_fields()))

    def test_the_paired_fields_are_a_single_checkbox(self):
        # A form with no "Attn:" line still names the negotiator under
        # "Witnessed by", so keeping one and dropping the other loses the row.
        at = run_app()
        labels = [c.label for c in self.picker(at)]
        self.assertIn("Negotiator/Agent Name · Witnessed by", labels)
        self.assertNotIn("Witnessed by", labels)

    def test_they_all_start_ticked(self):
        at = run_app()
        self.assertTrue(all(c.value for c in self.picker(at)))

    def test_there_is_no_way_to_add_a_field(self):
        # No free-text input anywhere in the picker: the schema is the engine's,
        # and the UI can only ever narrow it.
        at = run_app()
        self.assertEqual([i for i in at.text_input if i.label != "Sale No."], [])

    def test_unticking_one_leaves_the_rest_selected(self):
        at = run_app()
        first = self.picker(at)[0]
        at.checkbox(key=first.key).set_value(False).run()
        still_on = [c for c in self.picker(at) if c.value]
        self.assertEqual(len(still_on), len(app_module_units()) - 1)

    def test_clearing_everything_disables_extract(self):
        # The engine refuses a run with nothing to extract; the button should
        # never get that far.
        at = run_app()
        at.button(key="fields_none").click().run()
        self.assertTrue(all(not c.value for c in self.picker(at)))
        self.assertTrue(at.button(key="extract").disabled)

    def test_select_all_puts_them_back(self):
        at = run_app()
        at.button(key="fields_none").click().run()
        at.button(key="fields_all").click().run()
        self.assertTrue(all(c.value for c in self.picker(at)))
        self.assertFalse(at.button(key="extract").disabled)

    def test_still_exactly_one_extract_button_after_unticking(self):
        # Same structural hazard the grounding toggle hit: a conditional
        # element (the "pick at least one field" caption) renders above the
        # button, and the button's identity must not depend on it.
        at = run_app()
        at.button(key="fields_none").click().run()
        extract = [b for b in at.button if b.label in ("Extract", "Processing...")]
        self.assertEqual(len(extract), 1)


def app_module_fields():
    """What the engine publishes - asked the same way app.py asks."""
    import app as app_module
    return app_module.list_engine_fields(app_module.engine_fingerprint())


def app_module_units():
    """What Step 2 offers as one tick each - built the same way app.py builds
    it, so a change to the grouping rule doesn't need a second edit here."""
    import app as app_module
    import fields
    return fields.picker_units(
        app_module.list_engine_fields(app_module.engine_fingerprint()),
        app_module.VLLM_ALIASES.values())


if __name__ == "__main__":
    unittest.main(verbosity=2)
