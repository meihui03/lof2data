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


class TestLayoutBeforeARun(unittest.TestCase):
    def test_there_are_no_tabs(self):
        # The document lives in the left pane now, beside the table.
        self.assertEqual(len(run_app().tabs), 0)

    def test_document_pane_starts_empty(self):
        at = run_app()
        self.assertTrue(any("Your document shows here" in m.value for m in at.markdown))

    def test_header_has_no_step_indicator(self):
        # The Input / Results tabs say where you are; the header repeating it
        # was a second indicator for the same thing.
        at = run_app()
        header = next(m.value for m in at.markdown if 'class="lof-header"' in m.value)
        self.assertNotIn('class="progress"', header)

    def test_header_carries_the_logo(self):
        at = run_app()
        header = next(m.value for m in at.markdown if 'class="lof-header"' in m.value)
        self.assertIn('alt="Chiief Technology"', header)
        self.assertIn("data:image/png;base64,", header)

    def test_fields_live_in_an_expander_open_before_a_run(self):
        at = run_app()
        self.assertEqual([e.label for e in at.expander], ["Fields to extract"])

    def test_no_errors(self):
        self.assertEqual(len(run_app().exception), 0)

    def test_no_tabs_until_there_is_a_result(self):
        # Results has nothing to show yet, so there is nothing to switch to.
        with self.assertRaises(KeyError):
            run_app().button_group(key="work_view")


class TestLabelFallback(unittest.TestCase):
    """A value the model rewrote ("(1) ONE Year(s)" from "( 1" / ") ONE" /
    "Year (s)") or one too short to search for ("2") used to get no
    highlight; it is placed by its printed label's row instead."""

    def result(self):
        line = lambda i, t, x0, y0, x1, y1: {"index": i, "page": 0, "text": t,
                                              "x0": x0, "y0": y0, "x1": x1, "y1": y1}
        return {
            "fields": {"Period of Tenancy": "(1) ONE Year(s)", "Notice to Quit (months)": "2"},
            "_ocr_boxes": [line(11, "3) Period Of Tenancy", .065, .220, .30, .230),
                           line(12, ": (    1", .373, .217, .42, .229),
                           line(13, ")     ONE", .518, .217, .60, .229),
                           line(35, "10) Notice To Quit shall be     2", .065, .334, .55, .344)],
            "_raw": {"fields": {"Period of Tenancy": {"value": "(1) ONE Year(s)", "source_index": None},
                                "Notice to Quit (months)": {"value": "2", "source_index": None}},
                     "groups": {}},
        }

    def test_both_get_a_highlight(self):
        import app as app_module
        found = {b["display"]: b for b in app_module.collect_field_boxes(self.result())}
        self.assertEqual(found["Period of Tenancy"]["box"], (.065, .217, .60, .230))
        self.assertIn("Notice to Quit (months)", found)


def finished_run(at):
    """Session state as a completed extraction leaves it - injected, so no
    model, no sales API and no engine run is involved."""
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "Ref: ERHQ00008180", fontsize=11)
    at.session_state["pdf_bytes"] = doc.tobytes()
    at.session_state["pdf_name"] = "72941.pdf"
    at.session_state["result"] = {
        "fields": {"Reference No.": "ERHQ00008180", "Property Address": "B-9-16, PJ MIDTOWN",
                   "Tenant Address": "8, JALAN CONTOH 3"},
        "_ocr_boxes": [{"index": 0, "page": 0, "text": "Ref: ERHQ00008180",
                        "x0": 0.1, "y0": 0.1, "x1": 0.4, "y1": 0.13}],
        "_raw": {"fields": {"Reference No.": {"value": "ERHQ00008180", "confidence": 0.9,
                                              "source_index": 0}}, "groups": {}},
    }
    at.session_state["comparison_df"] = [
        {"Field": "Reference No.", "Ground Truth (API)": "ERHQ00008180",
         "Extracted": "ERHQ00008180", "Match": "✔️"},
        {"Field": "Property known as", "Ground Truth (API)": "B-9-18, PJ MIDTOWN",
         "Extracted": "B-9-16, PJ MIDTOWN", "Match": "✖️"},
        {"Field": "Total", "Ground Truth (API)": "—", "Extracted": "—", "Match": "—"},
    ]
    for key, value in {"elapsed": 7.7, "accuracy": 0.5, "correct": 1, "verifiable": 2,
                       "model_used": "qwen3-vl:8b-instruct", "excluded": set(),
                       "run_inputs": {"sale_no": "72941", "file": "72941.pdf",
                                      "model": "qwen3-vl:8b-instruct", "grounded": True,
                                      "kept": 29, "total": 29}}.items():
        at.session_state[key] = value
    at.run()
    # AppTest re-submits a button group's selection by iterating it as a list
    # (it was written for st.feedback); a single-select segmented_control
    # holds a plain string, so any rerun would hand format_func one letter at
    # a time. Setting the selection as a list sidesteps the harness - in a
    # browser the frontend sends the index itself and none of this arises.
    at.button_group(key="row_filter").set_value(["All"])
    at.button_group(key="work_view").set_value(["Results"])
    return at


def as_lists(at):
    """Hand the harness every single-select's value as a list again before the
    next interaction - see finished_run. Each rerun turns them back into
    strings, so a second click in one test needs this too."""
    for key in ("work_view", "row_filter"):
        try:
            group = at.button_group(key=key)
        except KeyError:
            continue
        if isinstance(group.value, str):
            group.set_value([group.value])
    return at


def markdown_of(at):
    return "\n".join(m.value for m in at.markdown)


class TestLayoutAfterARun(unittest.TestCase):
    def setUp(self):
        self.at = finished_run(AppTest.from_file(APP, default_timeout=TIMEOUT))
        self.markdown = "\n".join(m.value for m in self.at.markdown)

    def test_no_errors(self):
        self.assertEqual(len(self.at.exception), 0, [e.value for e in self.at.exception])

    def test_extract_becomes_extract_again_and_keeps_its_key(self):
        self.assertEqual(self.at.button(key="extract").label, "Extract again")

    def test_lands_on_the_results_tab(self):
        self.assertEqual(self.at.button_group(key="work_view").value, ["Results"])
        # Hidden, not removed: the uploader must keep its file for "Extract again".
        self.assertIn(".st-key-setup { display: none; }", self.markdown)
        self.assertNotIn(".st-key-card_results { display: none; }", self.markdown)
        self.assertIsNotNone(self.at.text_input(key="sale_no"))

    def test_a_stale_tab_value_falls_back_to_results(self):
        # A tab renamed while a session was open left it holding a value that
        # is no longer an option - and neither pane hid, so both showed.
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        at.session_state["work_view"] = "Step 2 · Results"
        finished_run(at)
        self.assertEqual(at.session_state["work_view"], "Results")
        self.assertIn(".st-key-setup { display: none; }", markdown_of(at))

    def test_no_step_numbers_anywhere(self):
        # "Input" and "Results" are the names; "Step 1/2" only repeated them.
        self.assertNotIn("Step 1", self.markdown)
        self.assertNotIn("Step 2", self.markdown)
        self.assertEqual(self.at.button_group(key="work_view").options[0].content, "Input")

    def test_fields_without_a_record_still_get_a_row(self):
        # Ticked and extracted, but the CRM has no tenant address to check it
        # against - it used to be dropped from the table altogether.
        self.assertIn('<th scope="row">Tenant Address</th><td>8, JALAN CONTOH 3</td>',
                      self.markdown)
        self.assertIn('<th scope="row">Monthly Rental</th>', self.markdown)

    def test_every_ticked_field_appears_in_the_table(self):
        # The rule, whatever the field list: a ticked field is either shown
        # inside a compared row that is in the table, or has a row of its own.
        # This result's compared rows are deliberately incomplete (3 of 17).
        import app as app_module
        shown_rows = {r.split('data-field="', 1)[1].split('"', 1)[0]
                      for r in self.markdown.split("<tr")[1:] if 'data-field="' in r}
        for field in app_module_fields():
            in_a_row = field in shown_rows or any(
                field in app_module.VLLM_ALIASES.get(row, []) for row in shown_rows)
            self.assertTrue(in_a_row, f"ticked field {field!r} has no row")

    def test_an_unticked_field_has_no_row(self):
        at = AppTest.from_file(APP, default_timeout=TIMEOUT)
        finished_run(at)
        at.session_state["excluded"] = {"Tenant Address"}
        as_lists(at).run()
        self.assertNotIn('<th scope="row">Tenant Address</th>', markdown_of(at))

    def test_results_say_what_the_run_used(self):
        self.assertIn('<span>Sale No.</span> 72941', self.markdown)
        self.assertIn('<span>Fields</span> 29 of 29', self.markdown)

    def test_input_tab_brings_the_inputs_back(self):
        self.at.button_group(key="work_view").set_value(["Input"]).run()
        markdown = markdown_of(self.at)
        self.assertNotIn(".st-key-setup { display: none; }", markdown)
        self.assertIn(".st-key-card_results { display: none; }", markdown)

    def test_clicking_the_open_tab_keeps_it_open(self):
        # A segmented control lets the open option be clicked off, which left
        # no tab marked at all while Results still showed.
        self.at.button_group(key="work_view").set_value([]).run()
        self.assertEqual(self.at.session_state["work_view"], "Results")

    def test_document_is_the_clickable_view(self):
        # One component frame: the page with every highlight as its own
        # clickable element, linked to the table rows by id.
        frames = self.at.get("iframe")
        self.assertEqual(len(frames), 1)
        srcdoc = frames[0].proto.srcdoc
        self.assertIn('class="region" data-id="1:1"', srcdoc)
        self.assertIn('data-links="1:1"', self.markdown)   # Reference No.'s row

    def test_document_starts_at_full_width(self):
        self.assertIn('<div class="zoom-pct">100%</div>', self.markdown)

    def test_zoom_in_and_out_change_the_page_width(self):
        self.at.button(key="zoom_in").click().run()
        markdown = markdown_of(self.at)
        self.assertIn('<div class="zoom-pct">125%</div>', markdown)
        self.assertIn("--zoom:125%", self.at.get("iframe")[0].proto.srcdoc)
        as_lists(self.at).button(key="zoom_out").click().run()
        self.assertIn('<div class="zoom-pct">100%</div>', markdown_of(self.at))

    def test_zoom_stops_at_its_limits(self):
        for _ in range(12):
            as_lists(self.at).button(key="zoom_out").click().run()
        self.assertIn('<div class="zoom-pct">50%</div>', markdown_of(self.at))
        self.assertTrue(self.at.button(key="zoom_out").disabled)
        self.assertFalse(self.at.button(key="zoom_in").disabled)

    def test_no_legend_under_the_page(self):
        # The table's # column says which number is which; a second key
        # under the page only repeated it.
        self.assertFalse(any(m.value.startswith("**1.**") for m in self.at.markdown))

    def test_table_row_carries_the_same_badge_as_the_page(self):
        # Reference No. was cited on line 0 - highlight 1 on the page.
        self.assertIn('<span class="badge">1</span></td><th scope="row">Reference No.</th>',
                      self.markdown)

    def test_the_mismatch_is_marked_down_to_the_digit(self):
        self.assertIn('B-9-1<mark class="diff">6</mark>', self.markdown)

    def test_filter_pills_carry_counts(self):
        # "All" counts every row the table shows - the compared ones and
        # the ones with no record to compare against.
        pills = self.at.button_group(key="row_filter")
        shown = self.markdown.count("<tr data-field=") + self.markdown.count('<tr class="miss" data-field=')
        self.assertEqual(pills.options[0].content, f"All  {shown}")

    def test_needs_a_look_filter_narrows_the_table(self):
        self.at.button_group(key="row_filter").set_value(["Needs a look"]).run()
        markdown = "\n".join(m.value for m in self.at.markdown)
        self.assertIn("Property known as", markdown)
        self.assertNotIn('<th scope="row">Reference No.</th>', markdown)


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
