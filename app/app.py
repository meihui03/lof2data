"""lof2data: standalone UI for the hybrid (vision + OCR-grounding +
deterministic choice-field override) LOF extraction approach - shows a
Sale No.-driven ground-truth comparison, plus bounding boxes on the source
document showing exactly where each extracted value came from.
"""

import base64
import html
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import fitz
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image

import boxes
import docview
import fields as field_picker
import models
import table as comparison_table
from groundtruth import (
    amounts_match, build_canonical_ground_truth, CANONICAL_FIELD_KINDS,
    CANONICAL_FIELDS, choice_match, dates_match, fetch_sale, ids_match,
    values_match,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO_ROOT / "engine"

# The engine runs in its own virtualenv - it carries PaddleOCR/OpenCV/PyMuPDF,
# which the UI deliberately doesn't (see DESIGN.md, "subprocess delegation").
# Overridable for a non-default layout, e.g. a shared or containerised env.
ENGINE_PYTHON = Path(
    os.environ.get("LOF2DATA_ENGINE_PYTHON", ENGINE_DIR / ".venv" / "bin" / "python3")
)

load_dotenv(REPO_ROOT / ".env")

# hybrid's run() deliberately uses vllm-schema-extractor's exact field
# names, so this is the same alias map streamlit_app/app.py already uses -
# copied here rather than imported, since this is a separate standalone app.
VLLM_ALIASES = {
    "Reference No.": ["Reference No."],
    "Attn: Negotiator(s) / Witness": ["Negotiator/Agent Name", "Witnessed by"],
    "Date of LOF": ["Date"],
    "Description of Property (Selected)": ["Description of Property/Property Type"],
    "Property known as": ["Property Address"],
    "Furnish Status": ["Furnished Status"],
    "Earnest Deposit Amount": ["Earnest Deposit Amount"],
    "Security Deposit": ["Security Deposit"],
    "Others (if any)": ["Other Deposit (if any)"],
    "Utilities Deposit": ["Utilities Deposit"],
    "Advance Rental": ["Advance Rental"],
    "Total": ["Total Payable"],
    "Tenant Name": ["Tenant Name(s)/Company", "Purchaser Name(s)"],
    "Tenant IC": ["Tenant NRIC/Passport No."],
    "Landlord Name": ["Landlord Name(s)", "Vendor Name"],
    "Landlord IC": ["Landlord NRIC/Passport No."],
    "Landlord Address": ["Landlord Address"],
}

# The palette - Chiief's: off-black ink on a light canvas, green for state and
# focus, red only for a mismatch. Up here rather than beside the CSS that also
# uses it, because the page overlay below draws its badges in INK - the same
# pill the table shows - and is defined first.
INK = "#0f1012"
CANVAS = "#f2f2f4"
PANEL = "#fdfdfd"
STEEL = "#5e5e5e"
ASH = "#8f8f8f"
GREEN = "#66984E"        # borders, rings, icons (3:1 and up)
GREEN_TEXT = "#4A7736"   # the same hue, dark enough for text on white (5.2:1)
GREEN_WASH = "rgba(102,152,78,.14)"
RED = "#B3261E"
RED_WASH = "#FBF3F2"
HAIR = "rgba(0,0,0,.06)"

# The logo for the header band, inlined so the page needs no static route.
# Missing, the header just shows the product name.
try:
    LOGO_B64 = base64.b64encode(
        (Path(__file__).resolve().parent / "assets" / "chiief-logo.png").read_bytes()
    ).decode("ascii")
except OSError:
    LOGO_B64 = None

# One colour for every box: the logo's green, as a light wash (docview's CSS)
# so the text it marks stays readable. Blue/orange once encoded HOW a
# box was located - the model's own citation vs. a text search - but that is a
# detail of the lookup, not something a reader of the document needs to act
# on. Both mean the same thing to them: this is where the value was read.
BOX_COLOR = GREEN

def field_match(kind, api_value, extracted_value):
    if kind == "date":
        return dates_match(api_value, extracted_value)
    if kind == "amount":
        return amounts_match(api_value, extracted_value)
    if kind == "id":
        return ids_match(api_value, extracted_value)
    if kind == "choice":
        return choice_match(api_value, extracted_value)
    return values_match(api_value, extracted_value)


def lookup_extracted(extracted_fields, aliases):
    for name in aliases:
        value = extracted_fields.get(name, "")
        if str(value).strip():
            return value
    return ""


class RateLimited(RuntimeError):
    """The chosen cloud model ran out of quota mid-extraction. Raised
    separately from a generic failure so the UI can take the model out of
    the selector instead of just printing a stack trace."""


def run_hybrid_extraction(pdf_path, output_dir, model, grounded=True, exclude=()):
    if not ENGINE_PYTHON.exists():
        raise RuntimeError(
            f"Extraction engine interpreter not found at {ENGINE_PYTHON}. "
            "Create it with:  python3 -m venv engine/.venv && "
            "engine/.venv/bin/pip install -r engine/requirements.txt"
        )
    command = [str(ENGINE_PYTHON), "extractor.py", "--pdf", str(pdf_path),
               "--model", model, "--output-dir", str(output_dir)]
    if not grounded:
        command.append("--no-grounding")
    for name in exclude:
        command += ["--exclude", name]
    result = subprocess.run(
        command, cwd=str(ENGINE_DIR), capture_output=True, text=True,
    )
    if result.returncode != 0:
        if models.looks_rate_limited(result.stderr):
            raise RateLimited(model)
        raise RuntimeError(result.stderr[-2000:])
    out_json = Path(output_dir) / f"{Path(pdf_path).stem}.json"
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def engine_fingerprint():
    """Enough of the engine module to notice it changed - its size and mtime.

    This is the cache key for the field list below. Streamlit keys a cached
    function on its own code and its arguments, and the engine's schema is
    neither: without this, editing extractor.py to add a field left the picker
    serving the list it read at startup, with the new field simply missing and
    nothing to suggest why. Restarting the app was the only cure, and there was
    no reason for anyone to guess at it.

    None when the module can't be stat'd, which is the same key every time -
    no worse than the unkeyed cache it replaces, and list_engine_fields()
    reports the unreachable engine anyway."""
    try:
        stat = (ENGINE_DIR / "extractor.py").stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def list_engine_fields(fingerprint=None):
    """Every field the engine can extract, in its own order.

    Asked of the engine rather than listed here, so the schema stays the one
    place a field is defined - add one there and it becomes pickable with no
    edit on this side. Cached on `fingerprint` (see engine_fingerprint): it is
    a subprocess launch per call, and the answer really does only change when
    the engine does.

    Returns [] if the engine can't be reached. The picker then says so and
    steps aside rather than blocking a run it isn't needed for - an
    unavailable engine is the extraction's problem to report, not the
    picker's."""
    if not ENGINE_PYTHON.exists():
        return []
    try:
        result = subprocess.run(
            [str(ENGINE_PYTHON), "extractor.py", "--list-fields"],
            cwd=str(ENGINE_DIR), capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return []
        return list(json.loads(result.stdout))
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def field_key(name):
    """The session-state key for one field's checkbox. Explicit so the
    checkbox keeps one identity across reruns no matter what is rendered
    above it - the same reason the Extract button carries a key."""
    return f"keep::{name}"


@st.cache_data(ttl=120, show_spinner=False)
def get_model_availability(session_rate_limited):
    """Which models are selectable. Cached because Streamlit reruns this
    whole script on every widget interaction, and the cloud probe is a real
    network round-trip - uncached it would fire on every keystroke in the
    Sale No. box. session_rate_limited is part of the cache key (hence a
    frozenset, not a set) so marking a model dead invalidates it at once
    rather than waiting out the TTL."""
    return models.available_models(session_rate_limited=session_rate_limited)


def collect_field_boxes(result):
    """Walks result["_raw"]["fields"] and result["_raw"]["groups"] to find
    every field/sub-field's box: prefer the LLM's own source_index citation
    when it named a valid OCR line, otherwise fall back to matching the
    extracted value's text directly against the OCR result (find_box_by_text)
    - this needs no knowledge of which of any number of possible form
    layouts the page uses. Returns [{"label": str, "display": str,
    "page": int, "box": (x0,y0,x1,y1) fractions, "color": str}, ...].

    "label" is the internal name a box was found under; "display" is what a
    reader should see. They differ for a group sub-field, which the raw reply
    nests ("agent"/"Name") and the table names flat ("Negotiator/Agent Name").
    Both passes below box the same line, so without the second name the
    legend printed one line twice, under two names for one field."""
    ocr_boxes = result.get("_ocr_boxes", [])
    ocr_by_index = {b["index"]: b for b in ocr_boxes}
    raw = result.get("_raw", {})
    # {sub-field: option list} for the choice fields, published by the engine
    # from its own schema (extractor.choice_candidates).
    candidates_by_field = raw.get("_choice_candidates", {})
    # Published by the engine alongside the choice candidates, for the same
    # reason: the UI needs the schema's own mapping and must not restate it.
    flat_names = boxes.flat_name_by_group_path(raw.get("_field_sources"))
    found = []

    def add_box(label, b, color, display=None):
        found.append({
            "label": label, "display": display or label, "page": b["page"],
            "box": (b["x0"], b["y0"], b["x1"], b["y1"]), "color": color,
        })

    def add_from_leaf(label, leaf, candidates=None, display=None):
        # The model does not always use the {value, confidence, source_index}
        # envelope the prompt asks for - a bare string is common, and used to
        # be dropped here while the flattener accepted it, so the value showed
        # in the table with no box. See boxes.normalize_leaf.
        leaf = boxes.normalize_leaf(leaf)
        value = leaf.get("value")
        # A model may cite more than one OCR line for a value that spans
        # them (and may cite it in shapes the prompt didn't ask for) - see
        # boxes.cited_boxes. Draw one box per cited line.
        #
        # The citation is checked against the line it names before being
        # drawn: an index that merely resolves is not evidence it is right,
        # and a confidently-boxed wrong line is worse than no box. A citation
        # that fails falls through to the text match below, which on a real
        # run located every signature-block field the model had misattributed.
        cited = boxes.cited_boxes(ocr_by_index, leaf.get("source_index"), value=value)
        if cited:
            for b in cited:
                add_box(label, b, BOX_COLOR, display)
            return
        if not value:
            return
        # A choice field's value is a short, common word ("NOT", "Land") that
        # occurs inside ordinary form prose, so locate its printed option LINE
        # instead - the line carrying most of the option list. Falls through
        # when this form states the field as free text with no printed list.
        b = boxes.find_choice_line(ocr_boxes, candidates) if candidates else None
        if b is None:
            b = boxes.find_box_by_text(ocr_boxes, value)
        if b is None:
            # Last resort: the row the field's printed label starts. Places a
            # value the model rewrote ("(1) ONE Year(s)" from three separate
            # lines) or one too short to search for ("2") - see
            # boxes.find_row_by_label.
            b = boxes.find_row_by_label(ocr_boxes, display or label)
        if b is not None:
            add_box(label, b, BOX_COLOR, display)

    for name, leaf in raw.get("fields", {}).items():
        add_from_leaf(name, leaf)

    for group_name, value in raw.get("groups", {}).items():
        occurrences = value if isinstance(value, list) else [value]
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            for sub_field, leaf in occurrence.items():
                display = boxes.group_display_name(group_name, sub_field, flat_names)
                # None = extracted to help the model structure its answer, but
                # offered by no checkbox and shown in no row, so it has no
                # business being highlighted and numbered on the page.
                if display is None:
                    continue
                # The label stays the raw path: it is what the sweep below
                # checks against, and changing it there would change which
                # values that safety net still looks for.
                add_from_leaf(f"{group_name}.{sub_field}", leaf,
                              candidates_by_field.get(sub_field), display=display)

    # Finally, sweep the FLATTENED fields - the ones the comparison table
    # actually shows. Everything above walks the model's raw reply, so a value
    # that reached the table by another route gets no box: a party the model
    # returned under a different role than the schema expected, or a group it
    # returned empty while the flattener still found the name. That is how a
    # whole signature block ends up unboxed while its names sit in the table.
    #
    # A repeatable role arrives here already joined with " / ", and no single
    # line holds both parties, so each is located separately.
    boxed = {label for label in (f["label"] for f in found)}
    for name, value in (result.get("fields") or {}).items():
        if not value or name in boxed:
            continue
        placed = False
        for part in boxes.split_occurrences(value):
            b = boxes.find_box_by_text(ocr_boxes, part)
            if b is not None:
                add_box(name, b, BOX_COLOR)
                placed = True
        if not placed:
            # As above: by the field's printed label when the value itself
            # can't be found as written.
            b = boxes.find_row_by_label(ocr_boxes, name)
            if b is not None:
                add_box(name, b, BOX_COLOR)

    return found


@st.cache_data(max_entries=4, show_spinner=False)
def render_plain_pages(pdf_bytes, dpi=110):
    """Every page as an image, unannotated - the preview between uploading a
    PDF and extracting it. Cached because every widget interaction reruns the
    script, and re-rasterising the PDF each time made the page feel stuck."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
    return images


# Wide enough to stay sharp at the 300% zoom in a ~600px pane, small enough
# to send to the browser on every rerun.
PAGE_IMAGE_PX = 1800


@st.cache_data(max_entries=4, show_spinner=False)
def page_images(pdf_bytes, width_px=PAGE_IMAGE_PX):
    """Every page as a JPEG data URI about `width_px` wide, for the clickable
    document view (docview). Plain - the highlights are laid over it as HTML,
    so a click can tell which one it hit. Cached: every widget interaction
    reruns the script."""
    out = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page in pdf:
            scale = width_px / page.rect.width
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=82)
            out.append({"src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
                        "width": img.width, "height": img.height})
    return out


TAB_INPUT, TAB_RESULTS = "Input", "Results"

# Document zoom, as a share of the pane's width. 100% fits the page to the
# pane; past it the page scrolls inside its panel and nothing else moves.
ZOOM_MIN, ZOOM_MAX, ZOOM_STEP = 50, 300, 25


def keep_a_tab():
    """A segmented control lets the open option be clicked off, which left no
    tab marked while Results still showed. Put the last one back."""
    if st.session_state.get("work_view") is None:
        st.session_state["work_view"] = st.session_state.get("_last_view", TAB_RESULTS)
    st.session_state["_last_view"] = st.session_state["work_view"]


# The document pane before anything is uploaded. The picture is a preview of
# what the pane will do - a page with lines highlighted in the logo green and
# numbered the way a real result is - rather than a generic file icon.
EMPTY_DOC_HTML = (
    '<div class="doc-empty">'
    '<svg class="doc-empty-art" viewBox="0 0 168 200" aria-hidden="true">'
    '<rect x="14" y="8" width="140" height="184" rx="10" fill="#fdfdfd" '
    'stroke="rgba(0,0,0,.10)"/>'
    '<rect x="34" y="30" width="62" height="7" rx="3.5" fill="#d9d9dd"/>'
    '<rect x="34" y="50" width="100" height="5" rx="2.5" fill="#e6e6e9"/>'
    '<rect x="30" y="64" width="108" height="13" rx="3" fill="rgba(102,152,78,.22)"/>'
    '<rect x="34" y="68" width="78" height="5" rx="2.5" fill="#cfd8c9"/>'
    '<rect x="34" y="88" width="92" height="5" rx="2.5" fill="#e6e6e9"/>'
    '<rect x="34" y="102" width="100" height="5" rx="2.5" fill="#e6e6e9"/>'
    '<rect x="30" y="116" width="84" height="13" rx="3" fill="rgba(102,152,78,.22)"/>'
    '<rect x="34" y="120" width="56" height="5" rx="2.5" fill="#cfd8c9"/>'
    '<rect x="34" y="140" width="96" height="5" rx="2.5" fill="#e6e6e9"/>'
    '<rect x="34" y="154" width="70" height="5" rx="2.5" fill="#e6e6e9"/>'
    '<circle cx="18" cy="70.5" r="9" fill="#0f1012"/>'
    '<text x="18" y="74" text-anchor="middle" font-size="10" fill="#fdfdfd" '
    'font-family="Inter, sans-serif">1</text>'
    '<circle cx="18" cy="122.5" r="9" fill="#66984E"/>'
    '<text x="18" y="126" text-anchor="middle" font-size="10" fill="#fdfdfd" '
    'font-family="Inter, sans-serif">2</text>'
    '</svg>'
    '<b>Your document shows here</b>'
    '<p>Upload a Letter of Offer on the right. After extraction, every value is '
    'highlighted on the page where it was read — click one to find its field.</p>'
    '<ol class="doc-empty-steps">'
    '<li><span>1</span>Upload the PDF</li>'
    '<li><span>2</span>Extract</li>'
    '<li><span>3</span>Click a highlight</li>'
    '</ol>'
    '</div>'
)


def card_title(title):
    """A card's heading, large and light. A div rather than a markdown
    heading, which would grow an anchor link."""
    return f'<div class="card-title">{html.escape(title)}</div>'


def chips_html(pairs):
    return '<div class="summary">' + "".join(
        f'<span class="chip"><span>{html.escape(k)}</span> {html.escape(str(v))}</span>'
        for k, v in pairs) + "</div>"


def metrics_html(accuracy, correct, verifiable, elapsed):
    return (
        '<div class="metrics">'
        f'<div class="metric"><span>Accuracy</span><b>{accuracy:.0%}</b>'
        f'<div class="bar"><i style="width:{accuracy * 100:.1f}%"></i></div></div>'
        f'<div class="metric"><span>Fields matched</span><b>{correct} / {verifiable}</b>'
        f'<div class="bar"><i style="width:{(correct / verifiable * 100) if verifiable else 0:.1f}%">'
        '</i></div></div>'
        f'<div class="metric"><span>Time taken</span><b>{elapsed:.1f} s</b></div>'
        "</div>"
    )


st.set_page_config(page_title="lof2data", page_icon=":material/diamond:", layout="wide")

if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "rate_limited" not in st.session_state:
    st.session_state["rate_limited"] = frozenset()
# A finished run lands on Step 2. Set here, before the tab widget is drawn:
# Streamlit refuses a write to a widget's value once the widget exists on
# this run, which is where the run itself finishes.
# Seeded here rather than passed as the widget's default: a widget given both
# a default and a value written through session state draws a Streamlit
# warning, and this one is written after every run.
#
# Also reset when the stored value is not one of the tabs at all - missing, or
# left over from before a tab was renamed. Neither pane then matched it, so
# neither hid and both showed at once.
if (st.session_state.pop("goto_results", False)
        or st.session_state.get("work_view") not in (TAB_INPUT, TAB_RESULTS)):
    st.session_state["work_view"] = TAB_RESULTS

has_result = "comparison_df" in st.session_state
processing = st.session_state["is_processing"]
# Which tab is open. No tabs before a first result - Results has nothing to
# show yet.
view = st.session_state["work_view"] if has_result else TAB_INPUT
# The tab that is not open is HIDDEN by CSS rather than not rendered:
# Streamlit forgets a widget's value - the uploaded PDF included - on any run
# where the widget isn't drawn, and "Extract again" needs every one of them.
input_hidden = view == TAB_RESULTS
results_hidden = has_result and view == TAB_INPUT

st.markdown(f"""
<style>
/* Type: Inter (loaded by the theme, .streamlit/config.toml) at a light 350 -
   the theme only takes whole hundreds - with tight tracking. */
.stApp {{ background-color: {CANVAS}; letter-spacing: -0.02em; }}
.stMarkdown p, .stCaption, [data-testid="stCaptionContainer"] {{ font-weight: 350; }}
.stApp strong, .stApp b {{ font-weight: 500; }}
[data-testid="stCaptionContainer"] {{ color: {STEEL}; }}
/* Room for the header band, which runs edge to edge under Streamlit's toolbar. */
/* The header band starts at the very top, under Streamlit's own toolbar -
   which is see-through and greyed below - so the top of the page reads as one
   dark header rather than a pale strip with "Deploy" above it. */
[data-testid="stMainBlockContainer"] {{
    padding: 0 34px 69px 34px; max-width: none;
}}
[data-testid="stHeader"] {{ background: transparent; }}
/* A markdown block that only carries a <style> (this sheet, the zoom rule) is
   invisible, but Streamlit still puts its 16px gap after it - the strip that
   showed above the header band. A style applies from a hidden element too. */
[data-testid="stElementContainer"]:has(style) {{ display: none; }}
/* Grey, not white: once the page scrolls the toolbar sits over the light
   canvas, and this reads on both. */
[data-testid="stHeader"] button, [data-testid="stHeader"] span,
[data-testid="stHeader"] [data-testid="stStatusWidget"] {{ color: #a3a3a8; }}

/* Header: a dark band, the logo's home. No step indicator - the Input /
   Results tabs in the right pane say where you are. */
.lof-header {{
    background: {INK}; color: {PANEL}; margin: 0 -34px 22px -34px; padding: 22px 34px;
    display: flex; align-items: center; justify-content: space-between; gap: 24px;
    flex-wrap: wrap;
}}
.lof-header .brandline {{ display: flex; align-items: center; gap: 22px; }}
.lof-header .brandline img {{ height: 34px; width: auto; display: block; }}
.lof-header .divider {{ width: .5px; align-self: stretch; background: rgba(255,255,255,.18); }}
.lof-header .brand {{ font-size: 18px; color: {PANEL}; font-weight: 400; line-height: 1.2; }}
.lof-header .tagline {{ color: {ASH}; font-size: 13px; margin-top: 2px; font-weight: 350; }}

/* Cards: the setup wrapper (Steps 1 + 2), the summary, the results. */
.st-key-setup, .st-key-card_results {{
    background: {PANEL}; border: .5px solid {HAIR}; border-radius: 32px; padding: 30px;
}}
.card-title {{ margin-bottom: 10px;
    font-size: 27px; font-weight: 350; line-height: 1.2; letter-spacing: -0.54px; color: {INK};
}}
.sub-step {{ border-top: .5px solid {HAIR}; padding-top: 16px; margin-top: 6px; }}
.sub-step small {{ font-size: 12px; color: {ASH}; }}
.summary {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }}
.chip {{
    display: inline-flex; align-items: center; gap: 6px; background: {CANVAS};
    border: .5px solid {HAIR}; border-radius: 9999px; padding: 5px 12px; font-size: 13px;
}}
.chip span {{ color: {ASH}; }}

/* Widgets. Buttons are pills (theme buttonRadius); the secondary ones are a
   soft grey wash. */
button[data-testid="stBaseButton-secondary"] {{
    background: rgba(12,13,15,.05); border: 0; color: #020201;
}}
button[data-testid="stBaseButton-secondary"]:hover {{ background: rgba(12,13,15,.09); color: #020201; }}
[data-testid="stDownloadButton"] [data-testid="stIconMaterial"] {{ color: {GREEN}; }}
/* The header's action sits at the right edge, at its own width. */
.st-key-card_results [data-testid="stDownloadButton"] {{
    display: flex; justify-content: flex-end;
}}
[data-testid="stDownloadButton"] button {{ white-space: nowrap; }}
/* Scoped to the card's own header row: a bare "column that contains the
   button" also matched the whole right-hand pane, and right-aligned the
   filter pills with it. */
.st-key-card_results div[data-testid="stColumn"]:last-child:has([data-testid="stDownloadButton"]) [data-testid="stVerticalBlock"] {{
    align-items: flex-end;
}}
.st-key-extract button {{ min-height: 3.3rem; font-size: 16px; }}
div[data-testid="stButton"] button:disabled,
div[data-testid="stButton"] button:disabled:hover {{
    background-color: rgba(15,16,18,.28) !important; color: {PANEL} !important;
    border: 0 !important; opacity: 1 !important;
}}
[data-testid="stFileUploaderDropzone"] {{
    border: .5px dashed rgba(0,0,0,.2); border-radius: 16px; background: {CANVAS};
}}
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {{
    border-color: {GREEN} !important; box-shadow: 0 0 0 3px {GREEN_WASH};
}}
.st-key-setup [data-testid="stExpander"] details {{
    border: 0; border-radius: 0; background: transparent;
}}
.st-key-setup [data-testid="stExpander"] summary {{
    background: transparent; padding-left: 0; padding-right: 0; font-size: 16px;
}}
/* Input / Results: a two-way switch across the top of the right pane, the
   open tab in ink. */
.st-key-work_view [data-testid="stButtonGroup"] {{
    background: {PANEL}; border: .5px solid {HAIR}; border-radius: 9999px; padding: 4px;
}}
.st-key-work_view button {{ border: 0; background: transparent; font-size: 15px; color: {STEEL}; }}
.st-key-work_view button[data-testid="stBaseButton-segmented_controlActive"] {{
    background: {INK}; color: {PANEL};
}}
/* Row filters: outlined pills, the chosen one in green. */
button[data-testid="stBaseButton-pills"], button[data-testid="stBaseButton-pillsActive"] {{
    background: transparent; border: .5px solid rgba(0,0,0,.12); font-size: 13px; color: {INK};
}}
button[data-testid="stBaseButton-pillsActive"] {{ border-color: {GREEN}; color: {GREEN_TEXT}; }}

/* Left pane: the document, pinned beside the work on a wide screen. */
.st-key-doc_pane {{
    background: {PANEL}; border: .5px solid {HAIR}; border-radius: 32px; padding: 16px 22px 18px;
    /* A keyed container lines its children up at the start, and a page image
       then shrinks to a thumbnail; stretch lets width="stretch" fill the pane. */
    align-items: stretch;
}}
/* No panel of its own: the page sits straight on the document card. It still
   scrolls here when zoomed past 100% - not the whole pane. */
.st-key-doc_pages {{ align-items: stretch; overflow: auto; }}
.st-key-doc_pages [data-testid="stImage"], .st-key-doc_pages [data-testid="stImageContainer"],
.st-key-doc_pages [data-testid="stFullScreenFrame"] {{ overflow: visible; }}
.st-key-doc_pages img {{ border-radius: 6px; border: .5px solid {HAIR}; }}
.st-key-doc_bar {{ border-bottom: .5px solid {HAIR}; padding-bottom: 8px; }}
.doc-bar {{ display: flex; align-items: baseline; gap: 12px; overflow-wrap: anywhere; }}
/* Zoom: icon buttons and the level between them, at the bar's right edge. */
.st-key-doc_bar button[data-testid="stBaseButton-tertiary"] {{
    color: {STEEL}; min-height: 0; padding: 4px;
}}
.st-key-doc_bar button[data-testid="stBaseButton-tertiary"]:hover {{ color: {INK}; }}
.st-key-doc_bar button[data-testid="stBaseButton-tertiary"]:disabled {{
    background: transparent !important; color: #c4c4c8 !important;
}}
/* Streamlit pulls markdown up by a -1rem bottom margin, which left "100%"
   in a 6px box hanging below the icons it sits between. */
.st-key-doc_bar [data-testid="stMarkdownContainer"] {{ margin-bottom: 0; }}
.zoom-pct {{ text-align: center; color: {STEEL}; font-size: 14px; white-space: nowrap; min-width: 3.2em; }}
.doc-bar .file {{ color: {INK}; font-weight: 400; }}
.doc-bar .meta {{ color: {ASH}; font-size: 13px; }}
.doc-foot {{ font-size: 13px; color: {STEEL}; display: flex; gap: 8px; align-items: center; }}
.swatch {{ width: 14px; height: 9px; background: rgba(102,152,78,.35); border-radius: 2px; display: inline-block; flex: none; }}
/* Empty: straight on the card - no second frame inside it - with a preview
   of the highlighted, numbered page and the three steps to get there. */
.doc-empty {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; gap: 8px; padding: 48px 28px; min-height: 30rem;
}}
.doc-empty-art {{ width: 132px; height: auto; margin-bottom: 14px;
    filter: drop-shadow(0 6px 18px rgba(15,16,18,.08)); }}
.doc-empty b {{ font-size: 20px; font-weight: 400; color: {INK}; letter-spacing: -0.3px; }}
.doc-empty p {{ margin: 0; color: {STEEL}; max-width: 26rem; font-size: 14px; line-height: 1.5; }}
.doc-empty-steps {{
    display: flex; flex-wrap: wrap; justify-content: center; gap: 6px;
    list-style: none; margin: 18px 0 0 0; padding: 0;
}}
.doc-empty-steps li {{
    display: inline-flex; align-items: center; gap: 8px; margin: 0;
    background: {CANVAS}; border-radius: 9999px; padding: 5px 12px 5px 5px;
    font-size: 13px; color: {STEEL};
}}
.doc-empty-steps span {{
    width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center;
    background: {PANEL}; border: .5px solid rgba(0,0,0,.12); font-size: 11px; color: {INK};
}}
@media (min-width: 992px) {{
    div[data-testid="stHorizontalBlock"]:has(.st-key-doc_pane) {{ align-items: flex-start; }}
    div[data-testid="stColumn"]:has(.st-key-doc_pane) {{
        position: sticky; top: 3.75rem; align-self: flex-start;
    }}
    .st-key-doc_pane {{ max-height: calc(100vh - 5rem); overflow-y: auto; }}
}}

/* Results: three soft tiles, accuracy and fields matched with a thin bar. */
.metrics {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin: 6px 0 4px 0; }}
.metric {{ background: {CANVAS}; border-radius: 20px; padding: 16px 18px; }}
.metric span {{ font-size: 12px; color: {ASH}; }}
.metric b {{
    display: block; font-size: 27px; font-weight: 350; letter-spacing: -0.54px;
    line-height: 1.2; margin-top: 2px; color: {INK};
}}
.metric .bar {{ height: 3px; border-radius: 2px; background: rgba(0,0,0,.08); margin-top: 10px; overflow: hidden; }}
.metric .bar i {{ display: block; height: 100%; background: {INK}; }}

/* The comparison table. Markup is ours (table.py), so these selectors are a
   contract with that file rather than with Streamlit's DOM. */
.comparison-wrap {{ overflow-x: auto; margin: 0 0 6px 0; }}
/* margin 0: Streamlit gives a markdown table a bottom margin, which showed as an
   empty strip under the last row. */
table.comparison {{
    width: 100%; border-collapse: collapse; font-size: 14px; margin: 0; border: 0;
}}
table.comparison thead th {{
    background: transparent; color: {ASH}; font-weight: 350; font-size: 12px;
    text-align: left; white-space: nowrap; padding: 8px 10px;
    border: 0; border-bottom: .5px solid rgba(0,0,0,.12);
}}
/* A field name breaks between words only - "Negotiato / r(s)" is unreadable -
   and keeps enough width that it rarely has to. */
table.comparison tbody th {{
    text-align: left; font-weight: 400; color: {INK}; min-width: 6.5rem; width: 27%;
    padding: 11px 10px; vertical-align: top; overflow-wrap: break-word;
    border: 0; border-bottom: .5px solid {HAIR};
}}
table.comparison tbody td {{
    color: {INK}; padding: 11px 10px; vertical-align: top; overflow-wrap: anywhere;
    border: 0; border-bottom: .5px solid {HAIR}; font-weight: 350;
}}
table.comparison tbody tr {{ background: transparent; }}
table.comparison tbody tr:hover {{ background: {CANVAS}; }}
table.comparison tbody tr.miss {{ background: {RED_WASH}; }}
/* Rows are clickable - docview links a row to its highlight. The selected
   one gets the grey ground and a green bar down its left edge. */
table.comparison tbody tr[data-field] {{ cursor: pointer; }}
table.comparison tbody tr.sel {{ background: {CANVAS}; }}
table.comparison tbody tr.sel > :first-child {{ box-shadow: inset 2px 0 0 {GREEN}; }}
table.comparison td.truth {{ color: {STEEL}; }}
/* Several badges stack rather than widen the column and squeeze the field. */
table.comparison td.where {{ width: 38px; padding-right: 0; }}
table.comparison td.status {{ width: 96px; white-space: nowrap; }}
.comparison .badge {{
    display: inline-grid; place-items: center; min-width: 20px; height: 20px;
    padding: 0 5px; margin: 0 3px 3px 0; border-radius: 9999px;
    background: {INK}; color: {PANEL}; font-size: 11px; font-weight: 400; letter-spacing: 0;
}}
.comparison .nobadge {{ display: inline-block; width: 20px; text-align: center; color: #c4c4c8; }}
.comparison mark.diff {{ background: #F4D7D3; border-radius: 2px; padding: 0 1px; color: #7A1A13; }}
/* Plain glyphs, not emoji: an emoji ignores colour on most platforms, so a
   wrong row would print in the same ink as a right one. */
.comparison .st {{
    display: inline-flex; align-items: center; gap: 4px; font-size: 12px; font-weight: 400;
    border-radius: 9999px; padding: 2px 10px; border: .5px solid;
}}
.comparison .st.ok {{ color: {GREEN_TEXT}; border-color: {GREEN}; }}
.comparison .st.bad {{ color: {RED}; border-color: {RED}; }}
.comparison .st.na {{ color: {ASH}; border-color: transparent; padding-left: 0; }}

/* The tab that isn't open - see input_hidden / results_hidden. */
{".st-key-setup { display: none; }" if input_hidden else ""}
{".st-key-extract { display: none; }" if input_hidden else ""}
{".st-key-card_results { display: none; }" if results_hidden else ""}
/* Below ~960px the two panes stack (Streamlit's own breakpoint is 640px, by
   which point half-width panes are already too narrow), and the document is
   no longer pinned. */
@media (max-width: 960px) {{
    div[data-testid="stHorizontalBlock"]:has(.st-key-doc_pane) {{ flex-direction: column; }}
    div[data-testid="stHorizontalBlock"]:has(.st-key-doc_pane) > div[data-testid="stColumn"] {{
        width: 100% !important; flex: 1 1 100% !important; min-width: 0;
    }}
    [data-testid="stMainBlockContainer"] {{ padding-left: 16px; padding-right: 16px; }}
    .lof-header {{ margin-left: -16px; margin-right: -16px; padding: 16px; }}
}}
@media (max-width: 560px) {{
    .metrics {{ grid-template-columns: 1fr; }}
    .lof-header .tagline, .lof-header .divider {{ display: none; }}
    .st-key-setup, .st-key-card_results {{ padding: 22px; border-radius: 24px; }}
}}
</style>
""", unsafe_allow_html=True)

logo = (f'<img src="data:image/png;base64,{LOGO_B64}" alt="Chiief Technology">'
        '<span class="divider" aria-hidden="true"></span>') if LOGO_B64 else ""
st.markdown(
    f'<div class="lof-header"><div class="brandline">{logo}<div>'
    '<div class="brand">lof2data</div>'
    '<div class="tagline">Extract Letter of Offer fields, verified against the real '
    'record.</div></div></div></div>',
    unsafe_allow_html=True,
)

# The document on the left, the work on the right. The right column is filled
# first and the left one last: what the left pane shows depends on whether a
# result exists for the PDF that is uploaded NOW, which is only known once the
# right column has read the uploader and drawn the results.
doc_col, work_col = st.columns([1.05, 1], gap="large")


def render_doc_pane(uploaded_file, pages):
    """The left pane, in one of three states: the annotated pages of the last
    run, a plain preview of a PDF not yet extracted, or an empty placeholder."""
    with doc_col, st.container(key="doc_pane"):
        uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        result_pdf = st.session_state.get("pdf_bytes")
        shows_result = (has_result and pages is not None
                        and (uploaded_bytes is None or uploaded_bytes == result_pdf))

        if shows_result:
            name = st.session_state.get("pdf_name", "Document")
            count = len(pages)
        elif uploaded_bytes is not None:
            name, count = uploaded_file.name, None
        else:
            st.markdown(EMPTY_DOC_HTML, unsafe_allow_html=True)
            return

        plain = None if shows_result else render_plain_pages(uploaded_bytes)
        count = count if count is not None else len(plain)
        zoom = st.session_state.get("doc_zoom", 100)
        with st.container(key="doc_bar"):
            name_col, zoom_col = st.columns([3, 1.3], vertical_alignment="center")
            name_col.markdown(
                f'<div class="doc-bar"><span class="file">{html.escape(name)}</span>'
                f'<span class="meta">Page 1 of {count}</span></div>',
                unsafe_allow_html=True,
            )
            # One tight row, right-aligned: out, level, in.
            with zoom_col, st.container(horizontal=True, horizontal_alignment="right",
                                        vertical_alignment="center", gap="small"):
                if st.button("", icon=":material/zoom_out:", key="zoom_out", type="tertiary",
                             help="Zoom out", disabled=zoom <= ZOOM_MIN):
                    st.session_state["doc_zoom"] = max(ZOOM_MIN, zoom - ZOOM_STEP)
                    st.rerun()
                st.markdown(f'<div class="zoom-pct">{zoom}%</div>', unsafe_allow_html=True,
                            width="content")
                if st.button("", icon=":material/zoom_in:", key="zoom_in", type="tertiary",
                             help="Zoom in", disabled=zoom >= ZOOM_MAX):
                    st.session_state["doc_zoom"] = min(ZOOM_MAX, zoom + ZOOM_STEP)
                    st.rerun()
        # Widened in CSS rather than re-rendered: the page images are cached,
        # and a zoom only needs them drawn bigger, not drawn again.
        st.markdown(
            "<style>.st-key-doc_pages [data-testid='stImage'] img "
            f"{{ width: {zoom}% !important; max-width: none !important; }}</style>",
            unsafe_allow_html=True,
        )

        if shows_result:
            if not st.session_state["result"].get("_ocr_boxes"):
                # Every highlight is anchored to a text line, so an ungrounded
                # run has nothing to draw. Say so rather than show a bare page
                # that looks like the highlights silently failed.
                st.info("No highlights for this run — text grounding was off. Run "
                        "again with it on to see where each value was read.")
            with st.container(key="doc_pages"):
                # Clickable: a highlight selects its row in the table, a row
                # its highlight here - see docview. The frame sizes itself to
                # the page; the height given is only its first guess.
                components.html(docview.doc_view_html(pages, zoom, results_tab=TAB_RESULTS,
                                                      all_filter=comparison_table.ALL),
                                height=900)
            st.markdown('<div class="doc-foot"><span class="swatch"></span> Highlighted '
                        'where each value was read. Click one, or a row on the right, '
                        'to link them.</div>', unsafe_allow_html=True)
        else:
            if has_result:
                st.caption("A different PDF from the last run — extract again to "
                           "see its highlights.")
            with st.container(key="doc_pages"):
                for i, image in enumerate(plain, start=1):
                    st.image(image, caption=f"Page {i} of {count}", width="stretch")


with work_col:
    # Always created, filled only after a run: a container that comes and goes
    # would shift every widget below it in the element tree - the cause of the
    # duplicate Extract button once already.
    tabs_slot = st.container()
    if has_result:
        with tabs_slot:
            st.segmented_control(
                "View", options=(TAB_INPUT, TAB_RESULTS),
                key="work_view", label_visibility="collapsed", width="stretch",
                disabled=processing, on_change=keep_a_tab,
            )

    # Input and fields are one card: the setup wrapper itself is styled as it.
    with st.container(key="setup"):
        with st.container():
            st.markdown(card_title("Input"), unsafe_allow_html=True)

            try:
                usable, blocked = get_model_availability(st.session_state["rate_limited"])
                ollama_error = None
            except Exception as e:
                usable, blocked, ollama_error = [], [], e

            # Keyed, like every widget here: the summary card above appears
            # only after a run, and an unkeyed widget's identity depends on its
            # position - an uploader taken for a new one would drop its file.
            sale_col, model_col, recheck_col = st.columns([1, 1.25, 0.5],
                                                          vertical_alignment="bottom")
            with sale_col:
                sale_no = st.text_input("Sale No.", placeholder="e.g. 72834", key="sale_no",
                                        help="Used to fetch ground truth from the sales API "
                                             "for comparison.")
            with model_col:
                selected_model = st.selectbox(
                    "Model",
                    options=usable,
                    index=usable.index(models.pick_default(usable)) if usable else None,
                    format_func=lambda name: f"{name}  ·  {models.label_for(name)}",
                    disabled=not usable,
                    placeholder="No model available",
                    key="model",
                    help="Cloud is more accurate and faster; local runs entirely on this "
                         "machine with no quota. A cloud model that has hit its usage "
                         "limit is not offered. Claude API sends the pages to Anthropic "
                         "and is billed per page from your API credit.",
                )
            with recheck_col:
                if st.button("Re-check", width="stretch", key="recheck",
                             disabled=processing):
                    # Both have to go: the cache holds the stale probe, and the
                    # session set would otherwise keep re-blocking a recovered model.
                    get_model_availability.clear()
                    st.session_state["rate_limited"] = frozenset()
                    st.rerun()

            # Full width: at half the page the drop zone is too cramped to read.
            uploaded_file = st.file_uploader("Letter of Offer PDF", type=["pdf"],
                                             key="pdf_upload")

            if ollama_error is not None:
                st.error(f"Can't reach Ollama at {models.OLLAMA_HOST} - is it running? "
                         f"({ollama_error})")
            for name, reason in blocked:
                st.caption(f":material/block: **{name}** unavailable — {reason}")

            grounded = st.toggle(
                "Give the model the document's text",
                value=True,
                key="grounding",
                # Deliberately short and non-technical - the measured numbers and
                # the reasoning behind them live in DESIGN.md, not in a tooltip.
                help="**On (recommended):** the app reads the document's text first and "
                     "gives it to the AI as a reference. More accurate, and needed to "
                     "highlight where each value was found.\n\n"
                     "**Off:** the AI reads the page as a picture only. Faster, but less "
                     "accurate and no highlights.\n\n"
                     "Turn it off for quick checks, or on clear computer-generated PDFs. "
                     "Leave it on for scans and handwriting.",
            )
            # A slot that always exists, filled only when the toggle is off.
            # Adding or removing an element here instead would shift everything
            # below it in the element tree, and Streamlit derives an unkeyed
            # widget's identity partly from that position - which briefly
            # rendered a SECOND Extract button on every toggle, until
            # reconciliation dropped the stale one.
            
            # grounding_note = st.empty()
            # if not grounded:
                # grounding_note.caption(
                #     ":material/bolt: Faster, but expect lower accuracy. Digital PDFs keep "
                #     "their highlights; scanned ones lose them, since finding them there "
                #     "means the slow step this skips."
                # )

        # Asked once per session (the answer changes only when the engine's
        # schema does) and read before the picker renders, so the checkboxes
        # below are the engine's own list rather than a copy of it kept here.
        all_fields = list_engine_fields(engine_fingerprint())

        with st.container():
            st.markdown('<div class="sub-step"><small>Fields</small></div>',
                        unsafe_allow_html=True)
            keep_flags = {}
            with st.expander("Fields to extract", expanded=not has_result):
                if not all_fields:
                    # An unreachable engine is the extraction's problem to report,
                    # with a far better message than this one could give. Don't
                    # block the run over a picker; extract everything.
                    st.caption(
                        ":material/info: Couldn't read the field list from the engine, so "
                        "every field will be extracted. Extraction itself will say why if "
                        "the engine really is unavailable."
                    )
                else:
                    # One tick can cover more than one field - see fields.picker_units.
                    units = field_picker.picker_units(all_fields, VLLM_ALIASES.values())

                    # Seed the defaults into session state instead of passing
                    # value=True to each checkbox. A widget that carries BOTH an
                    # explicit default and a session-state value written for its
                    # key draws a Streamlit warning - which is precisely what
                    # Clear and Select all do. State is the single source of truth.
                    for key, _, _ in units:
                        if field_key(key) not in st.session_state:
                            st.session_state[field_key(key)] = True

                    # Read from session state rather than from the checkboxes,
                    # which don't exist yet on this run - the count renders above them.
                    kept_now = sum(len(members) for key, _, members in units
                                   if st.session_state.get(field_key(key), True))
                    count_col, all_col, none_col = st.columns([3, 1, 1],
                                                              vertical_alignment="bottom")
                    with count_col:
                        st.caption(
                            f"**{kept_now} of {len(all_fields)}** selected. An unticked "
                            "field is never asked for, so it costs no time — and no row "
                            "in the table."
                        )
                    with all_col:
                        select_all = st.button("Select all", key="fields_all",
                                               width="stretch", disabled=processing)
                    with none_col:
                        clear_all = st.button("Clear", key="fields_none",
                                              width="stretch", disabled=processing)
                    if select_all or clear_all:
                        # Safe to write these keys here: the checkboxes they belong
                        # to are created below, so on this run they don't exist yet.
                        for key, _, _ in units:
                            st.session_state[field_key(key)] = bool(select_all)
                        st.rerun()

                    # Grouped because 28 checkboxes in one list is a wall - see
                    # fields.py, which owns the grouping.
                    section_cols = st.columns(2)
                    sections = field_picker.sectioned_units(all_fields, VLLM_ALIASES.values())
                    for i, (title, section) in enumerate(sections):
                        with section_cols[i % len(section_cols)]:
                            st.markdown(f"**{title}**")
                            for key, label, members in section:
                                keep_flags[key] = st.checkbox(
                                    label, key=field_key(key), disabled=processing,
                                    help=("These fields answer one comparison row, so they "
                                          "are extracted together — a form with no \"Attn:\" "
                                          "line still names the same person under "
                                          "\"Witnessed by\"."
                                          ) if len(members) > 1 else None)

            excluded = (field_picker.excluded_from_units(all_fields, units, keep_flags)
                        if all_fields else [])
            nothing_selected = bool(all_fields) and len(excluded) == len(all_fields)
            if nothing_selected:
                st.caption(":material/warning: Pick at least one field to extract.")

    # Full width, and keyed so it keeps one identity across reruns no matter
    # what is rendered above it - belt and braces with the slots above.
    extract_clicked = st.button(
        "Processing..." if processing else ("Extract again" if has_result else "Extract"),
        type="primary", width="stretch", key="extract",
        disabled=processing or not usable or nothing_selected,
    )

    if extract_clicked and not processing:
        if not sale_no.strip():
            st.error("Please key in a Sale No.")
        elif uploaded_file is None:
            st.error("Please upload a PDF.")
        elif not selected_model:
            st.error("No model is available to run the extraction.")
        elif not os.environ.get("STAGING_API_PASSWORD"):
            st.error(
                "STAGING_API_PASSWORD not found - copy .env.example to .env at the "
                "repo root and fill it in."
            )
        else:
            st.session_state["is_processing"] = True
            st.rerun()

    if processing:
        # The document stays in view while the run takes its minute or more.
        render_doc_pane(uploaded_file, None)
        try:
            with st.spinner("Extracting — reading the page, then asking the model…"):
                password = os.environ.get("STAGING_API_PASSWORD")
                sale = fetch_sale(sale_no.strip(), password)
                api_fields = build_canonical_ground_truth(sale)

                with tempfile.TemporaryDirectory() as tmpdir:
                    pdf_path = Path(tmpdir) / uploaded_file.name
                    pdf_path.write_bytes(uploaded_file.getvalue())

                    start = time.monotonic()
                    result = run_hybrid_extraction(pdf_path, tmpdir, selected_model, grounded,
                                                   exclude=excluded)
                    elapsed = time.monotonic() - start

                    # Copy the PDF bytes out of the temp dir before it's cleaned
                    # up - the document pane needs them to draw the pages.
                    st.session_state["pdf_bytes"] = uploaded_file.getvalue()
                    st.session_state["pdf_name"] = uploaded_file.name

            extracted_fields = result.get("fields", {})
            # Snapshot what this run left out, so the table below reflects the
            # run rather than whatever the checkboxes say later. The score
            # follows: a field nobody asked for is not one the extractor got wrong.
            excluded_now = set(excluded)
            st.session_state["excluded"] = excluded_now

            rows, verifiable, correct = [], 0, 0
            for field in CANONICAL_FIELDS:
                aliases = VLLM_ALIASES.get(field, [field])
                if field_picker.row_is_hidden(
                        field_picker.extractable_aliases(aliases, all_fields), excluded_now):
                    continue
                api_value = api_fields.get(field)
                extracted_value = lookup_extracted(extracted_fields, aliases)
                has_api_value = api_value not in (None, "")
                if not has_api_value:
                    match_display = comparison_table.NA
                else:
                    verifiable += 1
                    kind = CANONICAL_FIELD_KINDS.get(field, "text")
                    matched = bool(extracted_value) and field_match(kind, api_value, extracted_value)
                    correct += matched
                    match_display = comparison_table.MATCH if matched else comparison_table.MISS
                rows.append({
                    "Field": field,
                    "Ground Truth (API)": api_value if has_api_value else comparison_table.NA,
                    "Extracted": (extracted_value if extracted_value not in (None, "")
                                  else comparison_table.NA),
                    "Match": match_display,
                })

            st.session_state["result"] = result
            st.session_state["comparison_df"] = rows
            st.session_state["elapsed"] = elapsed
            st.session_state["accuracy"] = correct / verifiable if verifiable else 0.0
            st.session_state["correct"] = correct
            st.session_state["verifiable"] = verifiable
            st.session_state["model_used"] = selected_model
            # What the summary chips describe: this run, not the widgets now.
            st.session_state["run_inputs"] = {
                "sale_no": sale_no.strip(), "file": uploaded_file.name,
                "model": selected_model, "grounded": grounded,
                "kept": len(all_fields) - len(excluded) if all_fields else "all",
                "total": len(all_fields) if all_fields else "all",
            }
            st.session_state["goto_results"] = True
            st.session_state.pop("row_filter", None)
        except RateLimited as e:
            # Take it out of the selector for the rest of the session, so the
            # next attempt can't pick the same dead model. Re-check clears this.
            st.session_state["rate_limited"] |= {str(e)}
            get_model_availability.clear()
            st.session_state["notice"] = (
                "warning",
                f"**{e}** hit its cloud usage limit, so it's no longer selectable. "
                "Pick another model and try again, or use Re-check once the quota "
                "window resets.",
            )
        except Exception as e:
            st.session_state["notice"] = ("error", f"Extraction failed: {e}")
        finally:
            st.session_state["is_processing"] = False
            st.rerun()

    # Anything rendered inside the block above is discarded by its st.rerun(),
    # so failures have to survive as state and be drawn on the following run.
    notice = st.session_state.pop("notice", None)
    if notice:
        level, message = notice
        getattr(st, level)(message)

    pages = None
    if has_result:
        result = st.session_state["result"]
        # The compared rows, then every other ticked field that was extracted
        # but has no record value to check - Tenant Address, Monthly Rental,
        # the tenancy dates. Built here rather than at extraction time, so a
        # result already on screen shows them too. They are marked No record
        # and never count towards the score.
        #
        # "Covered" means covered by a compared row that is IN this table - not
        # by every row the alias map knows of. Otherwise a field whose compared
        # row isn't here (a result built before the comparison list changed)
        # would be taken as shown and silently dropped. The rule this keeps: a
        # ticked field always has a row, its own or a compared one.
        compared = st.session_state["comparison_df"]
        shown_aliases = {row["Field"]: VLLM_ALIASES.get(row["Field"], [row["Field"]])
                         for row in compared}
        rows = comparison_table.arrange_rows(compared, comparison_table.unrecorded_rows(
            result.get("fields") or {},
            all_fields or list((result.get("fields") or {}).keys()),
            shown_aliases, st.session_state.get("excluded") or set()))
        has_where = bool(result.get("_ocr_boxes"))

        # Numbered once here and reused by the document pane, so a table
        # row's badge and the page's are the same numbering by construction.
        if st.session_state.get("pdf_bytes"):
            images = page_images(st.session_state["pdf_bytes"])
            overlays = docview.page_overlays(collect_field_boxes(result), len(images))
            pages = [dict(image, overlays=page_overlays, legend=docview.legend(page_overlays))
                     for image, page_overlays in zip(images, overlays)]
        badge_index = comparison_table.badges_by_field(pages or [])
        page_count = len(pages or [])
        # "Where" and "Links" are on-screen only: added to copies, never to
        # the rows the CSV is written from. Links are what tie a row to its
        # highlights for the click-to-select linking.
        shown = [dict(row,
                      Where=comparison_table.where_for_row(
                          row["Field"], badge_index, page_count, VLLM_ALIASES),
                      Links=comparison_table.links_for_row(
                          row["Field"], badge_index, VLLM_ALIASES))
                 for row in rows]

        with st.container(key="card_results"):
            title_col, download_col = st.columns([2, 1], vertical_alignment="bottom")
            title_col.markdown(card_title("Results"), unsafe_allow_html=True)
            download_col.download_button(
                "Download CSV", icon=":material/download:",
                data=comparison_table.csv_bytes(rows),
                file_name="comparison.csv",
                mime="text/csv",
                width="content",
            )
            # What this run used - the run, not whatever Step 1 says now.
            run = st.session_state.get("run_inputs", {})
            st.markdown(chips_html([
                ("Sale No.", run.get("sale_no", "—")),
                ("File", run.get("file", "—")),
                ("Model", run.get("model", "—")),
                ("Fields", f"{run.get('kept', '—')} of {run.get('total', '—')}"),
                ("Text grounding", "On" if run.get("grounded", True) else "Off"),
            ]), unsafe_allow_html=True)

            left_out = st.session_state.get("excluded") or set()
            if left_out:
                # Otherwise a short table reads as fields the extractor lost.
                st.caption(
                    f":material/filter_alt: {len(left_out)} field(s) were left out of "
                    "this run, so they have no row below and count towards nothing."
                )

            st.markdown(metrics_html(st.session_state["accuracy"], st.session_state["correct"],
                                     st.session_state["verifiable"], st.session_state["elapsed"]),
                        unsafe_allow_html=True)

            counts = comparison_table.filter_counts(shown, has_where)
            # Separate pills, not a joined segment: each filter is its own choice.
            chosen = st.pills(
                "Show rows", options=comparison_table.FILTERS,
                format_func=lambda f: f"{f}  {counts[f]}",
                default=comparison_table.ALL, key="row_filter",
                label_visibility="collapsed",
            )
            visible = comparison_table.filter_rows(shown, chosen, has_where)

            # Our own markup rather than st.dataframe or st.table - see
            # table.py for why. Every value shows in full.
            columns = (comparison_table.COLUMNS_WITH_WHERE if has_where
                       else comparison_table.COLUMNS)
            if visible:
                st.markdown(comparison_table.comparison_table_html(visible, columns),
                            unsafe_allow_html=True)
            else:
                st.caption("No rows in this view.")
            if has_where:
                st.caption("A field with no number was read from the page image but "
                           "couldn't be tied to a line of text. Check it by eye.")

render_doc_pane(uploaded_file, pages)
