"""lof2data: standalone UI for the hybrid (vision + OCR-grounding +
deterministic choice-field override) LOF extraction approach - shows a
Sale No.-driven ground-truth comparison, plus bounding boxes on the source
document showing exactly where each extracted value came from.
"""

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
from dotenv import load_dotenv
from PIL import Image, ImageColor, ImageDraw, ImageFont

import boxes
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

# The palette. Up here rather than beside the CSS that also uses it, because
# the page overlay below needs DARK for its labels and is defined first.
PRIMARY = "#4FA8E0"
DARK = "#1E4D6B"
BG = "#F5F8FA"

# One colour for every box. The blue/orange split used to encode HOW the box
# was located - the model's own citation vs. a text search - but that is a
# detail of the lookup, not something a reader of the document needs to act
# on. Both mean the same thing to them: this is where the value was read.
BOX_COLOR = "#4FA8E0"

# Drawn as a marker wash rather than an outline: a reader is checking whether
# the highlighted text IS the value, so the text has to stay readable through
# it - an outline puts a hard 3px line right where a tight box clips the
# glyphs. Alpha is the whole trade: high enough to find at a glance on a grey
# scan, low enough to read printed text through. 64/255 holds both.
HIGHLIGHT_ALPHA = 64
# A little room above and below, so the wash reads as a highlighter stroke
# rather than a box that happens to be filled in.
HIGHLIGHT_PAD_PX = 2
# Each highlight is numbered rather than named on the page, and the names go
# in a legend under it. Field names written next to the wash were unreadable
# on a real form for two reasons at once: several fields cite the SAME OCR
# line, so their labels landed on each other exactly, and on a form whose
# lines are ~28px apart at render scale, a label sitting above its wash falls
# on the line above's text. A badge collides with neither - it is small, it
# sits in the left margin, and fields sharing a line share one badge.
BADGE_FILL = DARK
BADGE_TEXT_COLOR = "#FFFFFF"
BADGE_TEXT_SIZE_PX = 14
BADGE_HEIGHT_PX = 22
BADGE_MIN_WIDTH_PX = 22   # a pill, so a two-digit number widens instead of clipping
BADGE_GAP_PX = 5          # between the badge and the wash it belongs to
LEGEND_COLUMNS = 3


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
        for part in boxes.split_occurrences(value):
            b = boxes.find_box_by_text(ocr_boxes, part)
            if b is not None:
                add_box(name, b, BOX_COLOR)

    return found


def highlight_fill(color, alpha=HIGHLIGHT_ALPHA):
    """A hex colour as RGBA at the wash's alpha."""
    return ImageColor.getrgb(color) + (alpha,)


def label_font(size=BADGE_TEXT_SIZE_PX):
    """A readable font for the badge numbers, falling back to the bitmap
    default. Pillow's sized default needs FreeType, which the container has
    but a stripped environment might not - and a missing font must not cost
    the whole overlay."""
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def regions_on_page(field_boxes, page_num):
    """[(box, [field name, ...]), ...] for one page, in reading order.

    Fields that resolved to the SAME box are one region, not several: on a
    real form the negotiator, the property address and the property type all
    cite lines within a few pixels of each other, and drawing a separate
    marker per field printed them on top of one another. One badge listing
    both names says the true thing anyway - this line is where both were
    read from.

    Reading order (top to bottom, then left to right) so the numbers run down
    the page the way someone checking it would."""
    regions = {}
    for f in field_boxes:
        if f["page"] != page_num:
            continue
        key = tuple(round(v, 4) for v in f["box"])
        entry = regions.setdefault(key, {"color": f["color"], "labels": []})
        # By display name, so the same field found by both passes - once under
        # the raw group path, once under its flat name - is listed once.
        name = f.get("display") or f["label"]
        if name not in entry["labels"]:
            entry["labels"].append(name)
    ordered = sorted(regions.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    return [(box, entry["color"], entry["labels"]) for box, entry in ordered]


def draw_badge(draw, number, px0, px1, py0, py1, page_width, font):
    """The numbered pill for one region.

    Placed in the left margin beside its wash, which is where the eye looks
    for a marker and where a form leaves room. A clause printed hard against
    the left edge has no margin, so the badge goes to the RIGHT of the wash
    instead - a line's tail is nearly always blank, while its head is the
    text being pointed at. Only if neither side fits does it sit on the wash
    itself, covering a character or two, which beats not being drawn."""
    text = str(number)
    width = max(BADGE_MIN_WIDTH_PX, draw.textlength(text, font=font) + 10)
    centre_y = (py0 + py1) / 2
    top = centre_y - BADGE_HEIGHT_PX / 2

    left = px0 - BADGE_GAP_PX - width
    if left < 0:
        right_of = px1 + BADGE_GAP_PX
        left = right_of if right_of + width <= page_width else px0
    draw.rounded_rectangle([left, top, left + width, top + BADGE_HEIGHT_PX],
                           radius=BADGE_HEIGHT_PX / 2, fill=BADGE_FILL)
    draw.text((left + width / 2, centre_y), text, fill=BADGE_TEXT_COLOR,
              font=font, anchor="mm")


def render_annotated_pages(pdf_bytes, field_boxes, dpi=150):
    """Every page of the PDF, highlighted and numbered.

    Returns [{"image": PIL.Image, "legend": [(number, "Field · Field"), ...]},
    ...] - one entry per page, in order. Numbering restarts on each page,
    since each page is shown with its own legend directly beneath it.

    Every wash for a page goes onto ONE overlay layer, composited in a single
    pass. Drawn straight onto the page instead, two fields sharing a line
    (the deposits on an Item/Description/RM row) would stack their alpha and
    come out darker than the rest - reading as a stronger match when it is
    only a busier line. On one layer the second draw replaces the first's
    pixels, so every highlight has the same weight.

    Badges are drawn AFTER compositing, at full strength: the wash is meant to
    be seen through, the number is not."""
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
            regions = regions_on_page(field_boxes, page_num)

            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            wash = ImageDraw.Draw(overlay)
            placed = []
            for number, (box, color, labels) in enumerate(regions, start=1):
                x0, y0, x1, y1 = box
                px0, py0 = x0 * img.width, y0 * img.height
                px1, py1 = x1 * img.width, y1 * img.height
                py0, py1 = max(py0 - HIGHLIGHT_PAD_PX, 0), min(py1 + HIGHLIGHT_PAD_PX, img.height)
                wash.rectangle([px0, py0, px1, py1], fill=highlight_fill(color))
                placed.append((number, px0, px1, py0, py1))

            img = Image.alpha_composite(img, overlay).convert("RGB")
            draw = ImageDraw.Draw(img)
            font = label_font()
            for number, px0, px1, py0, py1 in placed:
                draw_badge(draw, number, px0, px1, py0, py1, img.width, font)

            legend = [(number, " · ".join(labels))
                      for number, (_, _, labels) in enumerate(regions, start=1)]
            pages.append({"image": img, "legend": legend})
    return pages


def render_legend(legend):
    """The number -> field name key for one page, under its image.

    Filled column-major so the numbers read straight down a column, matching
    the order they run down the page. A page whose fields all failed to
    locate has an empty legend and renders nothing rather than an empty row
    of columns."""
    if not legend:
        return
    per_column = -(-len(legend) // LEGEND_COLUMNS)   # ceil, no math import
    columns = st.columns(LEGEND_COLUMNS)
    for i, column in enumerate(columns):
        for number, names in legend[i * per_column:(i + 1) * per_column]:
            column.markdown(f"**{number}.**&nbsp; {names}", unsafe_allow_html=True)


st.set_page_config(page_title="lof2data", page_icon=":material/diamond:", layout="wide")

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; }}
.hero-title {{ color: {DARK}; font-size: 2.5rem; font-weight: 700; text-align: center; margin-bottom: 0.25rem; }}
.hero-tagline {{ color: #4A5A66; text-align: center; font-size: 1.1rem; margin-bottom: 1.5rem; }}
.step-card {{
    background: white; border-radius: 12px; padding: 1.75rem 1.25rem;
    border: 1px solid #E1E8ED; text-align: center; height: 160px;
    box-sizing: border-box; display: flex; flex-direction: column;
    /* flex-start, not center: centring the whole block makes a card whose
       description wraps to two lines sit its title higher than its
       neighbours', so the row of titles looks ragged. */
    justify-content: flex-start;
}}
.step-card h4 {{ color: {DARK}; margin: 0 0 0.6rem 0; line-height: 1.3; }}
.step-card p {{ color: #6B7A85; font-size: 0.9rem; margin: 0; line-height: 1.45; }}
.section-label {{
    color: white; background: {DARK}; display: inline-block;
    padding: 0.2rem 0.9rem; border-radius: 999px; font-size: 0.85rem;
    font-weight: 600; letter-spacing: 0.03em; margin-bottom: 0.5rem;
}}
.section-title {{ color: {DARK}; font-size: 1.4rem; font-weight: 700; margin: 0 0 0.75rem 0; }}
/* The comparison table. Markup is ours (table.py), so these selectors are a
   contract with that file rather than with Streamlit's DOM. */
.comparison-wrap {{
    border: 1px solid #DDE5EA; border-radius: 10px; overflow: hidden;
    background: white;
    /* The last row otherwise sits flush against the results card's own edge,
       so the two borders read as one and the table looks cropped. */
    margin: 0 0 1.25rem 0;
}}
table.comparison {{
    width: 100%; border-collapse: collapse; font-size: 0.93rem;
}}
/* The header has to be unmistakably NOT data: darker ground, heavier and
   smaller type, letter-spaced, and a rule under it. Same weight and colour as
   the rows left the eye with nothing to anchor on. */
table.comparison thead th {{
    background: #E7EEF3; color: {DARK};
    font-weight: 700; font-size: 0.74rem; letter-spacing: 0.06em;
    text-transform: uppercase; text-align: left; white-space: nowrap;
    padding: 0.65rem 0.85rem; border-bottom: 2px solid {PRIMARY};
}}
/* The field name is a row heading, not a value - semibold, and never wrapped
   mid-name if it can be helped. */
table.comparison tbody th {{
    text-align: left; font-weight: 600; color: {DARK};
    padding: 0.55rem 0.85rem; vertical-align: top; width: 22%;
}}
table.comparison tbody td {{
    color: #37474F; padding: 0.55rem 0.85rem; vertical-align: top;
}}
/* Striping and a hover row: on a 28-row table with wrapped addresses, the eye
   loses its place between the truth and the extracted value otherwise. */
table.comparison tbody tr:nth-child(even) {{ background: #FAFCFD; }}
table.comparison tbody tr:hover {{ background: #EFF7FC; }}
table.comparison tbody tr + tr th,
table.comparison tbody tr + tr td {{ border-top: 1px solid #E6ECF0; }}
table.comparison td.match-col {{ text-align: center; width: 5rem; }}
/* Plain glyphs, not emoji: an emoji ignores colour on most platforms, so a
   wrong row would print in the same ink as a right one. */
.comparison .mark {{ font-size: 1.05rem; font-weight: 700; }}
.comparison .mark.ok {{ color: #2E7D53; }}
.comparison .mark.bad {{ color: #C0392B; }}
.comparison .mark.na {{ color: #9AA8B2; font-weight: 400; }}
div[data-testid="stButton"] button:disabled,
div[data-testid="stButton"] button:disabled:hover,
div[data-testid="stButton"] button:disabled:focus,
div[data-testid="stButton"] button:disabled:active {{
    background-color: #9CCBEC !important; color: white !important;
    border: 1px solid #9CCBEC !important; opacity: 1 !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero-title">lof2data</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-tagline">Extract Letter of Offer fields with vision + OCR grounding, '
    'verified against the real record.</div>',
    unsafe_allow_html=True,
)

st.markdown("#### How it works")
STEPS = [
    ("Upload & Identify", "Upload the PDF, enter the Sale No."),
    ("Choose Fields", "Tick only what you need — the rest is never asked for"),
    ("Vision + OCR Grounding", "The image is read directly, with OCR text as a supporting reference"),
    ("Compare & Review", "Matched against the real record, field by field"),
]
# Derived from STEPS, not hardcoded: a fixed count leaves zip() silently
# dropping the spare columns, which reads as a gap on the right when a step
# is added or removed.
cols = st.columns(len(STEPS))
for col, (title, desc) in zip(cols, STEPS):
    with col:
        st.markdown(
            f'<div class="step-card"><h4>{title}</h4><p>{desc}</p></div>',
            unsafe_allow_html=True,
        )

st.write("")

if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "rate_limited" not in st.session_state:
    st.session_state["rate_limited"] = frozenset()

with st.container(border=True):
    st.markdown('<span class="section-label">STEP 1</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input</div>', unsafe_allow_html=True)

    in_col1, in_col2 = st.columns(2)
    with in_col1:
        sale_no = st.text_input("Sale No.", placeholder="e.g. 72834", help="Used to fetch ground truth from the sales API for comparison.")
    with in_col2:
        uploaded_file = st.file_uploader("Upload LOF PDF", type=["pdf"])

    try:
        usable, blocked = get_model_availability(st.session_state["rate_limited"])
        ollama_error = None
    except Exception as e:
        usable, blocked, ollama_error = [], [], e

    # vertical_alignment="bottom" so Re-check sits on the selectbox's baseline.
    # The label above the selectbox makes that column taller, and the previous
    # fix - an empty st.write("") as a spacer - overshot and pushed the button
    # below the input instead of level with it.
    model_col, recheck_col = st.columns([3, 1], vertical_alignment="bottom")
    with model_col:
        selected_model = st.selectbox(
            "Model",
            options=usable,
            index=usable.index(models.pick_default(usable)) if usable else None,
            format_func=lambda name: f"{name}  ·  {models.label_for(name)}",
            disabled=not usable,
            placeholder="No model available",
            help="Cloud is more accurate and faster; local runs entirely on this "
                 "machine with no quota. A cloud model that has hit its usage "
                 "limit is not offered. Claude API sends the pages to Anthropic "
                 "and is billed per page from your API credit.",
        )
    with recheck_col:
        if st.button("Re-check", use_container_width=True, key="recheck",
                     disabled=st.session_state["is_processing"]):
            # Both have to go: the cache holds the stale probe, and the
            # session set would otherwise keep re-blocking a recovered model.
            get_model_availability.clear()
            st.session_state["rate_limited"] = frozenset()
            st.rerun()

    if ollama_error is not None:
        st.error(f"Can't reach Ollama at {models.OLLAMA_HOST} - is it running? ({ollama_error})")
    for name, reason in blocked:
        st.caption(f":material/block: **{name}** unavailable — {reason}")

    grounded = st.toggle(
        "Give the model the document's text",
        value=True,
        key="grounding",
        # Deliberately short and non-technical - the measured numbers and the
        # reasoning behind them live in DESIGN.md, not in a tooltip.
        help="**On (recommended):** the app reads the document's text first and "
             "gives it to the AI as a reference. More accurate, and needed to "
             "highlight where each value was found.\n\n"
             "**Off:** the AI reads the page as a picture only. Faster, but less "
             "accurate and no highlights.\n\n"
             "Turn it off for quick checks, or on clear computer-generated PDFs. "
             "Leave it on for scans and handwriting.",
    )
    # A slot that always exists, filled only when the toggle is off. Adding or
    # removing an element here instead would shift everything below it in the
    # element tree, and Streamlit derives an unkeyed widget's identity partly
    # from that position - which briefly rendered a SECOND Extract button on
    # every toggle, until reconciliation dropped the stale one.
    grounding_note = st.empty()
    if not grounded:
        grounding_note.caption(
            ":material/bolt: Faster, but expect lower accuracy. Digital PDFs keep "
            "their highlights; scanned ones lose them, since finding them there "
            "means the slow step this skips."
        )

# Asked once per session (the answer changes only when the engine's schema
# does) and read before the picker renders, so the checkboxes below are the
# engine's own list rather than a copy of it kept over here.
all_fields = list_engine_fields(engine_fingerprint())
processing = st.session_state["is_processing"]

with st.container(border=True):
    st.markdown('<span class="section-label">STEP 2</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Fields to Extract</div>', unsafe_allow_html=True)

    keep_flags = {}
    if not all_fields:
        # An unreachable engine is the extraction's problem to report, with a
        # far better message than this one could give. Don't block the run
        # over a picker; extract everything, as it did before this step.
        st.caption(
            ":material/info: Couldn't read the field list from the engine, so "
            "every field will be extracted. Extraction itself will say why if "
            "the engine really is unavailable."
        )
    else:
        # One tick can cover more than one field - see fields.picker_units.
        units = field_picker.picker_units(all_fields, VLLM_ALIASES.values())

        # Seed the defaults into session state instead of passing value=True to
        # each checkbox. A widget that carries BOTH an explicit default and a
        # session-state value written for its key draws a Streamlit warning -
        # "created with a default value but also had its value set via the
        # Session State API" - which is precisely what Clear and Select all do.
        # State is the single source of truth for a tick from here on.
        for key, _, _ in units:
            if field_key(key) not in st.session_state:
                st.session_state[field_key(key)] = True

        # Read from session state rather than from the checkboxes, which don't
        # exist yet on this run - the count has to render above them.
        kept_now = sum(len(members) for key, _, members in units
                       if st.session_state.get(field_key(key), True))
        count_col, all_col, none_col = st.columns([4, 1, 1], vertical_alignment="bottom")
        with count_col:
            st.caption(
                f"**{kept_now} of {len(all_fields)}** selected. Untick anything you "
                "don't need: an unticked field is never put to the model, so it "
                "costs no time — and no column in the table. The list is the "
                "form's own; there is nothing to add to it. Two fields that "
                "answer the same comparison row are offered as one tick."
            )
        with all_col:
            select_all = st.button("Select all", key="fields_all",
                                   use_container_width=True, disabled=processing)
        with none_col:
            clear_all = st.button("Clear", key="fields_none",
                                  use_container_width=True, disabled=processing)
        if select_all or clear_all:
            # Safe to write these keys here: the checkboxes they belong to are
            # created below, so on this run they don't exist yet, and the rerun
            # renders them from the state just written.
            for key, _, _ in units:
                st.session_state[field_key(key)] = bool(select_all)
            st.rerun()

        # Grouped because 28 checkboxes in one list is a wall - see fields.py,
        # which owns the grouping and drops a section the engine no longer has.
        section_cols = st.columns(3)
        sections = field_picker.sectioned_units(all_fields, VLLM_ALIASES.values())
        for i, (title, section) in enumerate(sections):
            with section_cols[i % len(section_cols)]:
                st.markdown(f"**{title}**")
                for key, label, members in section:
                    keep_flags[key] = st.checkbox(
                        label, key=field_key(key), disabled=processing,
                        help=("These fields answer one comparison row, so they are "
                              "extracted together — a form with no \"Attn:\" line "
                              "still names the same person under \"Witnessed by\"."
                              ) if len(members) > 1 else None)
                st.write("")

    excluded = (field_picker.excluded_from_units(all_fields, units, keep_flags)
                if all_fields else [])
    nothing_selected = bool(all_fields) and len(excluded) == len(all_fields)
    if nothing_selected:
        st.caption(":material/warning: Pick at least one field to extract.")

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        # Explicit key so the button keeps one identity across reruns no matter
        # what is rendered above it - belt and braces with the slot above.
        extract_clicked = st.button(
            "Processing..." if processing else "Extract",
            type="primary", use_container_width=True, key="extract",
            disabled=processing or not usable or nothing_selected,
        )

if extract_clicked and not st.session_state["is_processing"]:
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

if st.session_state["is_processing"]:
    try:
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

            # Copy the PDF bytes out of the temp dir before it's cleaned up -
            # the results section needs the file to still exist for
            # rendering page images.
            st.session_state["pdf_bytes"] = uploaded_file.getvalue()
            st.session_state["pdf_name"] = uploaded_file.name

        extracted_fields = result.get("fields", {})
        # Snapshot what this run left out, so the table below reflects the run
        # rather than whatever the checkboxes say later. The score follows: a
        # field nobody asked for is not a field the extractor got wrong.
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

# Anything rendered inside the block above is discarded by its st.rerun(), so
# failures have to survive as state and be drawn on the following run instead.
notice = st.session_state.pop("notice", None)
if notice:
    level, message = notice
    getattr(st, level)(message)

st.write("")

if "comparison_df" in st.session_state:
    with st.container(border=True):
        st.markdown('<span class="section-label">STEP 3</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        left_out = st.session_state.get("excluded") or set()
        if left_out:
            # Otherwise a short table reads as fields the extractor lost.
            st.caption(
                f":material/filter_alt: {len(left_out)} field(s) were left out of "
                "this run, so they have no row below and count towards nothing."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Time taken", f"{st.session_state['elapsed']:.1f}s")
        m2.metric("Accuracy", f"{st.session_state['accuracy']:.0%}")
        m3.metric("Fields matched", f"{st.session_state['correct']}/{st.session_state['verifiable']}")
        # m4.metric("Model", st.session_state["model_used"])

        tab_compare, tab_source = st.tabs(["Field Comparison", "Source Document"])

        with tab_compare:
            df = pd.DataFrame(st.session_state["comparison_df"])

            _, download_col = st.columns([4, 1])
            with download_col:
                st.download_button(
                    "Download CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="comparison.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            # Our own markup rather than st.dataframe or st.table - see
            # table.py for why. Every value shows in full, and the header is
            # styled by this app's stylesheet instead of Streamlit's.
            st.markdown(
                comparison_table.comparison_table_html(st.session_state["comparison_df"]),
                unsafe_allow_html=True,
            )

        with tab_source:
            # Every box is anchored to an OCR line, so an ungrounded run has
            # nothing to draw. Say that rather than render an unannotated page
            # that looks like the boxes silently failed.
            if not st.session_state["result"].get("_ocr_boxes"):
                st.info(
                    "No source boxes for this run — OCR grounding was off, so there "
                    "are no recognized text lines to anchor them to. Re-run with the "
                    "**OCR grounding** toggle on to see where each value was read from."
                )
            else:
                st.caption(
                    "Each numbered highlight marks where a value was read from — the "
                    "legend under each page says which field is which. A field listed "
                    "nowhere was read from the page image but couldn't be matched to any "
                    "line of the document's text — worth checking by eye."
                )
            field_boxes = collect_field_boxes(st.session_state["result"])
            pages = render_annotated_pages(st.session_state["pdf_bytes"], field_boxes)
            for i, page in enumerate(pages, start=1):
                st.image(page["image"], caption=f"Page {i}", use_container_width=True)
                render_legend(page["legend"])
