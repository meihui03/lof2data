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
from PIL import Image, ImageDraw

import boxes
import models
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
    "Tenant Name": ["Tenant Name(s)/Company"],
    "Tenant IC": ["Tenant NRIC/Passport No."],
    "Landlord Name": ["Landlord Name(s)"],
    "Landlord IC": ["Landlord NRIC/Passport No."],
}

BOX_COLOR_CITED = "#4FA8E0"
BOX_COLOR_MATCHED = "#E08A3E"


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


def run_hybrid_extraction(pdf_path, output_dir, model):
    if not ENGINE_PYTHON.exists():
        raise RuntimeError(
            f"Extraction engine interpreter not found at {ENGINE_PYTHON}. "
            "Create it with:  python3 -m venv engine/.venv && "
            "engine/.venv/bin/pip install -r engine/requirements.txt"
        )
    result = subprocess.run(
        [str(ENGINE_PYTHON), "extractor.py", "--pdf", str(pdf_path),
         "--model", model, "--output-dir", str(output_dir)],
        cwd=str(ENGINE_DIR), capture_output=True, text=True,
    )
    if result.returncode != 0:
        if models.looks_rate_limited(result.stderr):
            raise RateLimited(model)
        raise RuntimeError(result.stderr[-2000:])
    out_json = Path(output_dir) / f"{Path(pdf_path).stem}.json"
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


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
    layouts the page uses. Returns [{"label": str, "page": int, "box":
    (x0,y0,x1,y1) fractions, "color": str}, ...]."""
    ocr_boxes = result.get("_ocr_boxes", [])
    ocr_by_index = {b["index"]: b for b in ocr_boxes}
    raw = result.get("_raw", {})
    # {sub-field: option list} for the choice fields, published by the engine
    # from its own schema (extractor.choice_candidates).
    candidates_by_field = raw.get("_choice_candidates", {})
    found = []

    def add_box(label, b, color):
        found.append({
            "label": label, "page": b["page"],
            "box": (b["x0"], b["y0"], b["x1"], b["y1"]), "color": color,
        })

    def add_from_leaf(label, leaf, candidates=None):
        if not isinstance(leaf, dict):
            return
        # A model may cite more than one OCR line for a value that spans
        # them (and may cite it in shapes the prompt didn't ask for) - see
        # boxes.cited_boxes. Draw one box per cited line.
        cited = boxes.cited_boxes(ocr_by_index, leaf.get("source_index"))
        if cited:
            for b in cited:
                add_box(label, b, BOX_COLOR_CITED)
            return
        value = leaf.get("value")
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
            add_box(label, b, BOX_COLOR_MATCHED)

    for name, leaf in raw.get("fields", {}).items():
        add_from_leaf(name, leaf)

    for group_name, value in raw.get("groups", {}).items():
        occurrences = value if isinstance(value, list) else [value]
        for occurrence in occurrences:
            if not isinstance(occurrence, dict):
                continue
            for sub_field, leaf in occurrence.items():
                add_from_leaf(f"{group_name}.{sub_field}", leaf,
                              candidates_by_field.get(sub_field))

    return found


def render_annotated_pages(pdf_bytes, field_boxes, dpi=150):
    """Renders every page of the PDF and draws every box that belongs to
    it, labeled with its field name. Returns a list of PIL Images, one per
    page, in order."""
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_num, page in enumerate(pdf):
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            draw = ImageDraw.Draw(img)
            for f in field_boxes:
                if f["page"] != page_num:
                    continue
                x0, y0, x1, y1 = f["box"]
                px0, py0, px1, py1 = x0 * img.width, y0 * img.height, x1 * img.width, y1 * img.height
                draw.rectangle([px0, py0, px1, py1], outline=f["color"], width=3)
                draw.text((px0, max(py0 - 16, 0)), f["label"], fill=f["color"])
            images.append(img)
    return images


PRIMARY = "#4FA8E0"
DARK = "#1E4D6B"
BG = "#F5F8FA"

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

    model_col, recheck_col = st.columns([3, 1])
    with model_col:
        selected_model = st.selectbox(
            "Model",
            options=usable,
            index=usable.index(models.pick_default(usable)) if usable else None,
            format_func=lambda name: f"{name}  ·  {'cloud' if models.is_cloud(name) else 'local'}",
            disabled=not usable,
            placeholder="No model available",
            help="Cloud is more accurate and faster; local runs entirely on this "
                 "machine with no quota. A cloud model that has hit its usage "
                 "limit is not offered.",
        )
    with recheck_col:
        st.write("")
        if st.button("Re-check", use_container_width=True,
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

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        extract_clicked = st.button(
            "Processing..." if st.session_state["is_processing"] else "Extract",
            type="primary", use_container_width=True,
            disabled=st.session_state["is_processing"] or not usable,
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
            result = run_hybrid_extraction(pdf_path, tmpdir, selected_model)
            elapsed = time.monotonic() - start

            # Copy the PDF bytes out of the temp dir before it's cleaned up -
            # the results section needs the file to still exist for
            # rendering page images.
            st.session_state["pdf_bytes"] = uploaded_file.getvalue()
            st.session_state["pdf_name"] = uploaded_file.name

        extracted_fields = result.get("fields", {})
        rows, verifiable, correct = [], 0, 0
        for field in CANONICAL_FIELDS:
            api_value = api_fields.get(field)
            extracted_value = lookup_extracted(extracted_fields, VLLM_ALIASES.get(field, [field]))
            has_api_value = api_value not in (None, "")
            if not has_api_value:
                match_display = "—"
            else:
                verifiable += 1
                kind = CANONICAL_FIELD_KINDS.get(field, "text")
                matched = bool(extracted_value) and field_match(kind, api_value, extracted_value)
                correct += matched
                match_display = "✔️" if matched else "✖️"
            rows.append({
                "Field": field,
                "Ground Truth (API)": api_value if has_api_value else "—",
                "Extracted": extracted_value if extracted_value not in (None, "") else "—",
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
        st.markdown('<span class="section-label">STEP 2</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

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

            st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_source:
            st.caption(
                ":blue[:material/square:] blue = the OCR line the LLM cited directly · "
                ":orange[:material/square:] orange = located by matching the extracted "
                "value against the OCR text (used when the value came from a visual "
                "mark, e.g. a circled or struck-through choice)"
            )
            field_boxes = collect_field_boxes(st.session_state["result"])
            pages = render_annotated_pages(st.session_state["pdf_bytes"], field_boxes)
            for i, page_img in enumerate(pages, start=1):
                st.image(page_img, caption=f"Page {i}", use_container_width=True)
