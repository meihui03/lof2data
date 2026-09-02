"""Step 1: PDF -> page image -> PaddleOCR, just to see the raw OCR output
before anything else (LLM calls, choice-field detection, etc.) gets layered
on top.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import requests
from PIL import Image
import io

from paddleocr import PaddleOCR

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/") + "/api/chat"
LLM_MODEL = "minimax-m3:cloud"


# Template-free choice-field reader. Replaces the previous dependency on
# bbox-ocr-extractor's detect_selected_choice_native()/detect_circled_choice():
# those are accurate, but each needs to know WHICH template it is looking at
# to pick the right hand-calibrated pixel box, and choice-field boxes existed
# for only 2 of the 3 known templates - so any new layout read nothing at all.
# mark_detector instead locates a choice line from the schema's own candidate
# list, then tests each option's own box, so an unseen template needs no
# calibration. Verified reproducing the old per-template detectors on both
# marked documents, and correctly abstaining on the two that state the field
# as free text plus the one whose printed choice line is genuinely unmarked.
# sys.path so this resolves regardless of the caller's cwd (hybrid_app runs
# this file as a subprocess).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mark_detector  # noqa: E402

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            lang="en",
        )
    return _ocr_engine


def pdf_page_to_image(pdf_path, page_num=0, dpi=150):
    """Renders one page of a PDF to a numpy image array (no predefined
    per-field bounding boxes - this is the whole page, as-is)."""
    with fitz.open(pdf_path) as pdf:
        pix = pdf[page_num].get_pixmap(dpi=dpi)
        img = np.array(Image.open(io.BytesIO(pix.tobytes("png"))))
    return img


def pdf_to_images(pdf_path, dpi=150):
    """Renders every page of a PDF to a numpy image array, in order."""
    with fitz.open(pdf_path) as pdf:
        num_pages = pdf.page_count
    return [pdf_page_to_image(pdf_path, page_num, dpi) for page_num in range(num_pages)]


def render_pdf_to_base64_images(pdf_path, dpi=200, max_dim=2000):
    """Renders every page to a base64 PNG for a vision-LLM call, capping
    the longer side at max_dim. Some LOF scans (Form2-style) have a much
    larger physical page size than standard A4 - rendering at a fixed DPI
    regardless blows past Ollama's request-body size limit (confirmed
    earlier this session: a 25MB single-page image was rejected with
    "http: request body too large")."""
    images = []
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            pix = page.get_pixmap(dpi=dpi)
            if max(pix.width, pix.height) > max_dim:
                scale = max_dim / max(pix.width, pix.height)
                pix = page.get_pixmap(dpi=int(dpi * scale))
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
    return images


def run_ocr(img):
    """Runs PaddleOCR on a full page image and returns the raw result dict
    (rec_texts, rec_scores, rec_boxes, etc.) - nothing filtered or reordered
    yet, just to see what PaddleOCR actually gives us."""
    return get_ocr_engine().predict(img)[0]


FIELD_TYPES = ("text", "number", "yes/no", "categorical")


def get_field_schema():
    """Returns the fields to extract from a PropNex Letter of Offer form,
    each with its field name, description (used to help identify/locate the
    value), and expected value type - one of FIELD_TYPES:
      "text"        - free-form string (names, addresses, dates, IDs, ...)
      "number"      - a plain numeric value (amounts, counts)
      "yes/no"      - a boolean fact (none of the current fields need this
                      yet, but it's a valid type for future fields)
      "categorical" - one of a fixed, known set of options (a value that's
                      selected by circling/striking one of a printed list)
    """
    return [
        {"field": "Reference No.", "description": "The LOF reference number, often printed near the top-right or in the letterhead (e.g. '206537 (31/03)' or 'ERBHQ00008074').", "type": "text"},
        {"field": "Date", "description": "The date the Letter of Offer itself was issued/prepared, usually near the top of the form.", "type": "text"},
        # Negotiator/Agent Name, Witnessed by, Landlord Name(s)/NRIC,
        # Tenant Name(s)/Signatory/NRIC/Address, and the property fields
        # (Description of Property, Property Address, Title Details,
        # Premise Usage, Furnished Status) moved to get_group_schema()
        # below - the party roles can have more than one person (two
        # co-landlords, a company tenant plus its signatory), and the
        # property fields are grouped together the same way LlamaExtract's
        # own schema nests them under a single "property" object rather
        # than as flat top-level fields.
        {"field": "Tenancy Commencement Date", "description": "The date the tenancy begins.", "type": "text"},
        {"field": "Tenancy Expiration Date", "description": "The date the tenancy ends, ONLY if that end date is explicitly printed/written on the form. Do NOT calculate or infer it yourself from the commencement date plus the tenancy period - if only a start date and a duration (e.g. '2 years') are given with no explicit end date printed anywhere, leave this blank.", "type": "text"},
        {"field": "Period of Tenancy", "description": "The length of the tenancy term, e.g. '(2) Two years'.", "type": "text"},
        {"field": "Option to Renew", "description": "The renewal option period, e.g. '(1) One year'.", "type": "text"},
        {"field": "Monthly Rental", "description": "The monthly rental amount, usually prefixed 'RM'.", "type": "number"},
        {"field": "Security Deposit", "description": "The security deposit amount. On templates with an Item/Description/RM table, this is the row labelled 'Security Deposit' - read the number from the RM column, NOT the Description column.", "type": "number"},
        {"field": "Utilities Deposit", "description": "The utilities deposit amount, for water & electricity. On templates with an Item/Description/RM table, read the number from the RM column on the 'Utilities Deposit' row, NOT the Description column.", "type": "number"},
        {"field": "Advance Rental", "description": "Any advance rental amount paid upfront, if this template has a separate line for it. On templates with an Item/Description/RM table, read the number from the RM column on the 'Advance Rental' row, NOT the Description column.", "type": "number"},
        {"field": "Other Deposit (if any)", "description": "Any other deposit amount not covered by security/utilities/advance rental (e.g. access card deposit). On templates with an Item/Description/RM table, read the RM column, not the Description column.", "type": "number"},
        {"field": "Total Payable", "description": "The total amount payable upon execution of the tenancy agreement, summing the deposits/rental above.", "type": "number"},
        {"field": "Earnest Deposit Amount", "description": "The earnest deposit amount enclosed with this Letter of Offer - specifically the number in the sentence 'The Tenant(s) hereby enclosed herewith the sum of RM ___' (or similar wording).", "type": "number"},
        {"field": "Legal Cost / Stamp Duty Allocation", "description": "Which party bears legal cost and stamp duty, if stated as a separate clause. Return the FULL clause/sentence as printed (e.g. 'Tenant shall pay the Tenancy Agreement Cost & Stamp Duty & Expenses'), not just a single word like 'Tenant' - a bare party name with no context is not a usable answer.", "type": "text"},
        {"field": "Special/Other Condition(s)", "description": "Any special conditions written on the form.", "type": "text"},
        {"field": "Notice to Quit (months)", "description": "The notice period (in months) required to terminate the tenancy.", "type": "number"},
    ]


def format_field_line(field):
    """One field as a prompt line: "- Name: description".

    The schema's declared "type" is deliberately NOT put in the prompt. It
    looks like free accuracy - three of the cloud model's misses were fields
    declared "number" that came back as prose ("RM 2,800 (Ringgit Malaysia Two
    Thousand Eight Hundred Only)") - so annotating the line as
    "Monthly Rental (number)" was tried and MEASURED on the local model:

        vision-only, file.pdf, no annotation :  26/28
        vision-only, file.pdf, "(number)"    :  19/28

    Every numeric field came back as an empty string instead. Whatever the
    mechanism, the model preferred returning nothing over returning a value it
    could not render as a bare number - trading a scoring miss for no data at
    all, which is worse. The prose-formatting problem is real, but the fix has
    to be something that does not let the model answer with silence.

    Left as a function rather than an inline f-string so the finding has
    somewhere to live and one place changes every prompt builder."""
    return f"- {field['field']}: {field['description']}"


def get_field_schema_df():
    """Same schema as get_field_schema(), as a DataFrame for easy viewing."""
    return pd.DataFrame(get_field_schema())


def get_group_schema():
    """Defines named groups of related fields on a Letter of Offer, each
    with a "multiple" flag: True for roles that can have more than one
    person/entity attached (two co-landlords, a company tenant plus its
    individual signatory, more than one witness) - one object per
    occurrence; False for a group that's just logically related fields
    bundled together (e.g. "property"), still exactly one object. Each
    group's "fields" list uses the same {field, description, type} shape as
    get_field_schema(). Grouping mirrors how LlamaExtract's own schema nests
    output (data.property, data.landlord, data.tenant, ...) rather than a
    flat list of every field."""
    return {
        "property": {
            "description": "The property being rented - one object, since a Letter of Offer only ever covers a single property.",
            "multiple": False,
            "fields": [
                # "candidates" drives mark_detector's template-free geometric
                # read of this field (see resolve_choice_fields_geometrically);
                # it is also what the prompt's option list is built from, so
                # the LLM and the detector can never disagree about what the
                # legal options are.
                {"field": "Description/Property Type", "description": "The property type, printed as a list of choices, e.g. 'House / Semi-D / Bungalow / Condo / Apartment / Shop / Shop Apt / Office / Factory / Land'. This template marks its selection ONE of two ways - either every option except the chosen one has a line struck through it (read the one option left NOT struck through), or the chosen option is circled while the rest are plain (read the circled one). Look carefully for which marking style this specific form uses before answering. Some templates instead state the property type as free text (e.g. 'Double Storey Cluster House') with no printed option list at all - in that case just read the text as written.", "type": "categorical",
                 "candidates": ["House", "Semi-D", "Bungalow", "Condo", "Apartment", "Shop", "Shop Apt", "Office", "Factory", "Land"]},
                {"field": "Address", "description": "The full address of the property being rented.", "type": "text"},
                {"field": "Title Details", "description": "Any land/title reference for the property, e.g. 'Mukim Tebrau'.", "type": "text"},
                {"field": "Usage", "description": "What the premise will be used for, e.g. 'F&B', 'RESIDENTIAL'.", "type": "text"},
                {"field": "Furnished Status", "description": "Whether the unit is Not / Partly / Fully furnished, printed as those 3 choices on one line. This template marks its selection ONE of two ways, either two of the three words are struck through and the one NOT struck through is the answer, or one word is circled and that's the answer. Determine which marking style is used on this form, then answer with just the single selected word (e.g. 'FULLY'). If you look carefully and NONE of the three words has any strikethrough or circle at all - the line is printed but genuinely unmarked - return an empty value rather than guessing one of the three words. Do not default to 'Not' or any other word without an actual visible mark.", "type": "categorical",
                 "candidates": ["NOT", "PARTLY", "FULLY"]},
            ],
        },
        "landlord": {
            "description": "Each landlord/lessor named on the form - there can genuinely be more than one co-landlord, each with their own name/NRIC, and each should be its own occurrence. Do NOT include the witness who signed 'Witnessed by' / 'Signed in the presence of:' next to the landlord's signature block - a witness is a separate role (see the 'witness' group) even if physically close to the landlord's block, or the same person who negotiated the deal.",
            "multiple": True,
            "fields": [
                {"field": "Name", "description": "The landlord's full name.", "type": "text"},
                {"field": "NRIC/Passport No.", "description": "For an individual landlord, their NRIC or passport number. For a COMPANY landlord, this is that company's own registration number - NOT a company stamp's reference number, and NOT a signatory's personal ID.", "type": "text"},
                {"field": "Designation (if any)", "description": "The landlord's designation/title if signing on behalf of a company (e.g. 'Director', 'Owner').", "type": "text"},
                {"field": "Signature Date", "description": "The date the landlord signed, if shown.", "type": "text"},
            ],
        },
        "tenant": {
            "description": "Each tenant named on the form - a company tenant AND its individual signatory both count as separate occurrences, as do multiple co-tenants, so there can genuinely be more than one. Do NOT include the witness who signed 'Witnessed by' / 'Signed in the presence of:' next to the tenant's signature block - a witness is a separate role (see the 'witness' group) even if physically close to the tenant's block, or the same person who negotiated the deal.",
            "multiple": True,
            "fields": [
                {"field": "Name/Company", "description": "The tenant's full name, or company name and registration number if a company.", "type": "text"},
                {"field": "NRIC/Passport No.", "description": "For an individual tenant, their NRIC or passport number. For a COMPANY tenant, this is that company's own registration number (e.g. '1626300-V', usually printed right after the company name in parentheses) - NOT a company stamp's reference number, and NOT a signatory's personal ID (that belongs on the signatory's own separate occurrence instead).", "type": "text"},
                {"field": "Address", "description": "The tenant's mailing/registered address, if printed on the form.", "type": "text"},
                {"field": "Designation (if any)", "description": "The tenant's designation/title if signing on behalf of a company (e.g. 'Director').", "type": "text"},
                {"field": "Signature Date", "description": "The date the tenant signed, if shown.", "type": "text"},
            ],
        },
        "agent": {
            "description": "The PropNex negotiator(s)/agent(s) handling this deal, printed after an explicit 'Attn:' or 'Negotiator(s):' label near the top of the form. Usually just one, but more than one is possible. If the form has NO such label at all, return an empty array [] - do NOT fill this with a landlord's, tenant's, or witness's name just because a name is present elsewhere on the page. A name only belongs here if it is genuinely printed next to an 'Attn:'/'Negotiator(s):' label.",
            "multiple": True,
            "fields": [
                {"field": "Name", "description": "The negotiator/agent's name.", "type": "text"},
            ],
        },
        "witness": {
            "description": "Each witness who signed 'Witnessed by' / 'Signed in the presence of:' on a signature block - the tenant's and landlord's witness may be the same person or different people, and there can be more than one.",
            "multiple": True,
            "fields": [
                {"field": "Name", "description": "The witness's name.", "type": "text"},
                {"field": "NRIC/Passport No.", "description": "The witness's NRIC or passport number, if shown.", "type": "text"},
                {"field": "Signature Date", "description": "The date the witness signed, if shown.", "type": "text"},
            ],
        },
    }


def get_group_schema_df():
    """Flattens get_group_schema() into one row per (group, sub-field) pair,
    for easy viewing - "multiple" repeats on every row of its group so the
    table stays flat instead of nesting a list inside a cell."""
    rows = []
    for role, spec in get_group_schema().items():
        for field in spec["fields"]:
            rows.append({
                "group": role,
                "multiple": spec["multiple"],
                "field": field["field"],
                "description": field["description"],
                "type": field["type"],
            })
    return pd.DataFrame(rows)


# The flattened name run() publishes <- (group, sub-field) it is read from.
# Single source of truth for that mapping, shared by flatten_fields() (which
# applies it) and field_sources() (which reports it). Written out once because
# two copies would drift, and a drifted name fails SILENTLY: the flattener, the
# comparison table and the ground-truth scorer all key on the exact display
# name, so a mismatch yields an empty row rather than an error.
FLAT_FROM_GROUPS = (
    ("Description of Property/Property Type", "property", "Description/Property Type"),
    ("Property Address", "property", "Address"),
    ("Title Details", "property", "Title Details"),
    ("Premise Usage", "property", "Usage"),
    ("Furnished Status", "property", "Furnished Status"),
    ("Landlord Name(s)", "landlord", "Name"),
    ("Landlord NRIC/Passport No.", "landlord", "NRIC/Passport No."),
    ("Tenant Name(s)/Company", "tenant", "Name/Company"),
    ("Tenant NRIC/Passport No.", "tenant", "NRIC/Passport No."),
    ("Tenant Address", "tenant", "Address"),
    ("Negotiator/Agent Name", "agent", "Name"),
    ("Witnessed by", "witness", "Name"),
)


def field_sources():
    """{flattened field name: where its value comes from}, in output order.

    ("field", name) for a top-level field, ("group", group, sub-field) for one
    the flattener lifts out of a group. This is the engine's answer to "what
    can be extracted" - the UI's picker is built from it rather than from a
    list of its own, so a field added to the schema becomes pickable with no
    second edit, and one removed stops being offered.

    Note the group sub-fields NOT listed here (a party's designation, a
    signature date) are deliberately absent: they never reach the flat output,
    so there is nothing for a picker to show or a table to hide.
    """
    sources = {f["field"]: ("field", f["field"]) for f in get_field_schema()}
    for flat_name, group, sub_field in FLAT_FROM_GROUPS:
        sources[flat_name] = ("group", group, sub_field)
    return sources


def selectable_fields():
    """Just the names from field_sources(), for --list-fields."""
    return list(field_sources())


def _excluded_names(exclude):
    """The exclusion argument as a set, tolerating None."""
    return {str(name) for name in (exclude or ())}


def field_schema_for(exclude=()):
    """get_field_schema() minus the excluded fields.

    Excluding is subtractive on purpose: the picker can only ever narrow this
    list, never extend it. A name the schema doesn't have is ignored rather
    than rejected - a stale name in a caller's list must not fail a run.
    """
    dropped = _excluded_names(exclude)
    return [f for f in get_field_schema() if f["field"] not in dropped]


def group_schema_for(exclude=()):
    """get_group_schema() minus the excluded fields, by their FLATTENED names.

    A group's sub-field is dropped when the flat name it feeds is excluded,
    and the whole group goes when every flat name it feeds has gone - asking
    for a landlord object whose name and NRIC are both switched off would burn
    prompt tokens on an answer nothing downstream reads. Sub-fields that feed
    no flat name (designation, signature date) ride along with their group,
    since no picker offers them and no table shows them.

    Known artifact, accepted: the landlord and tenant descriptions tell the
    model not to file a witness under them, "see the 'witness' group" - and
    that pointer dangles if the witness group has itself been dropped. The
    instruction that matters (do NOT include them here) still stands, so the
    field this protects is still protected; only the suggested destination is
    missing. Not worth rewriting descriptions per-selection unless a real run
    is shown to mis-file a witness because of it.
    """
    dropped = _excluded_names(exclude)
    if not dropped:
        return get_group_schema()

    sub_field_of = {}   # (group, sub-field) -> flat name
    exposed = {}        # group -> [flat name, ...]
    for flat_name, group, sub_field in FLAT_FROM_GROUPS:
        sub_field_of[(group, sub_field)] = flat_name
        exposed.setdefault(group, []).append(flat_name)

    kept = {}
    for group, spec in get_group_schema().items():
        if exposed.get(group) and all(name in dropped for name in exposed[group]):
            continue
        fields = [f for f in spec["fields"]
                  if sub_field_of.get((group, f["field"])) not in dropped]
        if not fields:
            continue
        kept[group] = {**spec, "fields": fields}
    return kept


def build_chat_payload(prompt, model, images, json_mode, schema=None):
    """The /api/chat body for one call.

    num_ctx is raised for EVERY call, not just vision ones. It used to be set
    only when images were attached, so a text-mode call - which carries the
    document's whole recognized text in its prompt - ran at Ollama's default
    context, overflowed, and had its reply truncated mid-JSON. That failed
    deterministically at the same character on every retry, which is what
    makes it worth pinning: it does not look like a context problem from the
    outside, it looks like the model emitting bad JSON."""
    message = {"role": "user", "content": prompt}
    if images:
        message["images"] = images
    payload = {
        "model": model,
        "messages": [message],
        "stream": False,
        "options": {"num_ctx": 16384},
    }
    if schema is not None:
        # A JSON Schema here makes the reply's shape a constraint on
        # generation rather than a request in the prompt - the model cannot
        # emit an unknown key or a wrong-typed value. Supersedes plain
        # "json", which only asks for valid JSON of any shape.
        payload["format"] = schema
    elif json_mode:
        payload["format"] = "json"
    return payload


def call_llm(prompt, model=LLM_MODEL, host=OLLAMA_URL, json_mode=True, timeout=900, images=None, return_meta=False, schema=None):
    """Sends a prompt to the given Ollama model via the local daemon -
    which proxies a '-cloud' model name to Ollama's cloud - and returns
    the raw text response. Pass images (a list of base64 PNGs, e.g. from
    render_pdf_to_base64_images) for a vision-mode call; omit it for the
    cheaper/faster text-only mode. Pass return_meta=True to get back
    (content, {"prompt_eval_count": ..., "eval_count": ..., "time_s": ...})
    instead of just content - for logging token/time cost per call.

    Retries with exponential backoff on a genuine transient 429, but fails
    fast (no retry) on a "session usage limit" 429 - confirmed this is a
    free-tier quota, not a short-lived burst limit (no Retry-After header,
    and it didn't clear after ~15s of backoff), so retrying just wastes
    time until the quota window resets on its own."""
    payload = build_chat_payload(prompt, model, images, json_mode, schema=schema)

    max_retries = 5
    for attempt in range(max_retries):
        start = time.monotonic()
        response = requests.post(host, json=payload, timeout=timeout)
        elapsed = time.monotonic() - start
        if response.status_code == 402:
            # The cloud allowance is spent and the model needs a paid plan.
            # Never retryable, so fail fast like the quota 429 below - and
            # keep Ollama's own wording, which names the fix. models.py's
            # looks_rate_limited() matches on it to take the model out of the
            # selector rather than let it be picked again.
            raise RuntimeError(
                f"Ollama cloud subscription required: {response.text.strip()}"
            )
        if response.status_code == 429:
            if "session usage limit" in response.text:
                raise RuntimeError(
                    f"Ollama cloud session usage limit reached: {response.text.strip()}"
                )
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  429 Too Many Requests - retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        if not return_meta:
            return content
        return content, {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "time_s": round(elapsed, 1),
        }


def ocr_to_dataframe(pdf_path, dpi=150):
    """Returns the page's text layer as one row per line: a stable numeric
    "index" (for the LLM to reference back to - see build_ocr_prompt), which
    page it's on, the text, confidence, and its bounding box (x0, y0, x1,
    y1) in pixel coordinates of that page's rendered image.

    Reads a born-digital PDF's own text layer instead of running OCR over a
    rendering of it - same output shape, but exact rather than recognized,
    and it takes milliseconds instead of ~40-60s per document. Falls back to
    PaddleOCR for scans, which have no text layer to read."""
    if _has_text_layer(pdf_path):
        return _native_text_dataframe(pdf_path, dpi=dpi)

    rows = []
    for page_num, img in enumerate(pdf_to_images(pdf_path, dpi)):
        result = run_ocr(img)
        for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"]):
            x0, y0, x1, y1 = box
            rows.append({
                "index": len(rows),
                "page": page_num,
                "text": text,
                "confidence": round(float(score), 2),
                "x0": int(x0), "y0": int(y0), "x1": int(x1), "y1": int(y1),
            })
    return pd.DataFrame(rows)


def _has_text_layer(pdf_path, min_chars=100):
    """True when page 1 carries real extractable text - i.e. a born-digital
    PDF rather than a scan. Same threshold the choice-field detector uses to
    pick its exact vector path over the pixel one."""
    with fitz.open(pdf_path) as pdf:
        return len(pdf[0].get_text().strip()) > min_chars


def _native_text_dataframe(pdf_path, dpi=150):
    """Builds the same frame ocr_to_dataframe returns, from a digital PDF's
    own text lines. Boxes are converted from PDF points to pixels at `dpi`
    so downstream consumers (get_ocr_boxes_normalized, the choice-field
    detector) need no special case. Confidence is 1.0 - this is the
    document's own text, not a recognition guess."""
    scale = dpi / 72.0
    rows = []
    with fitz.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf):
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(span["text"] for span in line["spans"]).strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = line["bbox"]
                    rows.append({
                        "index": len(rows),
                        "page": page_num,
                        "text": text,
                        "confidence": 1.0,
                        "x0": int(x0 * scale), "y0": int(y0 * scale),
                        "x1": int(x1 * scale), "y1": int(y1 * scale),
                    })
    return pd.DataFrame(rows)


def _page_pixel_size(pdf_path, page_num, dpi):
    """Returns (width_px, height_px) a page WOULD render to at the given
    dpi, computed from its point-size - no actual rendering needed, since
    PDF points are always 1/72 inch."""
    with fitz.open(pdf_path) as pdf:
        rect = pdf[page_num].rect
    return (rect.width * dpi / 72.0, rect.height * dpi / 72.0)


def get_ocr_boxes_normalized(pdf_path, ocr_df=None, dpi=150):
    """Returns one record per OCR-recognized line, with its bounding box
    normalized to fractions of that page's width/height (0.0-1.0) rather
    than raw pixels - so a box can be drawn correctly on a display image
    rendered at ANY resolution, not just the DPI OCR itself used. Includes
    the recognized text too, so a caller can locate a field's box by
    matching the extracted value against OCR text directly, without
    needing the LLM's source_index (e.g. for a value read off a visual
    mark, which the LLM cannot cite an OCR line for).
    [{"index": int, "page": int, "text": str, "x0": float, "y0": float,
    "x1": float, "y1": float}, ...]."""
    if ocr_df is None:
        ocr_df = ocr_to_dataframe(pdf_path, dpi=dpi)
    page_sizes = {}
    records = []
    for row in ocr_df.itertuples():
        if row.page not in page_sizes:
            page_sizes[row.page] = _page_pixel_size(pdf_path, row.page, dpi)
        pw, ph = page_sizes[row.page]
        records.append({
            "index": row.index, "page": row.page, "text": row.text,
            "x0": row.x0 / pw, "y0": row.y0 / ph,
            "x1": row.x1 / pw, "y1": row.y1 / ph,
        })
    return records


def strip_json_fence(text):
    """Some models wrap JSON in a ```json fence even when "format": "json"
    is set - strip that before parsing."""
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE).strip()


def parse_llm_json(text):
    """Parses an LLM's JSON response robustly: strips a ```json fence if
    present, then decodes only the FIRST complete JSON value and ignores
    anything after it. Some models (seen intermittently with minimax-m3)
    occasionally emit trailing text after the JSON even when told to
    return JSON only - a plain json.loads would reject that outright with
    "Extra data", so use raw_decode() instead, which stops at the end of
    the first valid value rather than requiring the whole string to be
    exactly one JSON document."""
    cleaned = strip_json_fence(text)
    obj, _ = json.JSONDecoder().raw_decode(cleaned)
    return obj


def verify_source_index(ocr_df, value, source_index):
    """Sanity-checks an extracted value against the OCR line it claims to
    come from (source_index) - the "semantic text highlighting" pattern
    from https://medium.com/@jameschen_78678/enhancing-image-text-extraction-with-llm-and-ocr-8221cb555cc5,
    applied to catch misattribution automatically instead of only by
    manually eyeballing results against ground truth (which is how the
    Reference No./Special Condition(s)/tenant-vs-witness bugs were all
    found this session).

    Returns (plausible, ocr_line_text):
      plausible is None   - source_index missing/null, nothing to check
      plausible is False  - source_index doesn't exist, or its OCR text
                             has no meaningful overlap with the value
      plausible is True   - some meaningful overlap found

    This is a heuristic, not a perfect validator - values are often
    reformatted from the raw OCR text (an added 'RM' prefix, punctuation),
    so it checks for meaningful overlap rather than an exact match."""
    if source_index is None or (isinstance(value, str) and not value.strip()):
        return None, None

    matches = ocr_df[ocr_df["index"] == source_index]
    if matches.empty:
        return False, None
    ocr_text = matches.iloc[0]["text"]

    def normalize(s):
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    value_norm, ocr_norm = normalize(value), normalize(ocr_text)
    if not value_norm:
        return None, ocr_text

    plausible = (
        value_norm in ocr_norm or ocr_norm in value_norm or
        (len(value_norm) >= 4 and any(value_norm[i:i + 4] in ocr_norm for i in range(len(value_norm) - 3)))
    )
    return plausible, ocr_text


def format_ocr_text(ocr_df):
    """Turns the OCR dataframe into a plain-text block, grouped by page,
    for embedding in a text-mode LLM prompt. Each line is tagged with its
    stable "index" (no bounding boxes here - just enough for the LLM to
    reference which line an answer came from, matching the
    "semantic text highlighting" pattern from
    https://medium.com/@jameschen_78678/enhancing-image-text-extraction-with-llm-and-ocr-8221cb555cc5
    - see build_ocr_prompt for how that reference is requested and
    verify_source_index for how it's checked afterward)."""
    blocks = []
    for page in sorted(ocr_df["page"].unique()):
        page_df = ocr_df[ocr_df["page"] == page]
        lines = "\n".join(f'  [{row.index}] "{row.text}" (confidence: {row.confidence:.2f})' for row in page_df.itertuples())
        blocks.append(f"OCR text detected on page {page}:\n{lines}")
    return "\n\n".join(blocks)


def build_ocr_prompt(ocr_df, fields):
    """Builds a text-mode extraction prompt: the OCR text as context, then
    the fields to extract. For each field, the LLM is asked to also
    self-report a confidence - low/0.0 when a field genuinely can't be
    determined from text alone (e.g. a circled/struck-through choice, which
    OCR text carries no signal for at all), so those fields can be flagged
    for a vision-mode fallback later instead of trusting a guess."""
    field_lines = "\n".join(format_field_line(f) for f in fields)
    return (
        "Below is OCR-extracted text from a scanned form, one line per "
        "detected text region with the OCR engine's own confidence score:\n\n"
        f"{format_ocr_text(ocr_df)}\n\n"
        "Extract these fields strictly from the OCR text above. For each "
        "field return an object with \"value\" (the extracted text, or \"\" "
        "if not found), \"confidence\" (0.0-1.0, your own confidence that "
        "the value is correct and complete, based on how directly it "
        "appears in the OCR text - use a low confidence rather than "
        "guessing if a field isn't clearly present), and \"source_index\" "
        "(the single [N] index of the OCR line the value came from, or "
        "null if not applicable):\n\n"
        f"{field_lines}\n\n"
        "Return ONLY a JSON object mapping each field name to its "
        "{\"value\": ..., \"confidence\": ..., \"source_index\": ...} object."
    )


def extract_fields_from_ocr(pdf_path, model=LLM_MODEL, ocr_df=None):
    """Step 2 of the hybrid pipeline: OCR first (cheap), then a single
    text-mode LLM call (no image - cheaper than vision mode) to extract
    every flat field in get_field_schema() from the OCR text alone. Returns
    {field: {"value": ..., "confidence": ...}}. Pass an already-computed
    ocr_df (e.g. from ocr_to_dataframe) to avoid re-running OCR when calling
    this alongside extract_groups_from_ocr on the same PDF."""
    if ocr_df is None:
        ocr_df = ocr_to_dataframe(pdf_path)
    prompt = build_ocr_prompt(ocr_df, get_field_schema())
    raw = call_llm(prompt, model=model)
    return parse_llm_json(raw)


def build_group_prompt(ocr_df, group_name, spec):
    """Same idea as build_ocr_prompt, but for one group from
    get_group_schema(): a "multiple" group (landlord/tenant/agent/witness)
    asks for a JSON ARRAY (one object per occurrence, since there can be
    more than one), a non-multiple group (property) asks for a single JSON
    object - same {value, confidence, source_index} shape either way."""
    field_lines = "\n".join(format_field_line(f) for f in spec["fields"])
    value_shape = "{\"value\": ..., \"confidence\": ..., \"source_index\": ...}"
    if spec["multiple"]:
        instruction = (
            f"There can be MORE THAN ONE {group_name} on this form - "
            f"{spec['description']}\n\n"
            f"For EACH {group_name} found, extract these sub-fields, each "
            f"with a \"source_index\" - the single [N] index of the OCR "
            "line the value came from, or null if not applicable:\n"
            f"{field_lines}\n\n"
            "Return ONLY a JSON ARRAY, one object per occurrence, each "
            f"mapping a sub-field name to {value_shape}. Return an empty "
            "array [] if none are found."
        )
    else:
        instruction = (
            f"Extract these sub-fields for the {group_name} on this form "
            f"({spec['description']}), each with a \"source_index\" - the "
            "single [N] index of the OCR line the value came from, or null "
            f"if not applicable:\n{field_lines}\n\n"
            "Return ONLY a JSON object mapping each sub-field name to "
            f"{value_shape}."
        )
    return (
        "Below is OCR-extracted text from a scanned form, one line per "
        "detected text region with the OCR engine's own confidence score:\n\n"
        f"{format_ocr_text(ocr_df)}\n\n"
        f"{instruction}"
    )


def extract_groups_from_ocr(pdf_path, model=LLM_MODEL, ocr_df=None):
    """Extracts every group in get_group_schema() - property, landlord,
    tenant, agent, witness - from OCR text. One text-mode LLM call PER
    GROUP rather than one big batched call: "multiple" groups need a
    different JSON shape (array) than "property" (object), and landlord's
    and tenant's sub-fields use near-identical names (Name, NRIC/Passport
    No., Signature Date) - batching them together was confirmed (twice) to
    cause cross-field mix-ups and lower accuracy overall, on top of being
    ~3x slower in wall-clock time despite being fewer round-trips. Returns
    {group_name: object or list, matching get_group_schema()'s "multiple"
    flag}."""
    if ocr_df is None:
        ocr_df = ocr_to_dataframe(pdf_path)
    results = {}
    for group_name, spec in get_group_schema().items():
        prompt = build_group_prompt(ocr_df, group_name, spec)
        raw = call_llm(prompt, model=model)
        parsed = parse_llm_json(raw)
        results[group_name] = _unwrap_group_array(parsed, spec)
    return results


def _unwrap_group_array(parsed, spec):
    """Weaker models sometimes wrap the requested bare JSON array inside an
    object instead (e.g. {"landlords": [...]} instead of just [...]),
    despite being told to return an array - unwrap it by finding the first
    list-valued key, rather than crashing downstream when code expecting a
    list gets a dict instead."""
    if spec["multiple"] and isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, list):
                return v
        return []
    return parsed


def _safe_leaf(v):
    """Normalizes a sub-field's parsed value to always be a dict with
    value/confidence/source_index keys, tolerating models that occasionally
    return a bare null (or other non-dict) instead of the requested
    {"value": ..., "confidence": ..., "source_index": ...} object."""
    return v if isinstance(v, dict) else {"value": v, "confidence": None, "source_index": None}


# Separator for a repeatable role flattened into one field. Deliberately the
# same one vision-form-reader/api_ground_truth.py uses to build its own
# "Landlord Name" ground truth (" / ".join(...) over every party), so the two
# sides of the comparison table are formatted identically.
OCCURRENCE_SEPARATOR = " / "


def join_occurrences(occurrences, field):
    """Collapses one sub-field across every occurrence of a repeatable role
    into a single string, e.g. two co-landlords -> "ALICE TAN / BRIAN
    LIM".

    Replaces an earlier "keep occurrences[0], drop the rest" flattening,
    which silently lost the second of two co-landlords on real forms - and
    was invisible in the comparison table, because groundtruth.values_match
    accepts either side as a substring of the other, so the truncated answer
    still scored as a match.

    Blanks are skipped and repeats are dropped case-insensitively while
    keeping document order: one witness who signs both the tenant's and the
    landlord's block is two genuine occurrences of the same person and
    should read once, not twice. Returns "" for an absent role, matching
    run()'s convention of "" rather than None for a genuinely empty field."""
    seen, values = set(), []
    for occurrence in occurrences or []:
        if not isinstance(occurrence, dict):
            continue
        value = _safe_leaf(occurrence.get(field)).get("value")
        if value is None:
            continue
        text = str(value).strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        values.append(text)
    return OCCURRENCE_SEPARATOR.join(values)


def build_vision_grounded_prompt(fields, group_schema, ocr_df, num_pages):
    """Vision-primary design: the page image is the primary source of
    truth (read directly for handwriting, layout, circled/struck marks -
    none of which OCR text can capture), with OCR-recognized text supplied
    as grounding context only, to help verify/locate PRINTED values. This
    is the opposite emphasis from build_ocr_prompt/build_group_prompt
    (OCR-text-only, no image), which this session's own testing showed
    underperforms on this kind of layout/mark-sensitive form - matching
    what's reported in the IDP literature (e.g. LandingAI's Agentic
    Document Extraction, Azure's GPT text+vision samples): OCR-as-citation
    alongside the image, not OCR-as-replacement-input."""
    field_lines = "\n".join(format_field_line(f) for f in fields)

    group_sections = []
    for group_name, spec in group_schema.items():
        sub_lines = "\n".join("    " + format_field_line(f) for f in spec["fields"])
        shape_note = (
            "return a JSON ARRAY of objects, one per occurrence, [] if none found"
            if spec["multiple"] else "return a single JSON object"
        )
        group_sections.append(f'"{group_name}" ({spec["description"]} - {shape_note}):\n{sub_lines}')
    groups_block = "\n\n".join(group_sections)

    page_note = (
        f"You are given all {num_pages} pages of the same form, in order - "
        "a field's value may be on any page, so check all of them.\n\n"
        if num_pages > 1 else ""
    )
    value_shape = "{\"value\": ..., \"confidence\": ..., \"source_index\": ...}"

    return (
        "You are reading a scanned or digitally-generated PropNex \"Letter "
        "of Offer to Rent\" form. Look directly at the page image(s) for "
        "every value, including handwritten entries, circled options, and "
        "struck-through options (a struck-through option is NOT selected; "
        "an un-struck or circled option IS selected) - the image is your "
        "primary source of truth.\n\n"
        f"{page_note}"
        "Below is OCR-recognized text from the same page(s) - a plain "
        "text-recognition pass with no understanding of layout or visual "
        "marks. Use it only to help verify/locate PRINTED values (it may "
        "contain misreadings); never let it override what you actually "
        "see in the image, and it has NO signal at all for circled/struck "
        "marks or handwriting:\n\n"
        f"{format_ocr_text(ocr_df)}\n\n"
        "Extract these top-level fields, each as an object with \"value\", "
        "\"confidence\" (0.0-1.0), and \"source_index\" (the [N] OCR index "
        "backing a printed value, or null if it's read from a visual mark, "
        "handwriting, or something the OCR text doesn't cover):\n"
        f"{field_lines}\n\n"
        "Also extract these groups of fields, using the same "
        f"{value_shape} shape for every sub-field:\n\n"
        f"{groups_block}\n\n"
        "Return ONLY a single JSON object with exactly two top-level keys: "
        "\"fields\" (an object mapping each top-level field name above to "
        f"its {value_shape}) and \"groups\" (an object with keys " +
        ", ".join(f'"{g}"' for g in group_schema) + ", each value an "
        "array of objects for a \"multiple\" group or a single object "
        "otherwise, as specified above)."
    )


def build_vision_only_prompt(fields, group_schema, num_pages):
    """The ungrounded prompt: identical schema and output contract to
    build_vision_grounded_prompt, with two things removed - the OCR text
    block, and the "source_index" citation, which means nothing when there is
    no OCR index to cite.

    Kept deliberately identical otherwise, because that is what makes the two
    a fair comparison: the A/B benchmark measured this arm at 92.0% on the
    cloud model against 94.6% grounded, in a third of the wall clock. Held
    constant so the only variable is the grounding itself."""
    field_lines = "\n".join(format_field_line(f) for f in fields)

    group_sections = []
    for group_name, spec in group_schema.items():
        sub_lines = "\n".join("    " + format_field_line(f) for f in spec["fields"])
        shape_note = (
            "return a JSON ARRAY of objects, one per occurrence, [] if none found"
            if spec["multiple"] else "return a single JSON object"
        )
        group_sections.append(f'"{group_name}" ({spec["description"]} - {shape_note}):\n{sub_lines}')
    groups_block = "\n\n".join(group_sections)

    page_note = (
        f"You are given all {num_pages} pages of the same form, in order - "
        "a field's value may be on any page, so check all of them.\n\n"
        if num_pages > 1 else ""
    )
    value_shape = "{\"value\": ..., \"confidence\": ...}"

    return (
        "You are reading a scanned or digitally-generated PropNex \"Letter "
        "of Offer to Rent\" form. Look directly at the page image(s) for "
        "every value, including handwritten entries, circled options, and "
        "struck-through options (a struck-through option is NOT selected; "
        "an un-struck or circled option IS selected) - the image is your "
        "only source of truth.\n\n"
        f"{page_note}"
        "Extract these top-level fields, each as an object with \"value\" "
        "and \"confidence\" (0.0-1.0):\n"
        f"{field_lines}\n\n"
        "Also extract these groups of fields, using the same "
        f"{value_shape} shape for every sub-field:\n\n"
        f"{groups_block}\n\n"
        "Return ONLY a single JSON object with exactly two top-level keys: "
        "\"fields\" (an object mapping each top-level field name above to "
        f"its {value_shape}) and \"groups\" (an object with keys " +
        ", ".join(f'"{g}"' for g in group_schema) + ", each value an "
        "array of objects for a \"multiple\" group or a single object "
        "otherwise, as specified above)."
    )


def ungrounded_box_lines(pdf_path, dpi=150):
    """Box-anchor lines for a run with grounding off, or [] when getting them
    would cost what the toggle was turned off to avoid.

    The toggle's purpose is to keep the document's text OUT OF THE PROMPT -
    that is where the thousands of tokens go. It was never meant to throw away
    the boxes too, and on a born-digital PDF it does not have to: the text
    layer is read in milliseconds and carries the document's own exact
    coordinates, so the source boxes survive an ungrounded run for free.

    A scan is different - its lines only exist after a full PaddleOCR pass,
    ~100s, which is precisely the cost being skipped. There the boxes go, and
    the UI says so."""
    if not grounding_is_cheap(_has_text_layer(pdf_path)):
        return []
    return get_ocr_boxes_normalized(
        pdf_path, ocr_df=_native_text_dataframe(pdf_path, dpi=dpi), dpi=dpi)


def grounding_is_cheap(has_text_layer):
    """Is OCR grounding effectively free for this document?

    A born-digital PDF is read through its own text layer in milliseconds
    (see ocr_to_dataframe), so grounding costs nothing and there is nothing
    for the toggle to save. A scan needs a full PaddleOCR pass - 100s+ per
    document, and roughly two thirds of a grounded run's wall clock. Turning
    grounding off only buys time in the second case."""
    return bool(has_text_layer)


def extract_vision_only(pdf_path, model=LLM_MODEL, return_meta=False, max_parse_retries=2, exclude=()):
    """Ungrounded extraction: image(s) + schema, one call, no OCR anywhere in
    the path. Mirrors extract_vision_grounded's retry-on-unparseable-JSON so a
    call-level fluke isn't scored as an accuracy difference between them.

    `exclude` names flattened fields the caller doesn't want; they are gone
    from the prompt entirely rather than filtered out of the answer, which is
    the whole point - an unasked field costs no tokens and no time."""
    group_schema = group_schema_for(exclude)
    images_b64 = render_pdf_to_base64_images(pdf_path)
    prompt = build_vision_only_prompt(field_schema_for(exclude), group_schema, len(images_b64))

    last_error = None
    for attempt in range(max_parse_retries + 1):
        result = call_llm(prompt, model=model, images=images_b64, return_meta=return_meta)
        raw, meta = result if return_meta else (result, None)
        try:
            parsed = parse_llm_json(raw)
            break
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Could not parse LLM response as JSON (attempt {attempt + 1}/"
                  f"{max_parse_retries + 1}): {e}. Retrying...", file=sys.stderr)
    else:
        raise last_error

    fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
    groups_raw = parsed.get("groups", {}) if isinstance(parsed, dict) else {}
    groups = {
        name: _unwrap_group_array(groups_raw.get(name), spec)
        for name, spec in group_schema.items()
    }
    extracted = {"fields": fields, "groups": groups}
    return (extracted, meta) if return_meta else extracted


SCHEMA_MODES = ("prompt", "pydantic")


def _format_schema(schema_mode, exclude=()):
    """The JSON Schema to constrain generation with, or None for the original
    prompt-only behaviour.

    "pydantic" imports schema_models lazily: that module imports this one, so
    a module-level import would be circular, and the dependency only exists
    when the mode is actually used."""
    if schema_mode not in SCHEMA_MODES:
        raise ValueError(f"unknown schema_mode {schema_mode!r}; expected one of {SCHEMA_MODES}")
    if schema_mode != "pydantic":
        return None
    import schema_models
    # The same filtered schema the prompt was built from. Constraining
    # generation to the FULL one would put every excluded field back as a
    # key the model is obliged to fill.
    return schema_models.extraction_json_schema(
        fields=field_schema_for(exclude), groups=group_schema_for(exclude))


def extract_vision_grounded(pdf_path, model=LLM_MODEL, ocr_df=None, return_meta=False, max_parse_retries=2, schema_mode="prompt", exclude=()):
    """Vision-primary extraction: sends the page image(s) plus OCR text
    (as grounding context, not a replacement) in a single call, covering
    both get_field_schema() and get_group_schema() at once. Returns
    {"fields": {...}, "groups": {...}}, or (that dict, meta) if
    return_meta=True - see call_llm's return_meta for meta's shape.

    Retries the whole call up to max_parse_retries times if the response
    can't be parsed as JSON - seen twice this session as two distinct
    failure modes (trailing "extra data" after valid JSON, and once a
    genuinely empty response body) - both are call-level flukes, not
    something a differently-worded prompt would fix, so a fresh retry is
    the right response rather than surfacing a crash.

    `exclude` names flattened fields to leave out; they never enter the
    prompt or the constrained schema."""
    if ocr_df is None:
        ocr_df = ocr_to_dataframe(pdf_path)
    group_schema = group_schema_for(exclude)
    images_b64 = render_pdf_to_base64_images(pdf_path)
    prompt = build_vision_grounded_prompt(field_schema_for(exclude), group_schema, ocr_df, len(images_b64))
    schema = _format_schema(schema_mode, exclude)

    last_error = None
    for attempt in range(max_parse_retries + 1):
        result = call_llm(prompt, model=model, images=images_b64, return_meta=return_meta,
                          schema=schema)
        raw, meta = result if return_meta else (result, None)
        try:
            parsed = parse_llm_json(raw)
            break
        except json.JSONDecodeError as e:
            last_error = e
            print(f"  Could not parse LLM response as JSON (attempt {attempt + 1}/"
                  f"{max_parse_retries + 1}): {e}. Retrying...", file=sys.stderr)
    else:
        raise last_error

    fields = parsed.get("fields", {}) if isinstance(parsed, dict) else {}
    groups_raw = parsed.get("groups", {}) if isinstance(parsed, dict) else {}
    groups = {
        name: _unwrap_group_array(groups_raw.get(name), spec)
        for name, spec in group_schema.items()
    }
    extracted = {"fields": fields, "groups": groups}
    return (extracted, meta) if return_meta else extracted


CHOICE_FIELD_TARGETS = {
    # group sub-field name -> the flat field name run() writes it to
    "Description/Property Type": "Description of Property/Property Type",
    "Furnished Status": "Furnished Status",
}


def choice_candidates(exclude=()):
    """{group sub-field name: its option list} for every choice field still
    being extracted.

    Published in run()'s output so the UI can locate a choice field's printed
    option line on the page the same way mark_detector does - by finding the
    line where the most distinct options co-occur. The UI needs the list but
    must not restate it: get_group_schema()'s "candidates" already drives both
    the prompt's option list and the geometric detector, and a second copy
    could drift out of agreement with them.
    """
    by_name = {
        field["field"]: field
        for spec in get_group_schema().values() for field in spec["fields"]
    }
    dropped = _excluded_names(exclude)
    return {
        sub_field: by_name[sub_field]["candidates"]
        for sub_field, target in CHOICE_FIELD_TARGETS.items()
        if target not in dropped and (by_name.get(sub_field) or {}).get("candidates")
    }


def resolve_choice_fields_geometrically(pdf_path, ocr_df=None, exclude=()):
    """Resolves the "pick one of N" choice fields by reading the page's
    geometry instead of trusting the vision-LLM, which is unreliable on
    these two regardless of prompting - refinement #1 in RECOMMENDATION.md
    found 4 of 5 answers still wrong at temperature=0 with a fixed seed, so
    the cause is visual ambiguity, not sampling noise.

    Template-free, unlike the previous per-template implementation: for a
    born-digital PDF the strikethrough is real vector geometry, and for a
    scan the choice line is located from the schema's own candidate list and
    then read from pixels. Nothing here knows or cares which template the
    document is.

    Returns {flat_field_name: value_or_None}. None means "no override, keep
    the LLM's answer" - which covers a form that states the field as free
    text with no printed option list, a genuinely unmarked line, and any
    ambiguous read. Abstaining rather than forcing an answer is deliberate:
    a wrong deterministic value is worse than deferring to the LLM.
    """
    dropped = _excluded_names(exclude)
    overrides = {target: None for target in CHOICE_FIELD_TARGETS.values()
                 if target not in dropped}
    by_name = {
        field["field"]: field
        for spec in get_group_schema().values() for field in spec["fields"]
    }

    with fitz.open(pdf_path) as pdf:
        native = len(pdf[0].get_text().strip()) > 100

    for sub_field, target in CHOICE_FIELD_TARGETS.items():
        candidates = (by_name.get(sub_field) or {}).get("candidates")
        # An excluded field skips the detector entirely: this is a page-scale
        # pixel/vector read, so not running it is real time saved, not just a
        # value discarded.
        if not candidates or target in dropped:
            continue
        try:
            if native:
                overrides[target] = _resolve_choice_native(pdf_path, candidates)
            else:
                if ocr_df is None:
                    ocr_df = ocr_to_dataframe(pdf_path)
                overrides[target] = _resolve_choice_raster(pdf_path, ocr_df, candidates)
        except Exception as exc:
            # A detector fault must never take the whole extraction down -
            # fall through to the LLM's own answer for this field.
            print(f"  choice-field detection failed for {target}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
    return overrides


def _resolve_choice_native(pdf_path, candidates):
    """Born-digital path: the strike is a real vector drawing, so this is an
    exact read rather than a pixel heuristic - and needs no OCR at all."""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            boxes = mark_detector.locate_native(page, candidates)
            if not boxes:
                continue
            answer, _ = mark_detector.resolve(
                struck=mark_detector.detect_struck_native(page, boxes))
            if answer:
                return answer
    return None


def _resolve_choice_raster(pdf_path, ocr_df, candidates, dpi=150):
    """Scan path: find the choice line via the candidate list, upscale it
    (a strike is only a few pixels tall at page scale), then test each
    option for a strike and look for a ring to read the inside of."""
    row, _ = mark_detector.locate_ocr_line(ocr_df, candidates)
    if row is None:
        return None

    img = pdf_page_to_image(pdf_path, int(row.page), dpi=dpi)
    crop, _, band, _ = mark_detector.crop_line(
        img, (row.x0, row.y0, row.x1, row.y1))
    if crop is None:
        return None

    ink = mark_detector._binary_ink(crop)
    engine = get_ocr_engine()
    word_boxes, _ = mark_detector.word_boxes_by_segment_ocr(
        crop, ink, band, candidates, engine)
    struck, _ = mark_detector.detect_struck_raster(crop, word_boxes)
    ring_pick, _ = mark_detector.find_ring_and_read(
        crop, ink, band, candidates, engine)

    circled = {c: (c == ring_pick) for c in word_boxes} if ring_pick else {}
    answer, _ = mark_detector.resolve(struck=struck, circled=circled)
    return answer or ring_pick


def run(pdf_path, model=LLM_MODEL, grounded=True, schema_mode="prompt", exclude=()):
    """Runs the vision-primary + OCR-grounding extraction
    (extract_vision_grounded) and flattens the nested {fields, groups}
    result into a single flat "fields" dict, using the SAME field names
    vllm-schema-extractor's FIELDS schema uses (get_field_schema()/
    get_group_schema() were deliberately named to match), so
    streamlit_app's existing VLLM_ALIASES can compare this approach too
    without a new alias dict. EVERY occurrence of a repeatable role
    (landlord/tenant/agent/witness) is flattened into its field, joined
    with " / " - see join_occurrences, and note that's the same separator
    api_ground_truth.py builds its own multi-party ground truth with. The
    full per-occurrence detail (designation, signature date, and which
    value came from which party) stays in raw["groups"] via
    extract_vision_grounded() directly, and is kept here too under "_raw".

    `exclude` is a list of flattened field names to leave out, as published
    by field_sources() - the UI's field picker passes whatever the user
    unticked. Excluded means excluded everywhere: out of the prompt, out of
    the constrained schema, out of the geometric choice-field pass, and out
    of the flat result. It can only ever narrow the schema; there is no way
    to add a field this way.

    grounded=False drops the text pass entirely and extracts from the image
    alone (extract_vision_only). Note "the text pass" is not always OCR:
    ocr_to_dataframe reads a born-digital PDF's own text layer and only falls
    back to PaddleOCR for a scan. It removes two costs, not one: obtaining
    that text, AND the several thousand tokens of recognized text that would
    otherwise be pasted into the prompt. The second matters even when the
    first is free - a born-digital PDF's text layer is read in milliseconds,
    yet dropping it still took the local model from 440s to 107s on file.pdf
    (8,070 -> 4,761 prompt tokens) for one field of accuracy, because that
    model is bound by how much prompt it reads. The cloud model is bound by
    how much it writes instead, so there the same switch bought 92.0% vs
    94.6% and was sometimes SLOWER. Note what else goes with it:
    no OCR lines means no source boxes for the UI to draw, and the
    choice-field override falls back to the LLM's own read on a SCAN (the
    born-digital path reads vector geometry and needs no OCR, so it still
    applies). The source boxes survive on a born-digital PDF, whose text layer
    is free to read - see ungrounded_box_lines - and go only on a scan, where
    obtaining them means the OCR pass this arm exists to skip.

    Returns {"fields": {...flat...}, "template": None, "_raw": {...},
    "_meta": {"prompt_eval_count", "eval_count", "time_s"}} - "template" is
    always None (this schema is universal like vllm's, not per-template
    like bbox-ocr-extractor's, so no classification needed)."""
    if not grounded:
        return _run_ungrounded(pdf_path, model=model, exclude=exclude)
    ocr_df = ocr_to_dataframe(pdf_path)
    raw, meta = extract_vision_grounded(pdf_path, model=model, ocr_df=ocr_df,
                                        return_meta=True, schema_mode=schema_mode,
                                        exclude=exclude)

    fields = flatten_fields(raw, exclude=exclude)

    # Override the LLM's own read of the choice fields with the geometric
    # detector - see resolve_choice_fields_geometrically's docstring for why,
    # and note it abstains (leaving the LLM's answer) rather than guessing.
    choice_overrides = resolve_choice_fields_geometrically(pdf_path, ocr_df=ocr_df,
                                                           exclude=exclude)
    for field_name, override_value in choice_overrides.items():
        if override_value is not None:
            fields[field_name] = override_value
    raw["_choice_field_overrides"] = choice_overrides
    # So the UI can box a choice field by its printed option line rather than
    # by the selected word alone - see boxes.find_choice_line.
    raw["_choice_candidates"] = choice_candidates(exclude)

    return {
        "fields": fields, "template": None, "_raw": raw, "_meta": meta,
        "grounded": True,
        "excluded": list(_excluded_names(exclude)),
        "_ocr_boxes": get_ocr_boxes_normalized(pdf_path, ocr_df=ocr_df),
    }


def flatten_fields(raw, exclude=()):
    """Collapses extract_vision_grounded/extract_vision_only's nested
    {fields, groups} into the single flat dict the UI and the ground-truth
    comparison expect. Shared by both arms so the toggle can only change
    WHERE the values came from, never how they are named or joined.

    The group-to-flat mapping is FLAT_FROM_GROUPS rather than a list written
    out here, so field_sources() (what the picker offers) and this (what the
    result holds) cannot disagree about a name.

    `exclude` drops those names from the output. They should already be
    absent - nothing asked for them - but a model can echo a key it was never
    given, and one that slipped through would render a row the user switched
    off."""
    dropped = _excluded_names(exclude)
    multiple = {name: spec["multiple"] for name, spec in get_group_schema().items()}

    def value_of(d, key):
        return _safe_leaf((d or {}).get(key)).get("value")

    def joined(group_name, sub_field):
        return join_occurrences(raw["groups"].get(group_name), sub_field)

    fields = {name: value_of(raw["fields"], name)
              for name in raw["fields"] if name not in dropped}

    # Every party on the form, not just the first - see join_occurrences,
    # which is why a repeatable role joins while a single-object group (the
    # property) reads one value.
    for flat_name, group, sub_field in FLAT_FROM_GROUPS:
        if flat_name in dropped:
            continue
        if multiple.get(group):
            fields[flat_name] = joined(group, sub_field)
        else:
            fields[flat_name] = value_of(raw["groups"].get(group) or {}, sub_field)

    # "" rather than None for a genuinely-absent field, matching the other
    # two extractors' output convention (app.py's lookup_extracted does
    # `str(value).strip()`, which would render None as the literal string
    # "None" otherwise).
    return {k: ("" if v is None else v) for k, v in fields.items()}


def _run_ungrounded(pdf_path, model=LLM_MODEL, exclude=()):
    """run()'s grounded=False path: one vision call, no OCR anywhere.

    Two things are deliberately absent rather than faked. "_ocr_boxes" is
    empty because there are no recognized lines to point at, so the UI draws
    no source boxes - see app.py, which says so rather than showing a blank
    tab. And the choice-field override runs ONLY on the born-digital path,
    which reads the PDF's own vector strikethrough and needs no OCR; on a
    scan it would need an OCR pass, which is the very cost this arm exists to
    avoid, so the LLM's own read stands."""
    raw, meta = extract_vision_only(pdf_path, model=model, return_meta=True,
                                    exclude=exclude)
    fields = flatten_fields(raw, exclude=exclude)

    with fitz.open(pdf_path) as pdf:
        native = len(pdf[0].get_text().strip()) > 100

    dropped = _excluded_names(exclude)
    choice_overrides = {target: None for target in CHOICE_FIELD_TARGETS.values()
                        if target not in dropped}
    if native:
        choice_overrides = resolve_choice_fields_geometrically(pdf_path, exclude=exclude)
        for field_name, override_value in choice_overrides.items():
            if override_value is not None:
                fields[field_name] = override_value
    raw["_choice_field_overrides"] = choice_overrides
    raw["_choice_candidates"] = choice_candidates(exclude)

    return {
        "fields": fields, "template": None, "_raw": raw, "_meta": meta,
        "grounded": False,
        "excluded": list(dropped),
        # Kept when they are free to obtain - see ungrounded_box_lines. The
        # toggle skips the text in the PROMPT, not the boxes, so a digital
        # form still shows where every value came from.
        "_ocr_boxes": ungrounded_box_lines(pdf_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid extraction: vision-primary, with OCR-recognized "
                     "text supplied as grounding context in the same call "
                     "(see extract_vision_grounded) rather than OCR-text-only."
    )
    # Not required=True: --list-fields is a question about the schema, and
    # answering it must not depend on having a document to hand. Checked
    # below instead, once we know an extraction is actually being asked for.
    parser.add_argument("--pdf", help="Path to the PDF to extract.")
    parser.add_argument("--list-fields", action="store_true",
                        help="Print the extractable field names as a JSON array on "
                             "stdout and exit. This is what the UI's field picker is "
                             "built from, so the schema stays the one place a field "
                             "is defined.")
    parser.add_argument("--exclude", action="append", default=[], metavar="FIELD",
                        help="A field to leave out, by its name from --list-fields. "
                             "Repeatable. Excluded fields never enter the prompt, so "
                             "this saves tokens and time as well as hiding a column. "
                             "Subtractive only - there is no matching way to add a "
                             "field that isn't in the schema.")
    parser.add_argument("--model", default=LLM_MODEL,
                        help=f"Ollama model to use (default: {LLM_MODEL}). Must be vision-capable.")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"),
                        help="Where to write <pdf-stem>.json (default: ./output).")
    parser.add_argument("--schema-mode", default="prompt", choices=list(SCHEMA_MODES),
                        help="How the reply's shape is enforced. 'prompt' (default) asks for "
                             "it in the prompt and parses whatever comes back. 'pydantic' "
                             "passes a JSON Schema generated from the same field definitions, "
                             "so Ollama constrains generation to it.")
    parser.add_argument("--no-grounding", action="store_true",
                        help="Don't give the model the document's text; extract from "
                             "the image alone. "
                             "Roughly 3x faster on a scan, for about 2.6 points of "
                             "accuracy - and no source boxes. Saves nothing on a "
                             "born-digital PDF, whose text layer is read in "
                             "milliseconds.")
    args = parser.parse_args()

    if args.list_fields:
        json.dump(selectable_fields(), sys.stdout)
        sys.stdout.write("\n")
        return
    if not args.pdf:
        parser.error("--pdf is required (except with --list-fields)")

    unknown = [name for name in args.exclude if name not in field_sources()]
    if unknown:
        parser.error("unknown field(s) to --exclude: " + ", ".join(repr(n) for n in unknown)
                     + "\nRun --list-fields to see the valid names.")
    if len(args.exclude) >= len(field_sources()):
        parser.error("every field was excluded - there would be nothing to extract.")

    pdf_paths = [args.pdf]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        print(f"\n=== {pdf_path} ===", file=sys.stderr)
        start = time.monotonic()
        result = run(pdf_path, model=args.model, grounded=not args.no_grounding,
                     schema_mode=args.schema_mode, exclude=args.exclude)
        elapsed = time.monotonic() - start
        print(f"Time taken: {elapsed:.1f}s", file=sys.stderr)

        out_path = output_dir / f"{Path(pdf_path).stem}.json"
        result["_time_taken_seconds"] = round(elapsed, 1)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
