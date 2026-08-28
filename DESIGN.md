# lof2data — Design Approach

## Goal

Extract structured fields from a PropNex "Letter of Offer to Rent" PDF and
show the result verified against the real record (fetched by Sale No.),
with a visual trace of where each value was read from on the source
document.

## Architecture: subprocess delegation

`app/app.py` is a lightweight Streamlit UI with no heavy ML
dependencies of its own. It shells out to a separate, heavy-dependency
venv (`engine/.venv`) to run the actual extraction:

```
app/app.py --[subprocess]--> engine/extractor.py --pdf <file> --output-dir <dir>
```

`engine/extractor.py` writes its result as JSON; `app/app.py` reads that JSON
back and renders it. This keeps the UI venv fast to install/run while the
extraction venv can carry PaddleOCR, OpenCV, PyMuPDF, etc. without
bloating the UI environment.

## Extraction design: vision-primary + OCR-as-grounding

The page image is sent to the vision-LLM as the **primary source of
truth**, with PaddleOCR's recognized text supplied **in the same call**
as supplementary grounding context (indexed as `[N] "text" (confidence)`)
— never as a separate text-only step, and never as a replacement for the
image. This was chosen after this project's own A/B testing showed
OCR-text-only extraction underperforms on this kind of layout/mark-
sensitive form (handwriting, circled/struck-through choices — none of
which OCR text can represent). Full rationale and iteration history live in
`RECOMMENDATION.md` in the `hybrid/` project this was extracted from — not
part of this repo.

On top of that single LLM call, two fields — `Description of
Property/Property Type` and `Furnished Status` — are **overridden with a
deterministic geometric detector** rather than trusted to the LLM's
visual read at all. Testing showed the LLM is unreliable on these two
specific fields regardless of prompting (it's genuine visual ambiguity,
not sampling noise — confirmed by 5 repeated near-deterministic calls
still getting it wrong 4/5 times). The override reuses
`bbox-ocr-extractor`'s `detect_selected_choice_native()` (reads the PDF's
actual vector strikethrough for the native-text template) and
`detect_circled_choice()` (OpenCV circle detection for the scanned
template).

## OCR engine: classic PaddleOCR, not PaddleOCR-VL

`engine/extractor.py` uses the
classic `PaddleOCR` detection+recognition pipeline (`PP-OCRv6`), not the
newer `PaddleOCR-VL` document-parsing model. Benchmarked on this machine
(Apple Silicon, no CUDA — `paddlepaddle-gpu` doesn't apply): `PaddleOCRVL`
took ~315s for a single page vs. classic PaddleOCR's ~54s — about 6x
slower, which would push total extraction time from ~200s toward 500s+ on
a multi-page document. PaddleOCR-VL genuinely reads short choice-field
words more accurately (it captured "FULLY" in full where classic
PaddleOCR dropped the trailing letter to "FULL"), but that accuracy risk
is already mitigated elsewhere — by the deterministic choice-field
override above, and by the box-matching prefix fallback below — so the
6x latency cost wasn't worth paying for no net accuracy gain.

## Bounding-box design: template-agnostic, not per-form-layout

Every extracted field's box comes from one of two sources, in
`app/app.py`'s `collect_field_boxes()`:

1. **The LLM's own `source_index` citation** (`BOX_COLOR_CITED`, blue) —
   every field the LLM extracts is asked to also cite which OCR-recognized
   line backed a printed value. When present, that OCR line's box (from
   `engine/extractor.py`'s `get_ocr_boxes_normalized()`) is used directly.
2. **A text-match fallback** (`BOX_COLOR_MATCHED`, orange) — for values
   the LLM can't cite an OCR line for (chiefly the 2 choice fields, whose
   value comes from a circled/struck visual mark, not printed OCR text),
   `find_box_by_text()` searches the OCR result directly for a line
   containing the extracted value as a substring, falling back to a
   4-character prefix match to absorb the "OCR drops a trailing character"
   failure mode described above.

This deliberately has **no knowledge of which known form layout
(form1/form2/form3) the page uses** — an earlier version computed choice-
field boxes via `bbox_extractor.classify_template()` (the same per-
template classification the accuracy override uses), but that assumed
real-world usage is limited to the 3 known templates, cost an extra ~65s
of redundant full-page OCR per run (classify_template was being run twice
per extraction), and would silently produce no box at all for any form
outside those 3. The current approach works for any layout the OCR can
read text from, at no extra OCR cost, since it reuses the same OCR pass
already run for grounding.

## Model selection: two benchmarked models, cloud gated on quota

The UI offers a choice of extraction model, passed straight through to
`engine/extractor.py --model`. `app/models.py` owns everything that
talks to the Ollama daemon and every decision about whether a model is
offerable, keeping `app.py` about UI.

Exactly **two** models are offered, as a fixed list rather than everything
Ollama has installed:

| Model | Where | A/B accuracy | Full pipeline | Time/doc |
|---|---|---|---|---|
| `minimax-m3:cloud` | cloud | 92.0% | **93.8%** | ~87s A/B, ~197s full |
| `qwen3-vl:8b-instruct` | local | 85.7% | not measured | ~254s |

Two different measurements, so keep them apart:

- **A/B accuracy** is `ab_benchmark.py`'s OCR-grounded arm (`B_vision_grounded`),
  which calls `extract_vision_grounded()` directly and therefore stops *before*
  the geometric choice-field override. It's the number that justified choosing
  OCR-grounding over vision-only, not the number the app delivers.
- **Full pipeline** is `run()` — what `app/app.py` actually shells out to,
  override included — scored against the same answer key.

Both are 4 documents × 28 fields against the hand-built `ground_truth.json`
in the sibling `hybrid/` project (neither the benchmark nor its answer key is
part of this repo; the documents carry real NRICs). Scoring gives no substring
credit, so a partial address or a truncated clause counts as wrong.

The local model has only ever been measured on the A/B arm, so its full-pipeline
figure is left blank rather than assumed to gain the same ~1.8 points.

Other installed models may be vision-capable, but they are unmeasured on this
document type, so they aren't offered — the list is meant to widen only after a
model has been benchmarked, not automatically.

**Rate-limit gating.** A cloud model that has exhausted its quota is removed
from the dropdown and shown beneath it with the reason, rather than left
selectable to fail mid-run. The distinction that matters is between the two
kinds of HTTP 429 `engine/extractor.py`'s `call_llm()` already separates:

- a transient burst limit, which `call_llm` retries with backoff and gets
  through — the model is fine and **stays selectable**; and
- a `session usage limit` 429, a quota that won't clear on retry timescales,
  which `call_llm` deliberately fails fast on — the model is **blocked**.

Treating every 429 as fatal would lock the user out of a working model, so
only the second disables anything.

Availability is established two ways, because neither alone is sufficient:

1. **A probe at page load** — a `num_predict: 1` call (~1.4s, ~180 tokens)
   to each cloud model, so a quota that was already exhausted is caught
   before the user spends a run and its OCR pass discovering it. Cached for
   120s via `st.cache_data`, since Streamlit reruns the whole script on every
   widget interaction and this is a real network round-trip.
2. **Marking on failure** — if an extraction fails with the quota error, that
   model joins a session-level blocked set that overrides the cached probe.
   A limit hit *during* a long run would otherwise leave it selectable until
   the TTL expired.

A **Re-check** button clears both the cache and the session set, for when the
quota window resets. Local models are never probed: there is no quota to hit,
and a cold local model has to load into memory first, which would cost far
more page-load latency than a cloud round-trip.

`models.py`'s decision logic is pure and covered by `test_models.py`
(stdlib `unittest`, no new dependency — `.venv/bin/python3 test_models.py`).
Getting `rate_limited` vs `available` wrong either locks the user out of a
working model or wastes a full extraction on a dead one, so that
classification is tested directly rather than only through the UI.

## Ground-truth comparison

`sale_no` is used to fetch the real record via
`vision-form-reader/api_ground_truth.py` (`fetch_sale`,
`build_canonical_ground_truth`), then compared field-by-field against the
extraction using type-aware matchers (`dates_match`, `amounts_match`,
`ids_match`, `choice_match`, falling back to `compare.py`'s
`values_match`). `VLLM_ALIASES` maps each canonical ground-truth field
name to the engine's own flat output field names, since `engine/extractor.py`
deliberately mirrors `vllm-schema-extractor`'s field naming so this same
alias map (already used by `streamlit_app/app.py`) works unmodified.
