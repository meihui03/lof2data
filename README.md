# lof2data

Extracts structured fields from a PropNex **Letter of Offer to Rent** PDF and
shows the result verified against the real CRM record, with the source
document highlighted and numbered where each value was read from.

Handles born-digital PDFs, scans, and photographed/CamScanner documents —
including handwriting and choice fields marked by circling or striking out
one option in a printed list.

## How it works

The page image goes to a vision LLM as the **primary** source of truth, with
PaddleOCR's recognized text supplied **in the same call** as grounding
context — not as a separate text-only pass, and not as a replacement for the
image. Two fields whose value comes from a visual mark rather than text are
then overridden by a deterministic geometric detector.

A/B measured across 4 documents × 28 fields, OCR grounding is worth **+2.7
points** on a strong cloud model and **+10.7** on a local 8B — it mainly
repairs character-level misreads of printed text (`OOI`→`DOI`, `B2C`→`BZC`),
which is largely a small-model failure. Full rationale, benchmark numbers and
the iteration log are in [DESIGN.md](DESIGN.md).

```
                  ┌─ page image ──────────────┐
   PDF ───────────┤                           ├──► vision LLM ──► JSON
                  └─ PaddleOCR text [N] ──────┘         │
                                                        ▼
                                          geometric override for the
                                          circled / struck-through fields
```

## Layout

Two packages with **separate virtualenvs**, on purpose: the UI stays light
and fast to install, while the engine carries PaddleOCR, OpenCV and PyMuPDF.
The UI shells out to the engine as a subprocess.

```
app/       Streamlit UI — model selector, comparison table, box overlay
engine/    extraction — OCR, prompting, choice-field detection
```

| File | What it does |
|---|---|
| `app/app.py` | the UI; shells out to the engine, renders results |
| `app/fields.py` | groups the engine's field list for the picker, and hides a removed field's row |
| `app/models.py` | which models are offerable, and cloud rate-limit gating |
| `app/boxes.py` | turns an LLM `source_index` citation into the region to highlight, and locates one by text when there's no citation |
| `app/table.py` | the comparison table's markup — hand-built so a long value is never clipped |
| `app/groundtruth.py` | fetches the CRM record and matches values against it |
| `engine/extractor.py` | OCR, prompt building, extraction, flattening |
| `engine/mark_detector.py` | template-free reader for circled/struck choices |

## Run with Docker

The quickest way in, and the one to hand someone else. Needs Docker and a
running [Ollama](https://ollama.com) on the host, with a vision model pulled:

```bash
ollama pull qwen3-vl:8b-instruct   # skip if you already have a vision model
docker compose up --build
```

Then open <http://localhost:8501>.

`.env` has to sit next to `docker-compose.yml` — if you were handed this folder
it's already there; if you cloned the repo, `cp .env.example .env` and fill in
`STAGING_API_PASSWORD` first. Compose reads it at startup, so the credential is
never baked into the image.

If port 8501 is already taken (`address already in use` — another Streamlit app
is the usual culprit), change `"8501:8501"` to `"8601:8501"` in
`docker-compose.yml` and use <http://localhost:8601> instead.

The first build downloads ~1GB of wheels and takes 10–15 minutes; after that
`docker compose up` starts in seconds. PaddleOCR fetches its OCR models on the
first extraction and keeps them in a named volume, so that cost is one-time too.

Ollama stays on the host rather than in the compose file — it wants GPU access,
and it's usually already installed. The container reaches it at
`host.docker.internal`. To use the cloud model instead, sign in to your own
ollama.com account on the host (`ollama signin`); the app probes for it at page
load and offers whichever models actually work on your machine.

The image is pinned to `linux/amd64` because `paddlepaddle` publishes no
linux/arm64 wheel. On Apple Silicon it runs emulated — fine for trying it out,
noticeably slower for real extraction.

## Setup without Docker

Requires Python 3.9+ and a running [Ollama](https://ollama.com) with at least
one vision model pulled.

```bash
git clone https://github.com/meihui03/lof2data.git
cd lof2data

# UI environment
python3 -m venv app/.venv
app/.venv/bin/pip install -r app/requirements.txt

# Engine environment (heavier — PaddleOCR et al.)
python3 -m venv engine/.venv
engine/.venv/bin/pip install -r engine/requirements.txt

# Models: one cloud, one local
ollama pull minimax-m3:cloud
ollama pull qwen3-vl:8b-instruct

cp .env.example .env      # then fill in STAGING_API_PASSWORD
```

## Running

```bash
app/.venv/bin/streamlit run app/app.py
```

Enter a Sale No., upload the LOF PDF, pick a model, choose which fields you
want, hit Extract.

**Choosing fields.** Step 2 lists every field the engine can extract, all
ticked. Untick one and it is left out of the prompt entirely — not extracted
and then hidden, so it costs no tokens and no time, and its row disappears
from the comparison table instead of showing up empty and scored as a miss.
The list is the engine's own (`--list-fields`), so the schema stays the single
place a field is defined; the picker can only ever narrow it, never add to it.

The engine also runs standalone:

```bash
cd engine
.venv/bin/python3 extractor.py --pdf /path/to/lof.pdf --model minimax-m3:cloud

# what can be extracted, and leaving some of it out
.venv/bin/python3 extractor.py --list-fields
.venv/bin/python3 extractor.py --pdf /path/to/lof.pdf \
    --exclude "Witnessed by" --exclude "Notice to Quit (months)"
```

## Models

Offered in the app: `qwen3-vl:8b-instruct` (local, selected by default so a run
never spends API credit unasked) and `claude-sonnet-5-5` (below).
`minimax-m3:cloud` is switched off in `app/models.py` — past its free allowance
it needs a paid Ollama subscription, and the Claude API covers the hosted
option. Uncomment its line to offer it again.

Both Ollama models were benchmarked on real LOF forms:

| Model | | Accuracy | Time/doc |
|---|---|---|---|
| `minimax-m3:cloud` | cloud | 93.8% | ~197s |
| `qwen3-vl:8b-instruct` | local | 90.2% | ~561s |

Measured end-to-end through `run()` — the path the app actually takes, geometric
override included — on 4 documents × 28 fields, with no substring credit. See
[DESIGN.md](DESIGN.md) for how this differs from the A/B benchmark figure.

3.6 points and ~3x the wall-clock is the real trade. Neither model dominates
field-by-field: the local one won a document outright, and on one address the
two misread *different* halves of the same line. The local model's remaining
weaknesses are concentrated — character-level misreads on handwriting, and
dropping the second party from a multi-occurrence role (a company tenant's
individual signatory, a second witness).

A cloud model that has exhausted its quota is removed from the dropdown with
the reason shown, rather than left selectable to fail mid-run — detected by a
1-token probe at page load and by marking the model on a failed run. **Re-check**
clears both once the quota resets.

To offer another model, add it to `SUPPORTED_MODELS` in `app/models.py`. The
list is deliberately fixed rather than auto-discovered: other installed models
may be vision-capable but unmeasured on this document type.

### Claude API (`claude-sonnet-5-5`)

A third option, **not yet benchmarked** like the two above. Set
`ANTHROPIC_API_KEY` in `.env` and it appears in the selector as
`claude api`; without the key it is listed as unavailable with the reason.
An organization-level key (refused with "not scoped to a workspace") also
needs `ANTHROPIC_WORKSPACE_ID=wrkspc_...` in `.env`. The
page-load check is a free model lookup — it confirms the key and spends no
tokens. Running out of prepaid credit only shows on a real run, and takes the
model out of the selector the same way a spent Ollama quota does.

Same prompt, same page images, same OCR grounding and choice-field override as
the Ollama models — only the call differs (`call_claude` in
`engine/extractor.py`). It runs at low effort, since output (thinking
included) is the expensive side. Rough cost, grounded: ~9–10k input and
~1.5–4.5k output tokens per page, about **$0.05 per page** at $2 / $10 per
million tokens.

Unlike the other two, this one sends the pages — names, NRIC numbers,
addresses — to Anthropic's API.

## Tests

Stdlib `unittest`, no extra dependency:

```bash
app/.venv/bin/python3 app/test_boxes.py
app/.venv/bin/python3 app/test_models.py
app/.venv/bin/python3 app/test_groundtruth.py
app/.venv/bin/python3 app/test_fields.py
app/.venv/bin/python3 app/test_table.py
app/.venv/bin/python3 app/test_docview.py
app/.venv/bin/python3 app/test_app_ui.py
engine/.venv/bin/python3 engine/test_extractor_flatten.py
engine/.venv/bin/python3 engine/test_schema_models.py
engine/.venv/bin/python3 engine/test_field_selection.py
engine/.venv/bin/python3 engine/test_claude.py
engine/.venv/bin/python3 engine/test_mark_detector_native.py
```

## Notes

**No sample documents are included.** Real Letters of Offer carry names,
NRIC/passport numbers and home addresses, so PDFs, extraction output and
ground-truth answer keys are all gitignored. Bring your own to run this.

**Credentials.** `.env` is gitignored; `STAGING_API_PASSWORD` never belongs in
a commit. Note that the sales API takes the password as a URL query parameter,
which tends to surface in server and proxy logs — worth revisiting if this
moves beyond staging.

**Some comments reference sibling projects** (`bbox-ocr-extractor`,
`vllm-schema-extractor`, `PDF-Form-Reader`) from the multi-project layout this
was extracted from. They're kept because they explain *why* a decision was
made; those projects aren't part of this repo.
