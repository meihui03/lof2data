# lof2data

Extracts structured fields from a PropNex **Letter of Offer to Rent** PDF and
shows the result verified against the real CRM record, with bounding boxes on
the source document marking where each value was read from.

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
| `app/models.py` | which models are offerable, and cloud rate-limit gating |
| `app/boxes.py` | turns an LLM `source_index` citation into boxes to draw, and locates a box by text when there's no citation |
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

Enter a Sale No., upload the LOF PDF, pick a model, hit Extract.

The engine also runs standalone:

```bash
cd engine
.venv/bin/python3 extractor.py --pdf /path/to/lof.pdf --model minimax-m3:cloud
```

## Models

Two are offered, both benchmarked on real LOF forms:

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

## Tests

Stdlib `unittest`, no extra dependency:

```bash
app/.venv/bin/python3 app/test_boxes.py
app/.venv/bin/python3 app/test_models.py
app/.venv/bin/python3 app/test_groundtruth.py
app/.venv/bin/python3 app/test_app_ui.py
engine/.venv/bin/python3 engine/test_extractor_flatten.py
engine/.venv/bin/python3 engine/test_schema_models.py
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
