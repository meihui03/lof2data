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

### Field descriptions name a role, not a label

A description is the only thing telling the model what a field IS, so it has
to survive a template it has never seen - the same constraint the box design
and the choice-field detector are held to. The agent group failed that test:
it required a name to be "printed next to an 'Attn:'/'Negotiator(s):' label",
which is a convention of the templates seen so far, not a definition of the
role. A form labelling the same person "Agent:" or "Marketed by:" would have
returned nothing, and the failure is silent - an empty field looks exactly
like a form that genuinely has no negotiator.

It now describes the role (the agency's representative on the deal, as
opposed to a party to the tenancy) and offers the labels as examples. The
guard that motivated the original wording is kept, because it came from a
real misattribution: a landlord, tenant, or witness name must not be promoted
into this role just because it appears on the page. Note the negotiator often
signs as the witness as well; that is handled downstream by the comparison
row accepting either (see the picker's paired fields), not by loosening what
a negotiator is.

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

1. **The LLM's own `source_index` citation** — every field the LLM
   extracts is asked to also cite which OCR-recognized
   line backed a printed value. When present, that OCR line's box (from
   `engine/extractor.py`'s `get_ocr_boxes_normalized()`) is used directly.
2. **A text-match fallback** — for values
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

Both sources are now drawn in **one colour**. The blue/orange split used to
encode which of the two found the region, but that is a fact about the
lookup, not something a reader of the document can act on — both mean the
same thing to them: this is where the value was read.

### Drawn as a highlight, not an outline

The regions are rendered as a translucent wash over the text rather than a
stroked rectangle. The reader's task is to check whether the marked text IS
the value, so the text has to stay legible through the mark; a 3px outline on
a tight box lands directly on the glyphs it is pointing at. Alpha is the whole
trade — visible at a glance on a grey scan, transparent enough to read printed
text through — and 64/255 holds both.

Every wash on a page goes onto one overlay layer, composited in a single pass.
Drawn straight onto the page, two fields sharing a line (the deposits on an
Item/Description/RM row) would stack their alpha and come out darker than the
rest, reading as a stronger match when it is only a busier line.

### Numbered, not named, on the page

Each region carries a numbered badge in the left margin, and the field names
go in a legend under the page. Writing the name beside its wash was tried
first and failed on a real form for two reasons at once: several fields cite
the **same** OCR line — the negotiator, the property address and the property
type all resolved to within a few pixels of each other — so their labels
printed on top of one another; and on a form whose lines are ~28px apart at
render scale, a label sitting above its wash lands on the line above's text.

A badge collides with neither. It is small, it sits in the margin the form
already leaves, and fields sharing a line share one badge whose legend entry
lists their names — which is the true statement anyway: this line is where both
were read from. The badge falls to the right of the wash for a clause printed
hard against the left edge, since a line's tail is nearly always blank while
its head is the text being pointed at.

A field is named in the legend the way the comparison table names it. Boxes
are found in two passes - one over the model's raw reply, which nests a party
under its group (`agent`/`Name`), and one over the flattened fields the table
shows (`Negotiator/Agent Name`) - and both land on the same line, so the
legend printed one line twice under two names for one field. The engine
publishes its own `field_sources()` mapping in the result (beside
`_choice_candidates`, and for the same reason: the UI needs the schema's
mapping and must not keep a second copy to drift). A sub-field with no
flattened name - a party's designation or signature date, extracted and boxed
but never shown in the table - is written out as words instead of the raw
path.

Only fields the reader can act on are drawn. A group can carry a sub-field the
UI has no other trace of — a witness's NRIC, extracted so the model can tell
one signature block from another, offered by no checkbox and shown in no row.
Highlighting it numbered a region in the legend for something nobody asked for
and nothing can switch off, so a sub-field is now drawn only when it feeds a
flattened field.

A party's **designation and signature date** were in that same position and
have been removed from the schema outright rather than merely hidden. They
cost prompt tokens on every run — the binding constraint on the local model —
to produce four values no picker offered, no row showed, and nothing scored.
Note what this gives up: they were plausibly helping the model keep a company
signatory apart from the company itself, so if the tenant group starts
merging those two, this is the first change to suspect.

Note the regions are still **line-level**: `ocr_to_dataframe` emits one row per
recognized line, so a wash covers the line a value sits on rather than the
value itself. Word-level boxes would tighten it, and are the one idea worth
taking from the article `verify_source_index` cites.

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
| `qwen3-vl:8b-instruct` | local | 85.7% | **90.2%** | ~254s A/B, ~561s full |

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

The override is worth more to the weaker model — **+4.5 points** locally against
**+1.8** on the cloud one — which is what you'd expect: it replaces exactly the
visual judgement an 8B is worst at with a deterministic read.

Two caveats the headline numbers hide:

- **The override's abstention leaks on a weak model.** When the detector finds a
  choice line with nothing marked, it returns `None`, which means "keep the LLM's
  answer" — so an unmarked Furnished Status became `"Not"` and `"Partly"` on two
  documents. That's 2 of the local model's 11 misses. `None` currently conflates
  "found the line, definitively unmarked" (the answer is `""`, and the detector
  knows it) with "couldn't locate a choice line at all" (genuinely defer). Only
  the second deserves deference. Invisible on the cloud model, which answers `""`
  unprompted.
- **OCR grounding reduces character-level misreads, it doesn't eliminate them.**
  The local model still produced `NICHOLAS DOI` for `NICHOLAS OOI` with grounding
  active — the very example cited above. The +10.7 A/B gain is a corpus-level
  statistical improvement, not a guarantee on any given field.

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

## Field selection: subtractive, and applied to the prompt

Step 2 lets the user drop fields before a run. Two decisions worth recording.

**It narrows the prompt, not the output.** The obvious implementation is to
extract everything and hide the columns nobody wanted; that is one line of UI
code and costs the user nothing they can see. It also spends the tokens and
the wall-clock on every dropped field, on a local 8B where prompt length is
the binding constraint — the same constraint the grounding toggle exists to
relieve. So an excluded field is removed from the prompt's field list, from
the constrained (`--schema-mode pydantic`) JSON Schema, and from the geometric
choice-field pass, which is a page-scale pixel read worth skipping in its own
right.

**It can only subtract.** There is no way to add a field from the UI, and no
free-text box that looks like one. A field the schema doesn't describe has no
prompt line, no type, no place in `flatten_fields`, and no ground-truth column
to be compared against — it would extract into nothing. Adding a field is a
schema edit, and the UI is built from the schema (`--list-fields`) rather than
from a list of its own, so it follows automatically.

The mapping from a flattened name to the group sub-field it comes from
(`FLAT_FROM_GROUPS`) is now the single source of truth shared by
`flatten_fields()` and `field_sources()`. It used to exist only inside the
flattener; publishing a second copy for the picker would have drifted, and a
drifted name fails silently — a checkbox that switches nothing off, or a table
row that can never be filled.

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
