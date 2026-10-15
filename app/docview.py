"""The document page as clickable HTML, linked to the results table.

Split out of app.py so it can be tested without running the Streamlit script.

The page used to be one flattened image, washes and numbers drawn in by PIL,
which Streamlit shows as a picture: a click on it reaches nobody. Here the page
is a plain image with every highlight laid over it as its own element, inside
a small component frame - so a click knows WHICH highlight it was, and the
script can light up that field's row in the table, and the reverse.

How the frame reaches the table: Streamlit (1.50) runs an HTML component in a
same-origin frame, so its script may touch the main page. It finds the table
rows by the attributes table.py puts on them - data-field and data-links,
our own markup rather than Streamlit's - and listens for clicks on them. If a
browser or a later Streamlit ever refuses that access, the frame still shows
its highlights and numbers and highlights a clicked box on the page alone; the
"#" numbers keep matching the table either way.

Link ids are "page:number", pages from 1 - the same numbering the legend gives
the table's badges (table.badges_by_field), so the two sides cannot disagree.
"""

import html
import json

# A pin sits in the margin left of its line; a line that starts this close to
# the page edge has no margin, so its pin goes on the right instead.
PIN_RIGHT_BELOW = 0.04


def page_overlays(field_boxes, page_count):
    """[[{"number", "box", "labels"}, ...], ...] - one list per page.

    Fields that resolved to the SAME box are one region, not several: on a
    real form the negotiator, the property address and the property type all
    cite lines within a few pixels of each other, and a marker per field
    printed them on top of one another. One number naming every field read
    there says the true thing - this line is where they were all read from.

    Numbered in reading order (top to bottom, then left to right), restarting
    on each page, so the numbers run down the page the way someone checking
    it would."""
    pages = []
    for page_num in range(page_count):
        regions = {}
        for f in field_boxes:
            if f["page"] != page_num:
                continue
            key = tuple(round(v, 4) for v in f["box"])
            labels = regions.setdefault(key, [])
            # By display name, so one field found by two passes - once under
            # its raw group path, once under its flat name - is listed once.
            name = f.get("display") or f["label"]
            if name not in labels:
                labels.append(name)
        ordered = sorted(regions.items(), key=lambda kv: (kv[0][1], kv[0][0]))
        pages.append([{"number": n, "box": box, "labels": labels}
                      for n, (box, labels) in enumerate(ordered, start=1)])
    return pages


def legend(overlays):
    """[(number, "Field · Field"), ...] - the shape table.badges_by_field reads."""
    return [(o["number"], " · ".join(o["labels"])) for o in overlays]


def _pct(v):
    return f"{v * 100:.4f}%"


def doc_view_html(pages, zoom, results_tab="Results", all_filter="All"):
    """The whole frame: pages stacked, each with its highlights and pins, and
    the script linking them to the table. `pages` are {"src", "width",
    "height", "overlays"}; `zoom` is the page width as a percentage of the
    frame (100 = fit). `results_tab` is the label of the tab that shows the
    table: a click on the page while another tab is open switches to it.
    `all_filter` is the row filter that shows every row: a click on a box
    whose row the current filter hides switches to it."""
    parts = []
    count = len(pages)
    for page_number, page in enumerate(pages, start=1):
        items = []
        for o in page["overlays"]:
            x0, y0, x1, y1 = o["box"]
            link = f"{page_number}:{o['number']}"
            title = html.escape(" · ".join(o["labels"]), quote=True)
            items.append(
                f'<div class="region" data-id="{link}" title="{title}" '
                f'style="left:{_pct(x0)};top:{_pct(y0)};width:{_pct(x1 - x0)};'
                f'height:{_pct(y1 - y0)}"></div>')
            side = "pin-right" if x0 < PIN_RIGHT_BELOW else "pin-left"
            anchor = x1 if side == "pin-right" else x0
            items.append(
                f'<button type="button" class="pin" data-id="{link}" title="{title}" '
                f'aria-label="Highlight {o["number"]}: {title}" '
                f'style="left:{_pct(anchor)};top:{_pct((y0 + y1) / 2)}" '
                f'data-side="{side}">{o["number"]}</button>')
        parts.append(
            f'<figure class="page" style="aspect-ratio:{page["width"]}/{page["height"]}">'
            f'<img src="{page["src"]}" alt="Page {page_number}">'
            + "".join(items) +
            f'</figure><p class="cap">Page {page_number} of {count}</p>')

    # A JS string literal; "</" escaped so a label can never end the script.
    tab = json.dumps(str(results_tab)).replace("</", "<\\/")
    every = json.dumps(str(all_filter)).replace("</", "<\\/")
    return (_TEMPLATE.replace("__ZOOM__", f"{int(zoom)}%")
            .replace("__RESULTS_TAB__", tab)
            .replace("__ALL_FILTER__", every)
            .replace("__BODY__", "".join(parts)))


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<style>
:root { --zoom:__ZOOM__; --green:#66984E; --ink:#0f1012; }
html, body { margin:0; padding:0; background:transparent; }
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
       overflow-x:auto; overflow-y:hidden; }
.pages { width:var(--zoom); }
.page { position:relative; margin:0; background:#fff; border-radius:6px; overflow:hidden;
        border:.5px solid rgba(0,0,0,.06); }
.page img { display:block; width:100%; height:auto; user-select:none; }
.cap { font-size:12px; color:#8f8f8f; text-align:center; margin:10px 0 16px; }
.region { position:absolute; background:rgba(102,152,78,.22); border-radius:2px;
          cursor:pointer; transition:background .15s, box-shadow .15s; }
.region:hover { background:rgba(102,152,78,.34); }
.region.on { background:rgba(102,152,78,.30); box-shadow:0 0 0 2px var(--green); }
.pin { position:absolute; transform:translate(calc(-100% - 4px),-50%); min-width:18px; height:18px;
       padding:0 5px; border:0; border-radius:9999px; background:var(--ink); color:#fdfdfd;
       font:400 11px/1 Inter, ui-sans-serif, system-ui, sans-serif; display:grid; place-items:center;
       cursor:pointer; transition:opacity .15s, background .15s; }
.pin[data-side="pin-right"] { transform:translate(4px,-50%); }
.pin:hover, .pin.on { background:var(--green); }
.has-sel .pin:not(.on) { opacity:.25; }
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style></head>
<body><div class="pages" id="pages">__BODY__</div>
<script>
(function () {
  const RESULTS_TAB = __RESULTS_TAB__;
  const ALL_FILTER = __ALL_FILTER__;
  const pagesEl = document.getElementById("pages");
  let P = null;   // the Streamlit page, when the frame may reach it
  try { P = window.parent.document; P.querySelector("body"); } catch (e) { P = null; }

  // Size the frame to its content: Streamlit gives an HTML component a fixed
  // height, and the page's height depends on the pane's width and the zoom.
  function fit() {
    try {
      const h = document.documentElement.scrollHeight;
      if (window.frameElement && window.frameElement.style.height !== h + "px") {
        window.frameElement.style.height = h + "px";
      }
    } catch (e) {}
  }
  new ResizeObserver(fit).observe(document.body);
  window.addEventListener("load", fit);
  try { new MutationObserver(fit).observe(window.frameElement, {attributes: true}); } catch (e) {}

  function rows() { return P ? [...P.querySelectorAll("table.comparison tbody tr[data-field]")] : []; }
  function linksOf(tr) { return (tr.dataset.links || "").split(" ").filter(Boolean); }

  // The selection lives on the Streamlit page, not in this frame: the frame is
  // rebuilt on zoom and the table on every filter click, and both put it back.
  function getSel() { return P ? (P.__lofSel || null) : (window.__lofSel || null); }
  function setSel(s) { if (P) P.__lofSel = s; else window.__lofSel = s; }

  function paint(scrollTo) {
    const sel = getSel();
    document.querySelectorAll(".on").forEach(e => e.classList.remove("on"));
    rows().forEach(tr => tr.classList.remove("sel"));
    pagesEl.classList.toggle("has-sel", !!sel);
    if (!sel) return;
    let boxes = [], hit = [];
    if (sel.kind === "box") {
      boxes = [sel.id];
      hit = rows().filter(tr => linksOf(tr).includes(sel.id));
    } else {
      hit = rows().filter(tr => tr.dataset.field === sel.field);
      boxes = hit.length ? linksOf(hit[0]) : [];
    }
    boxes.forEach(id => document.querySelectorAll(`[data-id="${id}"]`).forEach(e => e.classList.add("on")));
    hit.forEach(tr => tr.classList.add("sel"));
    if (scrollTo === "table" && hit[0]) hit[0].scrollIntoView({block: "center", behavior: "smooth"});
    if (scrollTo === "page") {
      const first = document.querySelector(`.region[data-id="${boxes[0]}"]`);
      if (first) first.scrollIntoView({block: "center", inline: "center", behavior: "smooth"});
    }
  }

  // The results card is hidden while the Input tab is open, and a hidden row
  // can be neither seen nor scrolled to.
  function resultsHidden() {
    const card = P && P.querySelector(".st-key-card_results");
    return !!card && getComputedStyle(card).display === "none";
  }

  function selectedRows() {
    const sel = getSel();
    if (!sel) return [];
    return rows().filter(tr => sel.kind === "box" ? linksOf(tr).includes(sel.id)
                                                  : tr.dataset.field === sel.field);
  }

  // The "All" pill of the row filter, unless it is already the one chosen.
  // Pills read "All  12", so match the label before the count.
  function allPillToClick() {
    const pills = [...P.querySelectorAll(".st-key-row_filter button")];
    const all = pills.find(b => b.innerText.trim().split(/\s+/)[0] === ALL_FILTER);
    if (!all) return null;
    // Streamlit (1.50) marks the chosen pill by its test id.
    return all.dataset.testid === "stBaseButton-pillsActive" ? null : all;
  }

  // Bring the selected row into view, however the table is showing now: the
  // Results tab may be closed (the Input tab is open) and the row filter may
  // hide the row ("No record" while the box is a matched field). Each fix is
  // a Streamlit rerun, a moment away, so this waits and looks again - one
  // step per rerun, never two clicks racing. Picked up again by a fresh copy
  // of this frame if a rerun rebuilds it.
  function revealSelected(tries) {
    if (!P) return;
    const done = () => { P.__lofScrollPending = false; P.__lofStep = null; };
    if (!getSel() || tries <= 0) return done();
    let step = null;
    if (resultsHidden()) step = "tab";
    else if (!selectedRows().length && allPillToClick()) step = "filter";
    else if (selectedRows().length) { done(); paint("table"); return; }
    if (step && P.__lofStep !== step) {
      P.__lofStep = step;
      if (step === "tab") {
        const tab = [...P.querySelectorAll(".st-key-work_view button")]
          .find(b => b.innerText.trim() === RESULTS_TAB);
        if (tab) tab.click();
      } else {
        allPillToClick().click();
      }
    }
    P.__lofScrollPending = true;
    setTimeout(() => revealSelected(tries - 1), 100);
  }

  // Page -> table: a highlight or its pin.
  pagesEl.addEventListener("click", e => {
    const el = e.target.closest("[data-id]");
    if (!el) return;
    const cur = getSel();
    const next = cur && cur.kind === "box" && cur.id === el.dataset.id ? null
                 : {kind: "box", id: el.dataset.id};
    setSel(next);
    paint("table");
    // The row may be out of sight: on the Input tab, or hidden by a filter.
    if (next && P) { P.__lofStep = null; revealSelected(80); }
  });

  if (P) {
    // Table -> page. One listener on the page, replacing the one an earlier
    // copy of this frame left behind, so clicks are never handled twice.
    if (P.__lofClick) P.removeEventListener("click", P.__lofClick, true);
    P.__lofClick = function (e) {
      try {
        const tr = e.target.closest && e.target.closest("table.comparison tbody tr[data-field]");
        if (!tr) return;
        const cur = getSel();
        setSel(cur && cur.kind === "row" && cur.field === tr.dataset.field ? null
               : {kind: "row", field: tr.dataset.field});
        paint("page");
      } catch (err) {}
    };
    P.addEventListener("click", P.__lofClick, true);
    // The table is re-drawn on every rerun (a filter, a zoom); keep the
    // selection painted on whatever rows are there now.
    try {
      if (P.__lofObserver) P.__lofObserver.disconnect();
      P.__lofObserver = new MutationObserver(() => { try { paint(); } catch (err) {} });
      const results = P.querySelector(".st-key-card_results") || P.body;
      P.__lofObserver.observe(results, {childList: true, subtree: true});
    } catch (err) {}
  }
  paint();
  fit();
  if (P && P.__lofScrollPending) revealSelected(80);
})();
</script></body></html>"""
