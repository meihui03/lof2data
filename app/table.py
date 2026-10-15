"""Building the comparison table's HTML.

Split out of app.py for the usual reason - importing app.py runs the whole
Streamlit script - but also because this is the one view a reader actually
works in, and its markup deserves tests.

Why hand-built rather than st.table or st.dataframe:

  st.dataframe renders to a canvas grid, which CLIPS a value too long for its
  column ("A-34-08, JALAN DUA, RESIDENSI TROFI 1, 55200, S...") and cannot be
  reached by CSS. A truncated value can't be checked against the record beside
  it, which is the whole job.

  st.table wraps, but styling it means targeting Streamlit's own DOM
  (div[data-testid="stTable"] th, and the like). That was tried: the header
  came out identical to the data, so the eye had nothing to anchor on, and the
  fix would have been another guess at an internal that is free to change
  between versions.

Owning the markup makes the header, the row tint, the result pills and the
diff marks a property of this file instead of a property of Streamlit's
stylesheet.

The rows are the dicts app.py builds, keyed by the CSV's own column names -
the CSV is written from the same dicts, so the two can't disagree. Only the
on-screen table renames and reorders them, and only it carries "Where": the
numbered badges linking a row to its highlight on the page image.
"""

import csv
import html
import io
import re
from collections import Counter
from difflib import SequenceMatcher

# The three states of the Match column. Defined here and used by app.py when it
# builds a row, so the CSV and this table can't drift apart on what a tick is.
MATCH = "✔️"
MISS = "✖️"
NA = "—"

# What each state renders as: a pill, and the class that colours it. Plain
# characters rather than the emoji above - an emoji ignores CSS colour on most
# platforms, so a wrong row would print in the same ink as a right one.
_RESULTS = {
    MATCH: ("✓", "ok"),
    MISS: ("✗", "bad"),
    NA: ("No record", "na"),
}

# Row keys. The CSV is written with these names, in this order.
FIELD, TRUTH, EXTRACTED, RESULT, WHERE, LINKS = (
    "Field", "Ground Truth (API)", "Extracted", "Match", "Where", "Links")
CSV_COLUMNS = (FIELD, TRUTH, EXTRACTED, RESULT)

# The on-screen table: the extracted value first, since it is what is being
# checked, with the record beside it as the reference.
COLUMNS = (FIELD, EXTRACTED, TRUTH, RESULT)
COLUMNS_WITH_WHERE = (WHERE,) + COLUMNS
HEADERS = {WHERE: "#", TRUTH: "Record (API)", RESULT: "Result"}

# Row filters, in the order the pills show them.
ALL, NEEDS_LOOK, MATCHED, NO_RECORD = "All", "Needs a look", "Matched", "No record"
FILTERS = (ALL, NEEDS_LOOK, MATCHED, NO_RECORD)


def _blank(value):
    return value is None or str(value).strip() in ("", NA)


def _cell(value):
    """One value as escaped HTML, with a blank rendered as the em dash the
    rest of the table uses rather than an empty cell that reads as a bug."""
    text = "" if value is None else str(value).strip()
    if not text:
        text = NA
    return html.escape(text)


def _result(value):
    """The Result column: a coloured pill when recognised, otherwise the value
    escaped as-is - an unknown state should show, not vanish."""
    label, css_class = _RESULTS.get(str(value).strip(), (None, None))
    if label is None:
        return f'<span class="st">{_cell(value)}</span>'
    return f'<span class="st {css_class}">{label}</span>'


def _where(labels):
    if not labels:
        return '<span class="nobadge">–</span>'
    return "".join(f'<span class="badge">{html.escape(str(label))}</span>' for label in labels)


# --------------------------------------------------------------- diff marks

# A "word" of a value: split on spaces, commas, slashes and semicolons, which
# is how addresses, names and amounts on these forms come apart.
_WORD = re.compile(r"[^\s,/;]+")

# How alike two leftover words must be to count as one word misread, rather
# than two unrelated words. "B-9-16" / "B-9-18" is 0.83; two formats of one
# company number are far below it.
_PAIR_MIN_RATIO = 0.5


def _words(text):
    return [(m.start(), m.end(), m.group().lower()) for m in _WORD.finditer(text)]


def _unmatched(words, other):
    """The words of one value that the other value doesn't also contain, as a
    multiset - so order doesn't matter, but a repeated word must be repeated."""
    available = Counter(w for *_, w in other)
    left = []
    for word in words:
        if available[word]:
            available[word] -= 1
        else:
            left.append(word)
    return left


def _render(text, marked):
    """`text` escaped, with each run of character positions in `marked`
    wrapped in <mark class="diff">."""
    out, i = [], 0
    while i < len(text):
        j = i
        flag = i in marked
        while j < len(text) and (j in marked) == flag:
            j += 1
        chunk = html.escape(text[i:j])
        out.append(f'<mark class="diff">{chunk}</mark>' if flag else chunk)
        i = j
    return "".join(out)


def diff_html(a, b):
    """(a, b) as escaped HTML, with the characters that differ marked.

    Matched word by word first, ignoring case and spacing, because the record
    and the page often list the same parts in a different order ("B-9-18,
    JALAN KEMAJUAN, PJ MIDTOWN" against "B-9-16, PJ MIDTOWN, Jalan
    Kemajuan"): a straight character diff of those marks "6, PJ MIDTOWN"
    against "8". Words found in both are left alone. The rest are paired by
    similarity, and only inside a pair are characters compared - so the mark
    lands on "16" / "18".

    A word with no counterpart is not marked: it is usually one side saying
    more ("..., 46200 Petaling Jaya, Selangor" against a record that stops at
    the postcode), not a misread. Values with nothing in common get no marks
    at all - there is no single place to point at."""
    a, b = "" if a is None else str(a), "" if b is None else str(b)
    left_a = _unmatched(_words(a), _words(b))
    left_b = _unmatched(_words(b), _words(a))
    marked_a, marked_b, taken = set(), set(), set()

    for start_a, _, word_a in left_a:
        best, best_ratio = None, _PAIR_MIN_RATIO
        for k, (_, _, word_b) in enumerate(left_b):
            if k in taken:
                continue
            ratio = SequenceMatcher(None, word_a, word_b, autojunk=False).ratio()
            if ratio >= best_ratio:
                best, best_ratio = k, ratio
        if best is None:
            continue
        taken.add(best)
        start_b, _, word_b = left_b[best]
        ops = SequenceMatcher(None, word_a, word_b, autojunk=False).get_opcodes()
        for tag, i1, i2, j1, j2 in ops:
            if tag != "equal":
                marked_a.update(range(start_a + i1, start_a + i2))
                marked_b.update(range(start_b + j1, start_b + j2))

    return _render(a, marked_a), _render(b, marked_b)


# ------------------------------------------------------------------- table

def comparison_table_html(rows, columns=COLUMNS):
    """The full <table> for the given rows.

    `rows` are the dicts app.py already builds, keyed by the CSV's column
    names (plus "Where" when `columns` includes it). A row missing a key
    renders as the em dash rather than raising: a partial row is still worth
    showing.
    """
    head = "".join(f"<th>{html.escape(HEADERS.get(name, name))}</th>" for name in columns)

    body = []
    for row in rows:
        missed = str(row.get(RESULT, "")).strip() == MISS
        if missed and not _blank(row.get(EXTRACTED)) and not _blank(row.get(TRUTH)):
            values = dict(zip((EXTRACTED, TRUTH), diff_html(
                str(row[EXTRACTED]).strip(), str(row[TRUTH]).strip())))
        else:
            values = {EXTRACTED: _cell(row.get(EXTRACTED)), TRUTH: _cell(row.get(TRUTH))}

        cells = []
        for name in columns:
            if name == WHERE:
                cells.append(f'<td class="where">{_where(row.get(WHERE))}</td>')
            elif name == FIELD:
                cells.append(f'<th scope="row">{_cell(row.get(FIELD))}</th>')
            elif name == RESULT:
                cells.append(f'<td class="status">{_result(row.get(RESULT))}</td>')
            elif name == TRUTH:
                cells.append(f'<td class="truth">{values[TRUTH]}</td>')
            elif name == EXTRACTED:
                cells.append(f"<td>{values[EXTRACTED]}</td>")
            else:
                cells.append(f"<td>{_cell(row.get(name))}</td>")
        # The field and its highlight ids ride on the row, so the document
        # pane can find a row from a clicked highlight and the reverse.
        attrs = (f' data-field="{html.escape(str(row.get(FIELD, "")), quote=True)}"'
                 f' data-links="{html.escape(" ".join(row.get(LINKS) or []), quote=True)}"')
        tint = ' class="miss"' if missed else ""
        body.append(f"<tr{tint}{attrs}>" + "".join(cells) + "</tr>")

    return (
        '<div class="comparison-wrap"><table class="comparison">'
        f"<thead><tr>{head}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )


def csv_bytes(rows):
    """The download: the CSV's own columns only. "Where" is a pointer into the
    page image on screen and means nothing in a file, so it never goes in."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction="ignore",
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def unrecorded_rows(extracted, engine_fields, aliases, excluded):
    """Rows for the extracted fields the CRM record has nothing to compare
    against - Tenant Address, Monthly Rental, the tenancy dates.

    The comparison rows only cover fields with a record value to check, so
    these were extracted and then never shown anywhere. Here they get a row of
    their own, in the engine's order, marked No record: shown, but never
    counted as a match or a miss. A field the user unticked has no row, the
    same as on the compared side."""
    covered = {alias for names in aliases.values() for alias in names}
    rows = []
    for field in engine_fields:
        if field in covered or field in excluded:
            continue
        value = extracted.get(field)
        rows.append({FIELD: field, TRUTH: NA,
                     EXTRACTED: value if not _blank(value) else NA, RESULT: NA})
    return rows


# Where a field with no record value sits: right after the compared row it
# belongs with, rather than in the block at the end. Add a pair to place
# another the same way.
ROW_AFTER = {
    "Tenant Address": "Tenant IC",
}


def arrange_rows(compared, extra, after=ROW_AFTER):
    """The table's rows: the compared ones in their order, each followed by
    any extra row anchored to it (`after`), then the remaining extra rows.

    An anchored row whose partner isn't in the table - its field unticked -
    goes to the end with the rest, never dropped: it was ticked, so it shows."""
    present = {row[FIELD] for row in compared}
    placed = {name: [] for name in present}
    tail = []
    for row in extra:
        partner = after.get(row[FIELD])
        (placed[partner] if partner in present else tail).append(row)
    rows = []
    for row in compared:
        rows.append(row)
        rows.extend(placed[row[FIELD]])
    return rows + tail


# ----------------------------------------------------------------- filters

def needs_look(row, has_where):
    """A row worth a second look: the value differs from the record, or it was
    read but couldn't be placed on the page. The second only means something
    when the run had highlights at all - with grounding off, no row has one."""
    if str(row.get(RESULT, "")).strip() == MISS:
        return True
    return has_where and not _blank(row.get(EXTRACTED)) and not row.get(WHERE)


def _keep(row, name, has_where):
    result = str(row.get(RESULT, "")).strip()
    if name == NEEDS_LOOK:
        return needs_look(row, has_where)
    if name == MATCHED:
        return result == MATCH
    if name == NO_RECORD:
        return result == NA
    return True


def filter_rows(rows, name, has_where):
    """The rows a filter pill shows. None - no pill selected - is All."""
    return [row for row in rows if _keep(row, name, has_where)]


def filter_counts(rows, has_where):
    return {name: len(filter_rows(rows, name, has_where)) for name in FILTERS}


# ------------------------------------------- badges: table row <-> page mark

def badges_by_field(pages):
    """{display name: [(page, number), ...]} from each page's legend.

    The legend is [(number, "Name · Name")] - one numbered highlight per line,
    listing every field read from it. Inverted, it says where each field's
    highlights are. Pages are numbered from 1."""
    index = {}
    for page_number, page in enumerate(pages, start=1):
        for number, names in page.get("legend", []):
            for name in str(names).split(" · "):
                index.setdefault(name.strip(), []).append((page_number, number))
    return index


def links_for_row(canonical_field, badge_index, aliases):
    """The ids of a row's highlights, "page:number" - what the document pane
    puts on each clickable highlight. Always page-qualified, unlike the badge
    labels, since they are matched rather than read."""
    links = []
    for alias in aliases.get(canonical_field, [canonical_field]):
        for page, number in badge_index.get(alias, []):
            link = f"{page}:{number}"
            if link not in links:
                links.append(link)
    return links


def where_for_row(canonical_field, badge_index, page_count, aliases):
    """The badge labels for one table row: every highlight of every engine
    field that answers it ("Attn: Negotiator(s) / Witness" is answered by the
    negotiator OR the witness). "4" on a one-page PDF, "p2·4" when there is
    more than one page and the number alone is ambiguous. Each label once."""
    labels = []
    for alias in aliases.get(canonical_field, [canonical_field]):
        for page, number in badge_index.get(alias, []):
            label = f"p{page}·{number}" if page_count > 1 else str(number)
            if label not in labels:
                labels.append(label)
    return labels
