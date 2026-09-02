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

Owning the markup makes the header, the row striping and the match marks a
property of this file instead of a property of Streamlit's stylesheet.
"""

import html

# The three states of the Match column. Defined here and used by app.py when it
# builds a row, so the CSV and this table can't drift apart on what a tick is.
MATCH = "✔️"
MISS = "✖️"
NA = "—"

# What each state renders as: a glyph, and the class that colours it. Kept as
# plain characters rather than the emoji above - an emoji ignores CSS colour on
# most platforms, so a wrong row would print in the same ink as a right one.
_MARKS = {
    MATCH: ("✓", "ok"),
    MISS: ("✗", "bad"),
    NA: ("—", "na"),
}

COLUMNS = ("Field", "Ground Truth (API)", "Extracted", "Match")


def _cell(value):
    """One value as escaped HTML, with a blank rendered as the em dash the
    rest of the table uses rather than an empty cell that reads as a bug."""
    text = "" if value is None else str(value).strip()
    if not text:
        text = NA
    return html.escape(text)


def _mark(value):
    """The Match column: a coloured glyph when recognised, otherwise the value
    escaped as-is - an unknown state should show, not vanish."""
    glyph, css_class = _MARKS.get(str(value).strip(), (None, None))
    if glyph is None:
        return f'<span class="mark">{_cell(value)}</span>'
    return f'<span class="mark {css_class}">{glyph}</span>'


def comparison_table_html(rows, columns=COLUMNS):
    """The full <table> for the given rows.

    `rows` are the dicts app.py already builds, keyed by the names in COLUMNS.
    A row missing a key renders as the em dash rather than raising: a partial
    row is still worth showing.
    """
    head = "".join(f"<th>{html.escape(str(name))}</th>" for name in columns)

    body = []
    for row in rows:
        cells = []
        for name in columns:
            value = row.get(name)
            if name == "Match":
                cells.append(f'<td class="match-col">{_mark(value)}</td>')
            elif name == "Field":
                cells.append(f'<th scope="row">{_cell(value)}</th>')
            else:
                cells.append(f"<td>{_cell(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    return (
        '<div class="comparison-wrap"><table class="comparison">'
        f"<thead><tr>{head}</tr></thead>"
        f'<tbody>{"".join(body)}</tbody>'
        "</table></div>"
    )
