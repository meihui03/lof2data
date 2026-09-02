"""Laying the engine's field list out for the Step 2 picker, and keeping
removed fields out of the comparison table.

Split out of app.py so it can be tested without executing the Streamlit
script. The engine owns WHAT the fields are (extractor.field_sources, read
over `--list-fields`); this module owns only how they are grouped for a
human to read, and what a removed field means downstream.
"""

# Presentation only - the engine has no notion of sections, and shouldn't:
# how a list reads on screen is not an extraction concern. Grouping matters
# here because 28 ungrouped checkboxes is a wall.
FIELD_SECTIONS = [
    ("Reference & Dates", [
        "Reference No.", "Date", "Tenancy Commencement Date",
        "Tenancy Expiration Date", "Period of Tenancy", "Option to Renew",
    ]),
    ("Property", [
        "Description of Property/Property Type", "Property Address",
        "Title Details", "Premise Usage", "Furnished Status",
    ]),
    ("Financial", [
        "Monthly Rental", "Security Deposit", "Utilities Deposit",
        "Advance Rental", "Other Deposit (if any)", "Total Payable",
        "Earnest Deposit Amount", "Legal Cost / Stamp Duty Allocation",
    ]),
    ("Parties", [
        "Landlord Name(s)", "Landlord NRIC/Passport No.", "Landlord Address",
        "Tenant Name(s)/Company", "Tenant NRIC/Passport No.",
        "Tenant Address", "Negotiator/Agent Name", "Witnessed by",
    ]),
    ("Terms", [
        "Special/Other Condition(s)", "Notice to Quit (months)",
    ]),
]


def sectioned_fields(all_fields):
    """[(section title, [field, ...]), ...] for the fields the engine offers.

    Driven by what the engine actually published rather than by the constant
    above, in both directions: a section whose fields have all gone is
    dropped, and a field this module has never heard of still appears, under
    "Other". Otherwise adding a field to the engine's schema would leave it
    extracting with no way to switch it off - a silent, invisible failure.
    """
    available = list(all_fields)
    grouped, placed = [], set()

    for title, names in FIELD_SECTIONS:
        present = [name for name in names if name in available]
        if present:
            grouped.append((title, present))
            placed.update(present)

    leftover = [name for name in available if name not in placed]
    if leftover:
        grouped.append(("Other", leftover))
    return grouped


def excluded_from(all_fields, keep_flags):
    """The fields to pass to the engine as --exclude, in the engine's own order.

    `keep_flags` is {field: bool} from the checkboxes. A field with no entry
    counts as kept: on the very first render no checkbox has been recorded
    yet, and defaulting the other way would silently extract nothing.
    """
    return [name for name in all_fields if not keep_flags.get(name, True)]


def picker_units(all_fields, alias_groups):
    """[(key, label, [field, ...]), ...] - what Step 2 offers as ONE tick.

    Most fields are their own unit. Fields that answer the SAME comparison
    row are one unit, because ticking one and not the other is never what
    someone means and quietly costs them the row.

    That is not hypothetical. "Attn: Negotiator(s) / Witness" is answered by
    the negotiator OR the witness, precisely because a form that carries no
    "Attn:" line still names the same person under "Witnessed by" - the
    extractor is told to return no agent rather than promote a witness into
    the role. On a document like that, keeping "Negotiator/Agent Name" while
    dropping "Witnessed by" left the row empty and scored it as a miss, when
    the answer was on the page and the pipeline was simply told not to read
    it. One tick for both makes that unreachable.

    `alias_groups` is the caller's alias map values (app.py's VLLM_ALIASES) -
    passed in rather than imported, since importing app.py runs the whole
    Streamlit script. Names in a group that the engine doesn't offer (a
    sibling extractor's "Purchaser Name(s)") are ignored, so they neither
    form nor grow a unit.
    """
    available = list(all_fields)
    order = {name: i for i, name in enumerate(available)}

    # field -> the full membership of its unit
    unit_of = {}
    for group in alias_groups or ():
        members = tuple(sorted({n for n in group if n in order}, key=order.get))
        if len(members) > 1:
            for name in members:
                # A field named in two alias groups keeps the first grouping
                # it was given, rather than silently merging three fields
                # into one tick.
                unit_of.setdefault(name, members)

    units, placed = [], set()
    for name in available:
        if name in placed:
            continue
        members = unit_of.get(name, (name,))
        placed.update(members)
        units.append((members[0], " · ".join(members), list(members)))
    return units


def sectioned_units(all_fields, alias_groups):
    """picker_units(), grouped into sections the way sectioned_fields groups
    plain names. A unit lands in the section of whichever member the sections
    mention first, and appears exactly once even if its members are listed in
    different sections."""
    units = picker_units(all_fields, alias_groups)
    unit_by_member = {member: unit for unit in units for member in unit[2]}

    grouped, placed = [], set()
    for title, names in sectioned_fields(all_fields):
        section = []
        for name in names:
            unit = unit_by_member.get(name)
            if unit is None or unit[0] in placed:
                continue
            placed.add(unit[0])
            section.append(unit)
        if section:
            grouped.append((title, section))
    return grouped


def excluded_from_units(all_fields, units, keep_flags):
    """The fields to pass to the engine as --exclude, given one flag per UNIT.

    A unit is kept or dropped whole: keeping it keeps every field in it, which
    is the entire point - see picker_units. A unit with no recorded flag counts
    as kept, for the same reason excluded_from does.
    """
    kept = set()
    for key, _, members in units:
        if keep_flags.get(key, True):
            kept.update(members)
    return [name for name in all_fields if name not in kept]


def extractable_aliases(aliases, all_fields):
    """The aliases in this canonical row that the engine can actually produce.

    A row's alias list is inherited from a sibling extractor's field names, so
    some entries ("Purchaser Name(s)", "Vendor Name") belong to a schema this
    engine doesn't have and can never be filled here. Left in, they would keep
    a row alive forever: row_is_hidden waits for EVERY alias to be excluded,
    and an alias the picker never offers can never be excluded.

    Falls back to the aliases as given when none of them is recognized, rather
    than returning nothing - an unrecognized row is one this can't reason
    about, and showing it is the safe way to be wrong.
    """
    available = set(all_fields or ())
    known = [alias for alias in aliases if alias in available]
    return known or list(aliases)


def row_is_hidden(aliases, excluded):
    """Should the comparison table skip this canonical field's row?

    The table walks CANONICAL_FIELDS, not the extraction result, so a removed
    field would otherwise still get a row - empty, and scored as a miss.
    Removing a field would then look exactly like breaking it, and would drag
    the accuracy figure down as if the extractor had failed.

    A canonical field can map to several engine field names, and the row is
    still answerable while any one of them is being extracted, so it hides
    only when every alias has gone.
    """
    return bool(aliases) and all(alias in excluded for alias in aliases)
