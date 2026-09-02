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
        "Landlord Name(s)", "Landlord NRIC/Passport No.",
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
