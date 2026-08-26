"""Authoritative ground truth for a sale, fetched from PropNex's internal
sales API and mapped to the flat field names the extractor produces, so
accuracy can be checked against the CRM record rather than eyeballed.

The API's schema only covers a subset of what's printed on the LOF form (no
signature dates, no furnished status, no rental/deposit breakdown), so
FORM_FIELD_MAPS below maps only the fields that genuinely have an API
counterpart. Fields with no mapping simply can't be validated this way.

Merged here from two modules in the original multi-project layout
(api_ground_truth.py + compare.py's value matchers) so this project has a
single, self-contained ground-truth dependency.
"""

import os
import re
from datetime import datetime

import requests
from dateutil import parser as dateutil_parser

# Overridable so the endpoint isn't baked into the source. Set
# LOF2DATA_API_URL to point at a different environment.
DEFAULT_BASE_URL = os.environ.get(
    "LOF2DATA_API_URL", "https://staging.mypropnex.com/api/sales/details"
)


def fetch_sale(sale_id, password, base_url=DEFAULT_BASE_URL):
    response = requests.get(base_url, params={"sale_id": sale_id, "password": password}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != 1:
        raise RuntimeError(f"API error for sale_id={sale_id}: {payload.get('msg')}")
    return payload["data"]["sale"]


def _tenant_client(sale):
    tenants = sale.get("amla_purpose", {}).get("purchasers_or_tenants") or []
    return tenants[0] if tenants else {}


def _landlord_parties(sale):
    return sale.get("amla_purpose", {}).get("vendors_or_landlords") or []


def _earliest_deposit(sale):
    deposits = sale.get("deposits") or []
    if not deposits:
        return {}
    return min(deposits, key=lambda d: d.get("deposit_date") or "9999-99-99")


def _property_address_parts(sale):
    p = sale.get("property", {})
    parts = [p.get("unit"), p.get("road"), p.get("building"), p.get("zip_code")]
    return ", ".join(part for part in parts if part)


# name -> function(sale_dict) -> raw value (str/None), per form template.
FORM_FIELD_MAPS = {
    "form1": {
        "Date": lambda s: s.get("date_of_lof"),
        "Reference No.": lambda s: s.get("lof_no"),
        # Form1 has no explicit "Attn: Negotiator(s):" field like Form2/3 -
        # the negotiator's name instead shows up as the "Witnessed by / Name"
        # signatory on this template's signature blocks.
        "Negotiator": lambda s: s.get("listing_negotiator", {}).get("name"),
        "Tenant(s)": lambda s: _tenant_client(s).get("name"),
        "Property Address": lambda s: s.get("property", {}).get("address"),
        "Description of Property (Selected)": lambda s: (s.get("property", {}).get("type") or "").title() or None,
        "Total": lambda s: s.get("total_deposit"),
        "Earnest Deposit Amount": lambda s: _earliest_deposit(s).get("amount"),
        "Landlord Name": lambda s: " / ".join(p["name"] for p in _landlord_parties(s)) or None,
        "Landlord NRIC/Passport No.": lambda s: " / ".join(p["nric"] for p in _landlord_parties(s)) or None,
    },
    "form2": {
        "Attn: Negotiator(s)": lambda s: s.get("listing_negotiator", {}).get("name"),
        "Date": lambda s: s.get("date_of_lof"),
        "Description of Property (Selected)": lambda s: (s.get("property", {}).get("type") or "").title() or None,
        "Property known as": _property_address_parts,
        "Total": lambda s: s.get("total_deposit"),
        "Earnest Deposit Amount": lambda s: _earliest_deposit(s).get("amount"),
        "Tenant Name": lambda s: _tenant_client(s).get("name"),
        "Landlord Name": lambda s: " / ".join(p["name"] for p in _landlord_parties(s)) or None,
        "Landlord NRIC No.": lambda s: " / ".join(p["nric"] for p in _landlord_parties(s)) or None,
        "Vacant Possession Date": lambda s: s.get("date_of_co"),
        "Tenancy Agreement Execution Date": lambda s: s.get("date_of_co"),
    },
    # Form3 is a different LOF template but the same underlying CRM sale
    # record, so it draws from the same API fields as form2.
    "form3": {
        "Attn: Negotiator(s)": lambda s: s.get("listing_negotiator", {}).get("name"),
        "Date": lambda s: s.get("date_of_lof"),
        "Description of Property (Selected)": lambda s: (s.get("property", {}).get("type") or "").title() or None,
        "Property known as": _property_address_parts,
        "Total": lambda s: s.get("total_deposit"),
        "Earnest Deposit Amount": lambda s: _earliest_deposit(s).get("amount"),
        "Tenant Name": lambda s: _tenant_client(s).get("name"),
        "Landlord Name": lambda s: " / ".join(p["name"] for p in _landlord_parties(s)) or None,
        "Landlord NRIC No.": lambda s: " / ".join(p["nric"] for p in _landlord_parties(s)) or None,
        "Vacant Possession Date": lambda s: s.get("date_of_co"),
        "Tenancy Agreement Execution Date": lambda s: s.get("date_of_co"),
    },
}

# How to compare each field - most fields are free text (handled by
# compare.py's values_match); dates and money amounts need format-aware
# comparison since the API and the form spell them out very differently
# ("2026-03-21" vs "1ST APRIL 2026"; "11320.00" vs "RM 11,320").
FIELD_KINDS = {
    "Date": "date",
    "Vacant Possession Date": "date",
    "Tenancy Agreement Execution Date": "date",
    "Total": "amount",
    "Earnest Deposit Amount": "amount",
    "Landlord NRIC/Passport No.": "id",
    "Landlord NRIC No.": "id",
    "Tenant(s)": "id",
    # ids_match strips all non-alphanumeric chars before comparing, which
    # also happens to fix reference numbers matching despite spacing
    # differences (e.g. "206537 (31/03)" vs "206537(31/03)").
    "Reference No.": "id",
}

# Fixed comparison checklist, independent of form template - it's the same
# CRM sale record regardless of which LOF template was used, so one API-side
# lambda per field covers all of them (unlike FORM_FIELD_MAPS above, which
# exists because each template's EXTRACTED field names differ, not because
# the API itself differs per template).
#
# Furnish Status / Security / Utilities / Others / Advance Rental have no
# API equivalent at all - the API only carries an aggregate total_deposit,
# no itemized breakdown - so these always resolve to None. The row still
# displays (see app.py): extracted value shown, but marked "not verifiable"
# rather than a false mismatch.
CANONICAL_FIELDS = [
    "Reference No.",
    "Attn: Negotiator(s) / Witness",
    "Date of LOF",
    "Description of Property (Selected)",
    "Property known as",
    "Furnish Status",
    "Security Deposit",
    "Utilities Deposit",
    "Advance Rental",
    "Others (if any)",
    "Earnest Deposit Amount",
    "Total",
    "Tenant Name",
    "Tenant IC",
    "Landlord Name",
    "Landlord IC",
]

CANONICAL_API_MAP = {
    "Reference No.": lambda s: s.get("lof_no"),
    "Attn: Negotiator(s) / Witness": lambda s: s.get("listing_negotiator", {}).get("name"),
    "Date of LOF": lambda s: s.get("date_of_lof"),
    "Description of Property (Selected)": lambda s: (s.get("property", {}).get("type") or "").title() or None,
    "Property known as": lambda s: s.get("property", {}).get("address"),
    "Furnish Status": lambda s: None,
    "Earnest Deposit Amount": lambda s: _earliest_deposit(s).get("amount"),
    "Security Deposit": lambda s: None,
    "Others (if any)": lambda s: None,
    "Utilities Deposit": lambda s: None,
    "Advance Rental": lambda s: None,
    "Total": lambda s: s.get("total_deposit"),
    "Tenant Name": lambda s: _tenant_client(s).get("name"),
    "Tenant IC": lambda s: _tenant_client(s).get("nric"),
    "Landlord Name": lambda s: " / ".join(p["name"] for p in _landlord_parties(s)) or None,
    "Landlord IC": lambda s: " / ".join(p["nric"] for p in _landlord_parties(s)) or None,
}

CANONICAL_FIELD_KINDS = {
    "Reference No.": "id",  # ids_match strips non-alphanumeric chars, tolerating spacing differences
    "Date of LOF": "date",
    "Earnest Deposit Amount": "amount",
    "Total": "amount",
    "Tenant IC": "id",
    "Landlord IC": "id",
    "Description of Property (Selected)": "choice",
    "Furnish Status": "choice",
}


def choice_match(a, b):
    """For "pick one of N" fields - unlike free text, values_match's
    substring check is actively wrong here: the raw unselected choice list
    (e.g. "House / Semi-D / Bungalow / Condo / ...") contains every single
    candidate as a substring, so ANY answer - including the wrong ones -
    would look like a match. Require exact equality (case/whitespace
    insensitive only) instead."""
    if a is None or b is None:
        return False
    na = re.sub(r"\s+", " ", str(a).strip().lower())
    nb = re.sub(r"\s+", " ", str(b).strip().lower())
    return bool(na) and na == nb


def build_canonical_ground_truth(sale):
    return {name: fn(sale) for name, fn in CANONICAL_API_MAP.items()}


def build_api_ground_truth(sale, form_key):
    field_map = FORM_FIELD_MAPS.get(form_key, {})
    return {name: fn(sale) for name, fn in field_map.items()}


def _to_float(text):
    if text is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(text))
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def amounts_match(a, b):
    fa, fb = _to_float(a), _to_float(b)
    return fa is not None and fb is not None and abs(fa - fb) < 0.01


def dates_match(api_value, extracted_value):
    """api_value is always the API's own ISO "YYYY-MM-DD" string - parsed
    strictly first so dayfirst never has a chance to misread it (dateutil's
    dayfirst flag, if applied to an already-unambiguous ISO string, silently
    swaps month/day). extracted_value is the human-written form date
    ("21/3/2026", "1ST APRIL 2026", ...), which genuinely is day-first.
    """
    if not api_value or not extracted_value:
        return False
    try:
        da = datetime.strptime(str(api_value), "%Y-%m-%d")
    except ValueError:
        try:
            da = dateutil_parser.parse(str(api_value), fuzzy=True)
        except (ValueError, OverflowError):
            return False
    try:
        db = dateutil_parser.parse(str(extracted_value), fuzzy=True, dayfirst=True)
    except (ValueError, OverflowError):
        return False
    return (da.year, da.month, da.day) == (db.year, db.month, db.day)


def ids_match(a, b):
    """Strips all non-alphanumeric characters before comparing - handles
    NRIC/registration numbers formatted with different dash/space/paren
    conventions between the CRM and the extracted text."""
    if a is None or b is None:
        return False
    na = re.sub(r"[^a-z0-9]", "", str(a).lower())
    nb = re.sub(r"[^a-z0-9]", "", str(b).lower())
    if not na or not nb:
        return False
    return na == nb or na in nb or nb in na


def _normalize(value):
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[.,]", "", value)  # ignore comma/period formatting differences
    return value


def values_match(a, b):
    na, nb = _normalize(a), _normalize(b)
    if not na and not nb:
        return True
    if not na or not nb:
        # Exactly one side is blank - not a match. (Without this guard,
        # Python's "" in "anything" == True was silently counting a missing
        # extraction as a correct match against any non-blank ground truth.)
        return False
    return na == nb or na in nb or nb in na
