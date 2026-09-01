"""Pydantic mirror of the extraction schema, for constrained decoding.

Ollama accepts a JSON Schema in a request's "format" instead of the plain
string "json". Passing one makes the shape of the reply a constraint on
generation rather than a request in the prompt - the model cannot emit a key
that is not in the schema, or a value of the wrong type.

The models here are GENERATED from extractor.get_field_schema() and
get_group_schema() rather than hand-written next to them. Two definitions
would drift, and a drifted field name fails silently: the flattener, the CRM
comparison and the ground-truth scorer all key on the exact display name, so
a rename yields empty rows rather than an error.

This is an alternative path, not a replacement. The prompt-only path stays
exactly as it was so the two can be measured against each other - see
extractor.run(schema_mode=...). Whether constraining the shape helps or hurts
this model is an open question: an earlier attempt at the same goal by a
different mechanism (stating the type in the prompt line) cost seven fields,
so nothing here is assumed to be an improvement until it is measured.
"""

import re
from typing import List, Optional

from pydantic import BaseModel, Field, create_model

import extractor as E


def safe_name(display_name):
    """A Python identifier for a display name like "Other Deposit (if any)".

    The display names carry dots, slashes, spaces and parentheses, none of
    which are legal in an attribute name - so each generated field gets this
    as its attribute and the display name as its alias. Only the alias ever
    reaches the model or the rest of the pipeline."""
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(display_name)).strip("_").lower()
    if not name:
        name = "field"
    if name[0].isdigit():
        name = f"f_{name}"
    return name


class Leaf(BaseModel):
    """One extracted value, in the same shape the prompt-only path asks for."""
    value: Optional[str] = Field(default=None, description="The extracted value, or null if absent")
    confidence: Optional[float] = Field(default=None, description="0.0-1.0 confidence in this value")
    source_index: Optional[int] = Field(
        default=None,
        description="The [N] index of the OCR line backing a printed value, or null",
    )


def _leaf_fields(fields):
    """{attribute: (type, FieldInfo)} for create_model, one Leaf per field,
    aliased to the display name and carrying the schema's own description."""
    built = {}
    for f in fields:
        built[safe_name(f["field"])] = (
            Optional[Leaf],
            Field(default=None, alias=f["field"], description=f["description"]),
        )
    return built


def build_extraction_model():
    """The {fields, groups} model matching what extract_vision_grounded asks
    for, with every group nested the way get_group_schema() declares it -
    a list for a "multiple" role, a single object otherwise."""
    flat = create_model("FlatFields", **_leaf_fields(E.get_field_schema()))

    group_attrs = {}
    for name, spec in E.get_group_schema().items():
        group_model = create_model(f"{name.capitalize()}Group", **_leaf_fields(spec["fields"]))
        annotation = List[group_model] if spec["multiple"] else Optional[group_model]
        group_attrs[safe_name(name)] = (
            annotation,
            Field(default=None, alias=name, description=spec["description"]),
        )
    groups = create_model("Groups", **group_attrs)

    return create_model(
        "Extraction",
        fields=(flat, Field(description="Every top-level field")),
        groups=(groups, Field(description="Every group of related fields")),
    )


def extraction_json_schema():
    """The JSON Schema to hand Ollama as "format". Keyed by display name, so
    the reply parses with the same code the prompt-only path uses."""
    return build_extraction_model().model_json_schema(by_alias=True)


def resolve(root, node):
    """Follows a local "$ref" into the schema's "$defs".

    Pydantic factors repeated shapes (every Leaf, each group model) into
    $defs and references them, so a caller walking the schema hits a bare
    {"$ref": ...} rather than the object it describes."""
    seen = 0
    while isinstance(node, dict) and "$ref" in node and seen < 10:
        ref = node["$ref"]
        if not ref.startswith("#/$defs/"):
            break
        node = root.get("$defs", {}).get(ref.split("/")[-1], {})
        seen += 1
    # An Optional[...] field renders as anyOf[<thing>, null]; the caller wants
    # the thing.
    if isinstance(node, dict) and "anyOf" in node and "type" not in node:
        for option in node["anyOf"]:
            if option.get("type") != "null":
                return resolve(root, option)
    return node
