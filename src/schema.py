"""Output-contract validation: JSON Schema (src/schema.json) + semantic invariants."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).with_name("schema.json")
SCHEMA_VERSION = "1.0"


class OutputValidationError(ValueError):
    """Raised when a pipeline output violates the contract."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _check_bbox(b: list[float], where: str, errors: list[str]) -> None:
    if b[0] > b[2] or b[1] > b[3]:
        errors.append(f"{where}: bbox not ordered x0<=x1, y0<=y1: {b}")


def semantic_errors(doc: dict[str, Any]) -> list[str]:
    """Invariants JSON Schema cannot express. Returns a list of human-readable problems."""
    errors: list[str] = []
    for p in doc.get("pages", []):
        tag = f"page {p['page']}"
        if p["needs_ocr"] != (p["route"] == "raster"):
            errors.append(f"{tag}: needs_ocr must equal (route == 'raster')")
        if p["route"] == "raster" and p["words"]:
            errors.append(f"{tag}: raster pages must not carry text-layer words")
        for w in p["words"]:
            _check_bbox(w["bbox"], f"{tag} word '{w['text']}'", errors)
        ids = [r["id"] for r in p["regions"]]
        if len(ids) != len(set(ids)):
            errors.append(f"{tag}: duplicate region ids")
        for r in p["regions"]:
            rtag = f"{tag} region {r['id']}"
            if (r["text_source"] == "needs_ocr") != (r["text"] is None):
                errors.append(f"{rtag}: text must be null iff text_source == 'needs_ocr'")
            if r["table"] is not None:
                t = r["table"]
                if r["class"] != "schedule_table":
                    errors.append(f"{rtag}: only schedule_table regions may carry a table")
                if len(t["headers"]) != len(t["canonical_headers"]):
                    errors.append(f"{rtag}: headers and canonical_headers differ in length")
                for i, row in enumerate(t["rows"]):
                    for c in row["cells"]:
                        _check_bbox(c["bbox"], f"{rtag} row {i}", errors)
                        if (c["text_source"] == "needs_ocr") != (c["text"] is None):
                            errors.append(f"{rtag} row {i}: cell text must be null iff needs_ocr")
    return errors


def validate_output(doc: dict[str, Any]) -> None:
    """Validate ``doc`` against the schema and semantic invariants; raise on failure."""
    schema_errs = sorted(_validator().iter_errors(doc), key=lambda e: list(e.absolute_path))
    msgs = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in schema_errs]
    if not msgs:
        msgs = semantic_errors(doc)
    if msgs:
        raise OutputValidationError("; ".join(msgs[:20]))
