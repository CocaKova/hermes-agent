"""Self-extending entity-type ontology for the SILAS Graphiti knowledge graph.

Gives the memory a real sense of *what kind of thing* each entity is — like a
brain's semantic categories. Unlike a hardcoded type list, the ontology is
DATA: it lives in ``~/.hermes/silas_ontology.json`` (which survives Hermes
reinstalls) and SILAS can grow it at runtime via the ``graphiti_define_type``
tool when it meets a kind of thing no existing category fits.

Graphiti classifies each extracted entity using the type NAME + its DOCSTRING,
then stamps the type as a Neo4j label (``:Entity:Place``). So each JSON entry's
``description`` becomes the model docstring — write them precise, with GOOD/BAD
examples, the style the local extractor is tuned on.

Models are field-less: we want classification, not attribute extraction.

Single-sourced — imported by __init__.py (live capture + the define-type tool),
the backfill/consolidation tooling, and the viz color logic.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, create_model

logger = logging.getLogger(__name__)

ONTOLOGY_PATH = Path.home() / ".hermes" / "silas_ontology.json"
MAX_TYPES = 15
DUP_SIMILARITY_THRESHOLD = 0.76  # definition cosine above this ⇒ reject as near-duplicate
                                 # (calibrated on BGE-M3: synonyms ~0.78-0.83, new types ~0.55-0.65)
_NAME_RE = re.compile(r"^[A-Z][A-Za-z]{2,19}$")  # CamelCase-ish, 3–20 alpha chars

# Fallback if the JSON is missing/corrupt — capture must never break.
_FALLBACK = {
    "Person": "A specific, named human being with a name, role, or relationship to the user.",
    "Place": "A physical, geographic location you can go to or be at (restaurant, city, venue).",
    "Organization": "A company, institution, or service provider (Tesla, Google, a bank).",
    "Software": "A digital tool, app, service, library, AI model, or AI assistant.",
    "Project": "An initiative or build the user is personally creating or driving.",
    "Device": "A piece of physical computing hardware (DGX Spark, a GPU, a server).",
    "Event": "A time-bound occurrence that happens then is over (a reservation, a meeting).",
}


def _load_raw() -> list[dict[str, Any]]:
    """Return the list of type dicts from the JSON file (or fallback)."""
    try:
        data = json.loads(ONTOLOGY_PATH.read_text())
        types = data.get("types") or []
        if types:
            return types
    except Exception as e:  # missing / malformed → fallback, never crash capture
        logger.warning("ontology load failed (%s); using fallback types", e)
    return [{"name": n, "description": d, "added_by": "seed",
             "created_at": str(date.today()), "usage_count": 0} for n, d in _FALLBACK.items()]


def _build_models(raw: list[dict[str, Any]]) -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for t in raw:
        name = str(t.get("name", "")).strip()
        desc = str(t.get("description", "")).strip()
        if not name or not desc:
            continue
        m = create_model(name, __base__=BaseModel)
        m.__doc__ = desc
        models[name] = m
    return models


# Live state, (re)built from disk.
_RAW: list[dict[str, Any]] = _load_raw()
ENTITY_TYPES: dict[str, type[BaseModel]] = _build_models(_RAW)


def reload() -> dict[str, type[BaseModel]]:
    """Re-read the JSON and rebuild ENTITY_TYPES in place. Returns the new map."""
    global _RAW, ENTITY_TYPES
    _RAW = _load_raw()
    ENTITY_TYPES.clear()
    ENTITY_TYPES.update(_build_models(_RAW))
    return ENTITY_TYPES


def type_names() -> list[str]:
    return list(ENTITY_TYPES.keys())


def describe(name: str) -> str:
    m = ENTITY_TYPES.get(name)
    return (m.__doc__ or "") if m else ""


# ---------------------------------------------------------------------------
# Self-extension — validation + near-duplicate detection + append.
# `embed_fn(text)->list[float]` is injected by the caller (the plugin already
# has a BGE-M3 embedder); if None, the duplicate check is skipped (name-only).
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def validate_new_type(name: str, description: str, embed_fn=None) -> tuple[bool, str]:
    """Check a proposed (name, description) against the guardrails.
    Returns (ok, reason). Does NOT write."""
    name = (name or "").strip()
    description = (description or "").strip()
    if not _NAME_RE.match(name):
        return False, f"name '{name}' must be CamelCase, 3–20 letters (e.g. 'Animal')"
    if name in ENTITY_TYPES:
        return False, f"type '{name}' already exists"
    if len(description) < 40:
        return False, "description too short — give a precise definition with GOOD/BAD examples (≥40 chars)"
    if len(ENTITY_TYPES) >= MAX_TYPES:
        return False, f"ontology is at the {MAX_TYPES}-type cap — consolidate/remove an unused type first"
    # near-duplicate check against existing types. Compare DEFINITIONS only (the
    # text before the GOOD/BAD examples) — embedding the full examples washes out
    # the semantic difference and makes synonyms look no closer than novel types.
    def _defn(d: str) -> str:
        return d.split("GOOD")[0].strip() or d
    if embed_fn is not None:
        try:
            cand = embed_fn(f"{name}: {_defn(description)}")
            for ex_name in ENTITY_TYPES:
                ex_vec = embed_fn(f"{ex_name}: {_defn(describe(ex_name))}")
                sim = _cosine(cand, ex_vec)
                if sim >= DUP_SIMILARITY_THRESHOLD:
                    return False, f"too similar to existing type '{ex_name}' (cosine {sim:.2f}) — reuse it instead"
        except Exception as e:
            logger.debug("dup-check embedding failed, skipping: %s", e)
    return True, "ok"


def add_type(name: str, description: str, added_by: str = "silas", embed_fn=None) -> tuple[bool, str]:
    """Validate then append a new type to the JSON and reload(). Returns (ok, message)."""
    ok, reason = validate_new_type(name, description, embed_fn=embed_fn)
    if not ok:
        return False, reason
    try:
        data = json.loads(ONTOLOGY_PATH.read_text())
    except Exception:
        data = {"types": _RAW}
    data.setdefault("types", [])
    data["types"].append({
        "name": name.strip(), "description": description.strip(),
        "added_by": added_by, "created_at": str(date.today()), "usage_count": 0,
    })
    ONTOLOGY_PATH.write_text(json.dumps(data, indent=2))
    reload()
    logger.info("ontology: added type '%s' (by %s); now %d types", name, added_by, len(ENTITY_TYPES))
    return True, f"Added entity type '{name}'. The graph now recognizes {len(ENTITY_TYPES)} categories."
