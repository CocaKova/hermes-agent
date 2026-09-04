"""Relationship (edge) ontology for the SILAS Graphiti knowledge graph.

Node typing tells the graph *what* each thing is; edge typing tells it *how*
things relate — turning prose facts ("Jonny lives in Round Rock") into
structured, queryable relationships (Jonny -[LivesIn]-> Round Rock). That lets
SILAS reason over relations ("what does Jonny PREFER?", "what RUNS on the
Spark?") instead of string-matching fact text.

Gated by the GRAPHITI_EDGE_TYPES env flag (default OFF): edge typing adds work
to every extraction, so it must be validated against the local extractor before
being switched on. Field-less models — we want the relation label, not
attributes.
"""

from __future__ import annotations

from pydantic import BaseModel


class LivesIn(BaseModel):
    """The subject resides at / is based in the object location. GOOD: a person
    lives in a city or neighborhood. BAD: a one-time visit, a temporary trip."""


class WorksOn(BaseModel):
    """The subject is building, developing, or actively working on the object
    (a project, software, or piece of work). GOOD: Jonny works on Sovereign Spire."""


class Runs(BaseModel):
    """The subject runs, hosts, or operates the object software/service. GOOD:
    SILAS runs ComfyUI; the Spark runs vLLM. BAD: merely mentioning the software."""


class Uses(BaseModel):
    """The subject uses or depends on the object (a tool, service, library).
    GOOD: graphiti-viz uses D3; Jonny uses GitHub. BAD: builds it (WorksOn)."""


class Prefers(BaseModel):
    """The subject likes, favors, or prefers the object (a preference/taste).
    GOOD: Jonny prefers dark mode / local models. BAD: a neutral fact."""


class Attended(BaseModel):
    """The subject attended or will attend the object event. GOOD: Jonny
    attended game night. BAD: an ongoing state, a place with no event."""


class Owns(BaseModel):
    """The subject owns or possesses the object (a device, vehicle, account).
    GOOD: Jonny owns a DGX Spark. BAD: merely uses something he doesn't own."""


class MemberOf(BaseModel):
    """The subject is part of / belongs to the object (an organization, group,
    or larger system). GOOD: a component is part of a project."""


class LocatedAt(BaseModel):
    """The object is physically located at / hosted at the subject location or
    address. GOOD: ComfyUI is reachable at a host/port; a venue at an address."""


class Has(BaseModel):
    """The subject has the object as a component, attribute, or part — the
    general fallback when no more specific relation fits. GOOD: a project has a
    config file; a system has a memory graph."""


# name → model (field-less; docstring guides the extractor)
EDGE_TYPES: dict[str, type[BaseModel]] = {
    "LivesIn": LivesIn, "WorksOn": WorksOn, "Runs": Runs, "Uses": Uses,
    "Prefers": Prefers, "Attended": Attended, "Owns": Owns,
    "MemberOf": MemberOf, "LocatedAt": LocatedAt, "Has": Has,
}

# Which edge types may connect which node-type pairs. Every SILAS entity carries
# the 'Entity' label, so a permissive catch-all lets the extractor choose the
# best-fitting relation between any two entities.
EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Entity", "Entity"): list(EDGE_TYPES.keys()),
}
