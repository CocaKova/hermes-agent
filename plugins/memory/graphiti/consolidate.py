#!/usr/bin/env python3
"""SILAS memory consolidation — the nightly "dream" pass.

Mirrors how a brain consolidates during sleep: replay recent memory, merge
duplicates, distill episodic specifics into durable gist, let weak/unused
traces fade, build higher-order structure, and tidy up. Runs deterministically
so the brain is never in the critical destructive path.

Steps & safety:
  1. BACKUP            rotating JSON dump of the whole graph            (always)
  2. DEDUP            coreferent entities (User/Jonny/cocakova) → merge AUTO
  3. DISTILL          event-fact clusters → one durable residue        AUTO
  4. DECAY            old + never-recalled + unpinned facts → outdated  AUTO (reversible)
  5. COMMUNITIES      pure-Python Louvain on entity graph (no LLM)     AUTO
  5.5 SUMMARIZE       regenerate entity summaries from live facts only  AUTO
  6. RETYPE           classify still-untyped entities into the ontology AUTO
  7. TYPE-MAINTENANCE refresh usage_count; flag unused silas types      AUTO
  8. HEALTH           append metrics snapshot to graphiti_health.log    AUTO

Modes:
  (default)         DRY-RUN — compute + print everything, change nothing.
  --apply           run ALL steps fully autonomously; send Matrix summary
                    of what was done. Anomaly guard pauses and alerts if
                    something looks catastrophically wrong (>30% drop).

Connection/creds from ~/.hermes/.env (GRAPHITI_*, MATRIX_*), same as the plugin.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid as _uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ontology as O  # noqa: E402

from neo4j import GraphDatabase  # noqa: E402
from openai import OpenAI  # noqa: E402

HOME = Path.home()
PROPOSALS_PATH = HOME / ".hermes" / "memory_proposals.json"
HEALTH_LOG = HOME / ".hermes" / "graphiti_health.log"
BACKUP_DIR = HOME / "graphiti-backups"
KEEP_BACKUPS = 14
DECAY_DAYS = 60


def _load_dotenv() -> None:
    for cand in (HOME / ".hermes" / ".env", Path.cwd() / ".env"):
        if not cand.is_file():
            continue
        for line in cand.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(k, d=""):
    return os.environ.get(k, d)


# ── LLM (reasoning-field fallback for the local thinking model) ──────────────
_client: OpenAI | None = None
def _thinking_extras(model: str, think: bool) -> dict:
    """Model-aware thinking knobs (SILAS patch, silas_ext/reapply.py).

    Mistral-native vLLM tokenizers (--tokenizer-mode mistral) reject any request
    carrying chat_template_kwargs with HTTP 400; their dial is the top-level
    reasoning_effort param (only none/high accepted). Qwen-style templates use
    chat_template_kwargs.enable_thinking. Mirrors _thinking_client.py's guard."""
    normalized = (model or "").strip().lower()
    if normalized.startswith("mistral") or "/mistral" in normalized:
        return {"reasoning_effort": "high" if think else "none"}
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": think}}}


def _llm(prompt: str, max_tokens: int = 4000, think: bool = False) -> str:
    """Call the local brain. With think=True the 27B reasons first (best for
    judgment tasks like coreference); the qwen reasoning parser puts the CoT in
    `reasoning` and the final answer in `content` — but degrades to dumping
    everything in `reasoning`, so we fall back to it. Timeout-bounded so a
    stalled/overloaded model degrades gracefully. Returns '' on failure."""
    global _client
    if _client is None:
        _client = OpenAI(base_url=_env("GRAPHITI_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
                         api_key="x", timeout=300, max_retries=1)
    try:
        model = _env("GRAPHITI_LLM_MODEL", "qwen3.6-27b")
        resp = _client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 if think else 0.0, max_tokens=max_tokens,
            **_thinking_extras(model, think))
        m = resp.choices[0].message
        return (m.content or "").strip() or (getattr(m, "reasoning", "") or "").strip()
    except Exception as e:
        print(f"  [llm] call failed/timed out: {e}", file=sys.stderr)
        return ""


def _json_block(txt: str):
    """Extract the LAST valid JSON array/object in the text. Robust to a thinking
    model that reasons (with stray brackets) before emitting its final answer."""
    if not txt:
        return None
    dec = json.JSONDecoder()
    best, best_len = None, 0
    for i, ch in enumerate(txt):
        if ch not in "[{":
            continue
        try:
            obj, end = dec.raw_decode(txt[i:])
        except Exception:
            continue
        if isinstance(obj, (list, dict)) and end > best_len:  # largest valid JSON = the real answer
            best, best_len = obj, end
    return best


# ── Graph helpers ────────────────────────────────────────────────────────────
def _driver():
    return GraphDatabase.driver(
        _env("GRAPHITI_NEO4J_URI", "bolt://127.0.0.1:8687"),
        auth=(_env("GRAPHITI_NEO4J_USER", "neo4j"), _env("GRAPHITI_NEO4J_PASSWORD") or _env("NEO4J_PASSWORD", "")))


def _gid():
    return _env("GRAPHITI_GROUP_ID", "silas")


# ── Step 1: backup ───────────────────────────────────────────────────────────
def backup(drv) -> str:
    BACKUP_DIR.mkdir(exist_ok=True)
    g = _gid()
    out = {"entities": [], "episodics": [], "relates_to": [], "mentions": []}
    with drv.session() as s:
        out["entities"] = [r.data()["n"] for r in s.run("MATCH (n:Entity {group_id:$g}) RETURN properties(n) AS n", g=g)]
        out["episodics"] = [r.data()["n"] for r in s.run("MATCH (n:Episodic {group_id:$g}) RETURN properties(n) AS n", g=g)]
        out["relates_to"] = [r.data() for r in s.run("MATCH (a:Entity)-[e:RELATES_TO {group_id:$g}]->(b:Entity) RETURN a.uuid AS a,b.uuid AS b,properties(e) AS e", g=g)]
        out["mentions"] = [r.data() for r in s.run("MATCH (a)-[m:MENTIONS]->(b:Entity {group_id:$g}) RETURN a.uuid AS a,b.uuid AS b,properties(m) AS m", g=g)]
    path = BACKUP_DIR / f"silas-graph-{datetime.now():%Y%m%d-%H%M%S}-consolidate.json"
    path.write_text(json.dumps(out, default=str))
    # rotate
    backups = sorted(BACKUP_DIR.glob("*-consolidate.json"))
    for old in backups[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return str(path)


# ── Step 2: dedup → propose ──────────────────────────────────────────────────
# The local extractor is unreliable at open-ended coreference (conservative →
# misses real dups; pushed harder → hallucinates cross-type merges like
# person→restaurant). So dedup is defense-in-depth:
#   (a) deterministic owner-identity merge from ~/.hermes/silas_identity.json
#       (the one high-value case that needs world knowledge);
#   (b) a conservative LLM pass for spelling/spacing variants, HARD-GUARDED so a
#       proposed group can never span two different entity types.
IDENTITY_PATH = HOME / ".hermes" / "silas_identity.json"


def _load_identity() -> dict:
    try:
        return json.loads(IDENTITY_PATH.read_text())
    except Exception:
        return {}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _name_variant(a: str, b: str) -> bool:
    """True only if a and b are string-variants of the SAME token (spelling/
    spacing/casing/abbreviation), NOT merely related words."""
    from difflib import SequenceMatcher
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb or na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= 0.82


def _mk_proposal(canon, members, by_name, require_similar=False) -> dict | None:
    names = [n for n in members if n in by_name]
    if len(names) < 2:
        return None
    # HARD type-safety guard: never merge across distinct entity types.
    types = {by_name[n]["type"] for n in names if by_name[n]["type"]}
    if len(types) > 1:
        return None
    if canon not in by_name:
        canon = max(names, key=lambda n: by_name[n]["degree"])
    dups = [n for n in names if n != canon]
    # For the LLM "variant" pass: every duplicate must be a genuine string-variant
    # of the canonical (kills hallucinated merges like React→Vite that share a type).
    if require_similar:
        dups = [n for n in dups if _name_variant(n, canon)]
        if not dups:
            return None
    return {"kind": "merge", "canonical": canon, "canonical_uuid": by_name[canon]["uuid"],
            "duplicates": [{"name": n, "uuid": by_name[n]["uuid"]} for n in dups]}


# Scalable dedup: cheap embedding blocking narrows the whole graph to a few small
# candidate clusters; only those reach the (reasoning-on) adjudicator. This is why
# it survives a big graph — we never reason over all N entities at once.
BLOCK_COSINE = 0.72   # name-embedding similarity to flag a merge CANDIDATE
MAX_CLUSTER = 8       # ignore pathologically large clusters (bad blocking, not dups)


def _block_candidates(ents: list[dict]) -> list[list[str]]:
    """O(n²) name-embedding blocking (numpy) → candidate clusters via union-find.
    Only above-threshold, type-compatible pairs link; returns small clusters."""
    import numpy as np
    rows = [e for e in ents if e.get("emb")]
    if len(rows) < 2:
        return []
    names = [e["name"] for e in rows]
    by = {e["name"]: e for e in ents}
    M = np.asarray([e["emb"] for e in rows], dtype=float)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    S = M @ M.T
    iu = np.triu_indices(len(rows), k=1)
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(*[a[S[iu] >= BLOCK_COSINE] for a in iu]):
        a, b = names[int(i)], names[int(j)]
        ta, tb = by[a]["type"], by[b]["type"]
        if ta and tb and ta != tb:           # cross-type can't be the same thing
            continue
        parent[find(a)] = find(b)
    clusters: dict[str, list[str]] = defaultdict(list)
    for n in names:
        clusters[find(n)].append(n)
    return [c for c in clusters.values() if 2 <= len(c) <= MAX_CLUSTER]


def _adjudicate(members: list[str], by_name: dict) -> list[dict]:
    """Reasoning-ON adjudication of ONE small candidate cluster. Small in, small
    out — so thinking is cheap here even though it'd be hopeless over the whole
    graph. Returns merge groups [{canonical,names,why}]."""
    lines = "\n".join(f"- {n} ({by_name[n]['type'] or 'untyped'}): {by_name[n]['summary']}" for n in members)
    prompt = (
        "These entities from one person's memory graph scored as similar. Decide which (if any) "
        "are the SAME real-world thing — true aliases or spelling/spacing variants — versus merely "
        "related or distinct (a tool and its plugin, a service and a sub-feature are NOT the same). "
        "Be conservative: only merge if genuinely identical. Use ONLY the exact entity names listed "
        "below — never invent or abbreviate a name.\n\n"
        f"{lines}\n\n"
        'Reason briefly, then output ONLY JSON {"merges":[{"canonical":"..","names":["..",".."],'
        '"why":".."}]} — empty merges if none are truly the same.')
    data = _json_block(_llm(prompt, max_tokens=1000, think=True))
    if isinstance(data, dict):
        return data.get("merges") or []
    # Guard: list may contain strings if the model returned a name-list instead of merge-dicts
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def find_duplicates(drv) -> list[dict]:
    g = _gid()
    with drv.session() as s:
        ents = [r.data() for r in s.run(
            "MATCH (n:Entity {group_id:$g}) OPTIONAL MATCH (n)-[r:RELATES_TO]-() "
            "RETURN n.uuid AS uuid, n.name AS name, [l IN labels(n) WHERE l<>'Entity'][0] AS type, "
            "left(coalesce(n.summary,''),100) AS summary, count(r) AS degree, "
            "n.name_embedding AS emb ORDER BY n.name", g=g)]
    if len(ents) < 2:
        return []
    by_name = {e["name"]: e for e in ents}
    lower = {e["name"].lower(): e["name"] for e in ents}
    proposals: list[dict] = []
    used: set[str] = set()

    # (a) deterministic owner-identity merge (name-DISSIMILAR aliases blocking can't
    #     surface — User/Jonny/cocakova; comes from ~/.hermes/silas_identity.json)
    ident = _load_identity()
    aliases = [lower[a.lower()] for a in ident.get("owner_aliases", []) if a.lower() in lower]
    if len(aliases) >= 2:
        canon = ident.get("owner_canonical") if ident.get("owner_canonical") in by_name else \
            max(aliases, key=lambda n: by_name[n]["degree"])
        p = _mk_proposal(canon, aliases, by_name)
        if p:
            p["why"] = "configured owner identity"
            proposals.append(p)
            used.update(aliases)

    # (b) embedding blocking → small candidate clusters. Obvious same-type string
    #     variants auto-merge; ambiguous clusters get reasoning-on adjudication.
    for cluster in _block_candidates(ents):
        members = [n for n in cluster if n not in used]
        if len(members) < 2:
            continue
        if (len(members) == 2 and _name_variant(members[0], members[1])
                and by_name[members[0]]["type"] == by_name[members[1]]["type"]):
            p = _mk_proposal(None, members, by_name)
            if p:
                p["why"] = "spelling/spacing variant"
                proposals.append(p)
                used.update(members)
                used.add(p["canonical"])
            continue
        for grp in _adjudicate(members, by_name):
            if not isinstance(grp, dict):
                continue
            gm = [n for n in grp.get("names", []) if n in by_name and n not in used]
            p = _mk_proposal(grp.get("canonical"), gm, by_name)
            if p:
                p["why"] = grp.get("why", "")
                proposals.append(p)
                used.update(d["name"] for d in p["duplicates"])
                used.add(p["canonical"])
    return proposals


def apply_merge(drv, prop: dict) -> None:
    """Merge each duplicate node INTO the canonical via APOC, which repoints ALL
    relationships (RELATES_TO + MENTIONS) with their properties intact and deletes
    the duplicate. mergeRels=false keeps every fact-edge (no combine = no fact
    loss); harmless parallel/self edges get tidied by cleanup_graph's dup step."""
    canon = prop["canonical_uuid"]
    with drv.session() as s:
        for dup in prop["duplicates"]:
            s.run("""
                MATCH (c:Entity {uuid:$canon}), (d:Entity {uuid:$du})
                CALL apoc.refactor.mergeNodes([c, d], {mergeRels:false, properties:'discard'})
                YIELD node RETURN node
                """, canon=canon, du=dup["uuid"])


# ── Step 3: distillation (event-fact clusters) → propose ─────────────────────
def find_distillations(drv) -> list[dict]:
    g = _gid()
    with drv.session() as s:
        facts = [r.data() for r in s.run(
            "MATCH (a:Entity)-[e:RELATES_TO {group_id:$g}]->(b:Entity) "
            "WHERE e.invalid_at IS NULL AND e.expired_at IS NULL "
            "RETURN e.uuid AS uuid, a.uuid AS sub, a.name AS sub_name, e.fact AS fact", g=g)]
    clusters = defaultdict(list)
    for f in facts:
        prefix = " ".join((f["fact"] or "").split()[:2]).lower()  # e.g. "silas recommends"
        clusters[(f["sub"], prefix)].append(f)
    out = []
    for (sub, prefix), group in clusters.items():
        if len(group) >= 4:
            out.append({"kind": "distill", "subject": group[0]["sub_name"], "pattern": prefix,
                       "count": len(group), "fact_uuids": [x["uuid"] for x in group],
                       "facts": [x["fact"] for x in group]})
    return out


def apply_distill(drv, prop: dict) -> None:
    g = _gid()
    joined = "; ".join(prop["facts"])
    residue = _llm(
        "Summarize these related facts into ONE durable, general fact (no transient specifics). "
        f"Return ONLY the sentence.\nFacts: {joined}", max_tokens=200, think=True).strip().strip('"').splitlines()[0]
    if not residue:
        return
    with drv.session() as s:
        # store residue as a new edge subject->subject (self), confidence 0.8, pinned false
        s.run("""
            MATCH (a:Entity {name:$sn, group_id:$g}) WITH a LIMIT 1
            CREATE (a)-[:RELATES_TO {uuid:$u, group_id:$g, fact:$f, created_at:datetime(),
                    confidence:0.8, name:'CONSOLIDATED'}]->(a)
            """, sn=prop["subject"], g=g, u=str(_uuid.uuid4()), f=residue)
        # expire the specifics (kept as history, not deleted)
        s.run("MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $u "
              "SET e.invalid_at=datetime(), e.expired_at=datetime()", u=prop["fact_uuids"])


# ── Step 4: salience decay → AUTO (reversible: marks outdated, never deletes) ─
def decay(drv, apply: bool) -> int:
    g = _gid()
    q = (f"MATCH ()-[e:RELATES_TO {{group_id:$g}}]->() "
         f"WHERE e.invalid_at IS NULL AND e.expired_at IS NULL "
         f"AND coalesce(e.pinned,false)=false AND coalesce(e.recall_count,0)=0 "
         f"AND e.created_at < datetime() - duration({{days:{DECAY_DAYS}}})")
    with drv.session() as s:
        n = s.run(q + " RETURN count(e) AS c", g=g).single()["c"]
        if apply and n:
            s.run(q + " SET e.invalid_at=datetime(), e.expired_at=datetime()", g=g)
    return n


# ── Step 4.5: node-level salience prune → AUTO (garbage-collects dead nodes) ──
def prune_dead_nodes(drv, apply: bool) -> int:
    """Delete Entity nodes with no LIVE fact-edges left — the node-level companion
    to edge decay. An entity whose every RELATES_TO edge is invalid/expired (or
    that has none) contributes nothing to recall; it is pure graph bloat (e.g. an
    'Euboea' node left over after its one essay-derived fact decayed). Guards:
    never touch a node with any live OR pinned edge, and never touch a node newer
    than the decay window. Communities are untouched (only :Entity)."""
    g = _gid()
    q = (f"MATCH (n:Entity {{group_id:$g}}) "
         f"WHERE coalesce(n.pinned,false)=false "
         f"AND n.created_at < datetime() - duration({{days:{DECAY_DAYS}}}) "
         f"AND NOT EXISTS {{ MATCH (n)-[e:RELATES_TO]-() "
         f"WHERE (e.invalid_at IS NULL AND e.expired_at IS NULL) "
         f"OR coalesce(e.pinned,false)=true }}")
    with drv.session() as s:
        n = s.run(q + " RETURN count(n) AS c", g=g).single()["c"]
        if apply and n:
            s.run(q + " DETACH DELETE n", g=g)
    return n


# ── Step 6: re-type — untyped AUTO + re-examine MISTYPED → propose ────────────
def retype(drv, apply: bool) -> dict:
    """Re-classify entities during the dream. Untyped nodes get a type
    auto-applied (safe gain). Already-typed nodes the classifier strongly
    disagrees with become PROPOSED corrections — we don't silently overturn a
    prior typing decision (which the user may have fixed by hand)."""
    g = _gid()
    with drv.session() as s:
        ents = [r.data() for r in s.run(
            "MATCH (n:Entity {group_id:$g}) "
            "RETURN n.uuid AS uuid, n.name AS name, "
            "[l IN labels(n) WHERE l<>'Entity'][0] AS cur, "
            "left(coalesce(n.summary,''),120) AS summary", g=g)]
    if not ents:
        return {"applied": {}, "corrections": []}
    ont = "\n".join(f"- {n}: {O.describe(n).split('GOOD')[0].strip()}" for n in O.type_names())
    listing = "\n".join(f"{i+1}. {e['name']} — {e['summary']}" for i, e in enumerate(ents))
    mapping = _json_block(_llm(
        f"Classify each into ONE type, or 'Entity' if none fit / it's generic noise.\n{ont}\n\n"
        f"Entities:\n{listing}\n\nReturn ONLY a JSON object name->type.", think=True)) or {}
    valid = set(O.type_names())
    applied, corrections = {}, []
    with drv.session() as s:
        for e in ents:
            t = mapping.get(e["name"])
            if t not in valid:
                continue
            cur = e["cur"]
            if not cur:                       # untyped → auto-apply
                applied[e["name"]] = t
                if apply:
                    s.run(f"MATCH (n:Entity {{uuid:$u}}) SET n:{t}, n.labels=['Entity',$t]", u=e["uuid"], t=t)
            elif cur != t:                    # mistyped → propose a correction
                corrections.append({"name": e["name"], "uuid": e["uuid"], "from": cur, "to": t})
    return {"applied": applied, "corrections": corrections}


def apply_type_correction(drv, c: dict) -> None:
    with drv.session() as s:
        s.run(f"MATCH (n:Entity {{uuid:$u}}) REMOVE n:`{c['from']}` SET n:`{c['to']}`, n.labels=['Entity',$t]",
              u=c["uuid"], t=c["to"])


# ── Step 5: communities — pure-Python Louvain, no LLM, no deadlock risk ──────
def build_communities(drv, apply: bool) -> str:
    """Detect topic clusters via Louvain on the entity RELATES_TO graph.
    Stores community_id on each entity node; creates lightweight Community
    nodes so the health snapshot counter works. No LLM involved — the old
    Graphiti build_communities deadlocked on the local model."""
    if not apply:
        return "skipped (dry-run)"
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        g = _gid()
        with drv.session() as s:
            ents = [r["uuid"] for r in s.run("MATCH (n:Entity {group_id:$g}) RETURN n.uuid AS uuid", g=g)]
            edges = [(r["a"], r["b"]) for r in s.run(
                "MATCH (a:Entity {group_id:$g})-[e:RELATES_TO]->(b:Entity {group_id:$g}) "
                "WHERE e.invalid_at IS NULL RETURN a.uuid AS a, b.uuid AS b", g=g)]
        if len(ents) < 3:
            return "skipped (too few entities)"
        G = nx.Graph()
        G.add_nodes_from(ents)
        G.add_edges_from(edges)
        # undirected Louvain; seed for determinism
        communities = louvain_communities(G, seed=42)
        with drv.session() as s:
            # clear old community assignments and nodes
            s.run("MATCH (n:Community) DETACH DELETE n")
            s.run("MATCH (n:Entity {group_id:$g}) REMOVE n.community_id", g=g)
            for i, members in enumerate(communities):
                cid = f"{g}_comm_{i}"
                s.run("CREATE (:Community {id:$cid, group_id:$g, size:$sz})",
                      cid=cid, g=g, sz=len(members))
                s.run("MATCH (n:Entity {group_id:$g}) WHERE n.uuid IN $uuids SET n.community_id=$cid",
                      g=g, uuids=list(members), cid=cid)
        sizes = sorted([len(c) for c in communities], reverse=True)
        return f"{len(communities)} clusters (largest: {sizes[0]})"
    except Exception as e:
        return f"failed: {e}"


# ── Step 5.5: summarize key entities from live facts ──────────────────────────
def summarize_entities(drv, apply: bool) -> int:
    """Regenerate the `summary` field for Jonny and any entity with ≥3 live
    facts, from ONLY current live facts.  This is the fix for the core bug:
    the old rolling summary never shrinks when corrupt/expired facts are deleted,
    so hallucinated content (Nisha, Denver, GMC) stays baked in forever.

    Returns the number of nodes updated."""
    g = _gid()
    identity = _load_identity()
    owner = identity.get("owner_canonical", "Jonny")

    with drv.session() as s:
        # Entities to refresh: the owner node always; others with ≥3 live facts
        candidates = [r["name"] for r in s.run("""
            MATCH (n:Entity {group_id:$g})
            OPTIONAL MATCH (n)-[e:RELATES_TO {group_id:$g}]->(m)
            WHERE e.invalid_at IS NULL AND e.expired_at IS NULL
            WITH n, count(e) AS live
            WHERE n.name = $owner OR live >= 3
            RETURN n.name AS name
        """, g=g, owner=owner)]

    updated = 0
    for name in candidates:
        with drv.session() as s:
            rows = s.run("""
                MATCH (n:Entity {group_id:$g, name:$name})
                OPTIONAL MATCH (n)-[e:RELATES_TO {group_id:$g}]->(m)
                WHERE e.invalid_at IS NULL AND e.expired_at IS NULL
                OPTIONAL MATCH (p)-[e2:RELATES_TO {group_id:$g}]->(n)
                WHERE e2.invalid_at IS NULL AND e2.expired_at IS NULL
                RETURN collect(DISTINCT e.fact) + collect(DISTINCT e2.fact) AS facts
            """, g=g, name=name).single()
        if not rows:
            continue
        facts = [f for f in (rows["facts"] or []) if f]
        if not facts:
            continue
        joined = "\n".join(f"- {f}" for f in facts[:30])
        new_summary = _llm(
            f"Write a concise factual summary of '{name}' from ONLY these current facts. "
            f"2-5 sentences. Include only what is explicitly stated — no inferences, no invented details.\n\n"
            f"Facts:\n{joined}\n\nReturn ONLY the summary text.",
            max_tokens=300,
        ).strip().strip('"')
        if not new_summary or len(new_summary) < 20:
            continue
        if apply:
            with drv.session() as s:
                s.run("MATCH (n:Entity {group_id:$g, name:$name}) SET n.summary = $s",
                      g=g, name=name, s=new_summary)
        updated += 1
        print(f"   {'updated' if apply else 'would update'}: {name} ({len(facts)} facts)")
    return updated


def type_maintenance(drv, apply: bool) -> dict:
    g = _gid()
    with drv.session() as s:
        counts = {r["t"]: r["c"] for r in s.run(
            "MATCH (n:Entity {group_id:$g}) UNWIND [l IN labels(n) WHERE l<>'Entity'] AS t "
            "RETURN t AS t, count(*) AS c", g=g)}
    if apply:
        try:
            data = json.loads(O.ONTOLOGY_PATH.read_text())
            for t in data.get("types", []):
                t["usage_count"] = counts.get(t["name"], 0)
            O.ONTOLOGY_PATH.write_text(json.dumps(data, indent=2))
        except Exception:
            pass
    return counts


def health_snapshot(drv, apply: bool, extra: dict) -> dict:
    g = _gid()
    with drv.session() as s:
        ent = s.run("MATCH (n:Entity {group_id:$g}) RETURN count(n) AS c", g=g).single()["c"]
        typed = s.run("MATCH (n:Entity {group_id:$g}) WHERE size([l IN labels(n) WHERE l<>'Entity'])>0 RETURN count(n) AS c", g=g).single()["c"]
        orph = s.run("MATCH (n:Entity {group_id:$g}) WHERE NOT (n)-[:RELATES_TO]-() RETURN count(n) AS c", g=g).single()["c"]
        cur = s.run("MATCH ()-[e:RELATES_TO {group_id:$g}]->() WHERE e.invalid_at IS NULL AND e.expired_at IS NULL RETURN count(e) AS c", g=g).single()["c"]
        out = s.run("MATCH ()-[e:RELATES_TO {group_id:$g}]->() WHERE e.invalid_at IS NOT NULL OR e.expired_at IS NOT NULL RETURN count(e) AS c", g=g).single()["c"]
        comm = s.run("MATCH (n:Community) RETURN count(n) AS c").single()["c"]
    snap = {"ts": datetime.now(timezone.utc).isoformat(), "entities": ent,
            "typed_pct": round(100 * typed / ent, 1) if ent else 0,
            "orphans": orph, "current_facts": cur, "outdated_facts": out,
            "communities": comm, "active_types": len(O.type_names()), **extra}
    if apply:
        with HEALTH_LOG.open("a") as f:
            f.write(json.dumps(snap) + "\n")
    return snap


# ── Health anomaly detection (#3 alert + #1 safety guard) ────────────────────
ANOMALY_DROP_FRAC = 0.30          # >30% drop in entities / current_facts vs last snapshot
ANOMALY_TYPED_DROP = 25.0         # typed_pct fell this many points
ANOMALY_ORPHAN_RISE_FRAC = 0.30   # orphans rose >30% relative to prev snapshot (spike, not drift)
ANOMALY_ORPHAN_ENTITY_FRAC = 0.50 # OR orphans now exceed 50% of total entities (runaway)


def _last_health() -> dict | None:
    try:
        lines = [l for l in HEALTH_LOG.read_text().splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception:
        return None


def _health_anomalies(prev: dict | None, cur: dict) -> list[str]:
    """Compare current graph metrics to the last snapshot; return human-readable
    anomalies that suggest corruption/data-loss (a smoke alarm)."""
    if not prev:
        return []
    out = []
    for key, label in (("entities", "entities"), ("current_facts", "current facts")):
        p, c = prev.get(key, 0), cur.get(key, 0)
        if p and c < p * (1 - ANOMALY_DROP_FRAC):
            out.append(f"{label} dropped {p}→{c} (>{int(ANOMALY_DROP_FRAC*100)}%)")
    pt, ct = prev.get("typed_pct", 0), cur.get("typed_pct", 0)
    if pt and ct < pt - ANOMALY_TYPED_DROP:
        out.append(f"typed-share fell {pt}%→{ct}%")
    # Orphan check: ignore day-to-day drift; only fire on a sudden spike (>30% relative
    # rise vs previous run) or if orphans become the majority of the graph (>50%).
    prev_orph, cur_orph = prev.get("orphans", 0), cur.get("orphans", 0)
    cur_ents = max(cur.get("entities", 1), 1)
    orphan_spike = prev_orph > 0 and cur_orph > prev_orph * (1 + ANOMALY_ORPHAN_RISE_FRAC)
    orphan_majority = cur_orph / cur_ents >= ANOMALY_ORPHAN_ENTITY_FRAC
    if orphan_spike or orphan_majority:
        pct = round(100 * cur_orph / cur_ents)
        out.append(f"orphans spiked {prev_orph}→{cur_orph} ({pct}% of graph)")
    return out


# ── Dream journal — what SILAS reads to recount his "dream" in the morning ───
DREAM_JOURNAL = HOME / ".hermes" / "last_dream.md"


def write_dream_journal(snap: dict, *, applied: dict, n_decay: int, communities: str,
                        merges: list, distills: list, corrections: list,
                        n_summarized: int = 0, anomalies: list | None = None) -> None:
    ts = (snap.get("ts", "") or "")[:16].replace("T", " ")
    L = [f"# Last memory consolidation — {ts} UTC\n"]
    if anomalies:
        L.append(f"⚠ I held off consolidating — something looked wrong: {'; '.join(anomalies)}. "
                 "I changed nothing, to avoid making it worse. Worth a look — `restore.py` can roll "
                 "back to a backup if memory was lost.")
        DREAM_JOURNAL.write_text("\n".join(L))
        return
    L.append("What I tidied autonomously:")
    L.append(f"- Faded {n_decay} old, never-recalled fact(s) to 'outdated'.")
    if merges:
        L.append(f"- Merged {len(merges)} duplicate(s): " +
                 "; ".join(f"{[d['name'] for d in m['duplicates']]} → {m['canonical']}" for m in merges) + ".")
    if distills:
        L.append(f"- Distilled {len(distills)} redundant fact cluster(s): " +
                 "; ".join(f"{d['subject']} ({d['count']}×→1)" for d in distills) + ".")
    if applied:
        L.append(f"- Typed/re-typed {len(applied)} entit(ies): "
                 + ", ".join(f"{k}→{v}" for k, v in list(applied.items())[:6]) + ".")
    if n_summarized:
        L.append(f"- Refreshed {n_summarized} entity summary(ies) from live facts only.")
    L.append(f"- Topic clusters: {communities}.")
    L.append(f"- Graph: {snap.get('entities')} entities ({snap.get('typed_pct')}% typed), "
             f"{snap.get('current_facts')} live facts, {snap.get('outdated_facts')} faded.")
    L.append("\nNothing needs your attention — all changes applied automatically.")
    DREAM_JOURNAL.write_text("\n".join(L))


# ── Matrix notify ────────────────────────────────────────────────────────────
def notify_matrix(text: str) -> bool:
    hs, tok, room = _env("MATRIX_HOMESERVER"), _env("MATRIX_ACCESS_TOKEN"), _env("MATRIX_HOME_ROOM")
    if not (hs and tok and room):
        return False
    try:
        import requests
        txn = _uuid.uuid4().hex
        r = requests.put(f"{hs.rstrip('/')}/_matrix/client/v3/rooms/{room}/send/m.room.message/{txn}",
                         headers={"Authorization": f"Bearer {tok}"},
                         json={"msgtype": "m.text", "body": text}, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


# ── Orchestration ────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="run AUTO steps + write proposals + notify")
    ap.add_argument("--apply-pending", action="store_true", help="execute approved merges/distillations from the proposals file")
    args = ap.parse_args()
    _load_dotenv()
    if not (_env("GRAPHITI_NEO4J_PASSWORD") or _env("NEO4J_PASSWORD")):
        print("ERROR: no Neo4j password in env", file=sys.stderr)
        return 2
    drv = _driver()

    # ── apply-pending: execute approved destructive proposals, then stop ──
    if args.apply_pending:
        if not PROPOSALS_PATH.is_file():
            print("No pending proposals.")
            return 0
        props = json.loads(PROPOSALS_PATH.read_text())
        # SAFETY guard: refuse a merge set that would delete an unreasonable share
        # of the graph (a sign the proposals are bad / the graph changed).
        with drv.session() as s:
            cur_ents = s.run("MATCH (n:Entity {group_id:$g}) RETURN count(n) AS c", g=_gid()).single()["c"]
        to_delete = sum(len(p.get("duplicates", [])) for p in props.get("merges", []))
        if cur_ents and to_delete > cur_ents * ANOMALY_DROP_FRAC:
            msg = (f"Refusing apply-pending: the {len(props.get('merges', []))} merge(s) would delete "
                   f"{to_delete} of {cur_ents} entities (>{int(ANOMALY_DROP_FRAC*100)}%). Review proposals first.")
            print("⚠ " + msg)
            notify_matrix("⚠️ " + msg)
            drv.close()
            return 1
        bk = backup(drv)
        print(f"backup: {bk}")
        for p in props.get("merges", []):
            apply_merge(drv, p)
            print(f"merged {[d['name'] for d in p['duplicates']]} → {p['canonical']}")
        for p in props.get("distillations", []):
            apply_distill(drv, p)
            print(f"distilled {p['count']} '{p['pattern']}' facts for {p['subject']}")
        for c in props.get("type_corrections", []):
            apply_type_correction(drv, c)
            print(f"re-typed {c['name']}: {c['from']} → {c['to']}")
        PROPOSALS_PATH.unlink(missing_ok=True)
        print("Applied pending proposals; proposals file cleared.")
        drv.close()
        return 0

    apply = args.apply
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"=== Consolidation [{mode}] group={_gid()} ===\n")

    if apply:
        print(f"1. BACKUP        {backup(drv)}")
    else:
        print("1. BACKUP        (skipped in dry-run)")

    # ── SAFETY: compare to the last health snapshot. A big unexplained drop
    #    means likely corruption/data-loss — ALERT and skip every step so the
    #    pass can't make it worse. (smoke alarm + #1 guard) ──
    prev_snap = _last_health()
    early = health_snapshot(drv, False, {})
    anomalies = _health_anomalies(prev_snap, early)
    if anomalies:
        print(f"\n⚠ ANOMALY GUARD: {'; '.join(anomalies)}")
        if apply:
            prev_line = (f"Before: {prev_snap['entities']} entities · "
                         f"{prev_snap['current_facts']} live facts · "
                         f"{prev_snap['orphans']} orphans"
                         if prev_snap else "Before: no prior snapshot")
            now_line = (f"Now:    {early['entities']} entities · "
                        f"{early['current_facts']} live facts · "
                        f"{early['orphans']} orphans")
            notify_matrix(
                f"⚠️ Memory consolidation paused — something looked wrong.\n"
                f"Issue: {'; '.join(anomalies)}\n"
                f"{prev_line}\n"
                f"{now_line}\n"
                f"Nothing was changed. If this is real data loss: restore.py --apply\n"
                f"If it's a false alarm: re-run consolidate.py --apply")
            health_snapshot(drv, True, {"anomaly": "; ".join(anomalies)})
            write_dream_journal(early, applied={}, n_decay=0, communities="skipped",
                                merges=[], distills=[], corrections=[], anomalies=anomalies)
        print("→ all destructive steps skipped; run restore.py if this is data loss.")
        drv.close()
        return 0

    merges = find_duplicates(drv)
    print(f"\n2. DEDUP — {len(merges)} merge group(s):")
    for m in merges:
        print(f"   {[d['name'] for d in m['duplicates']]} → {m['canonical']}  ({m.get('why','')})")
    if apply:
        for m in merges:
            apply_merge(drv, m)
            print(f"   ✓ merged {[d['name'] for d in m['duplicates']]} → {m['canonical']}")

    distills = find_distillations(drv)
    print(f"\n3. DISTILL — {len(distills)} cluster(s):")
    for d in distills:
        print(f"   {d['subject']}: {d['count']}× '{d['pattern']}…'")
    if apply:
        for d in distills:
            apply_distill(drv, d)
            print(f"   ✓ distilled {d['count']}× '{d['pattern']}' for {d['subject']}")

    n_decay = decay(drv, apply)
    print(f"\n4. DECAY ({'applied' if apply else 'would'}) — {n_decay} old/unrecalled fact(s) → outdated")

    n_pruned = prune_dead_nodes(drv, apply)
    print(f"\n4.5. PRUNE ({'applied' if apply else 'would'}) — {n_pruned} dead entity node(s) (no live edges) removed")

    comm_status = build_communities(drv, apply)
    print(f"\n5. COMMUNITIES   {comm_status}")

    n_summarized = summarize_entities(drv, apply)
    print(f"\n5.5. SUMMARIZE ({'applied' if apply else 'would'}) — {n_summarized} entity summary(ies) regenerated from live facts")

    rt = retype(drv, apply)
    auto_typed, corrections = rt["applied"], rt["corrections"]
    if apply:
        for c in corrections:
            apply_type_correction(drv, c)
            print(f"   ✓ re-typed {c['name']}: {c['from']} → {c['to']}")
    all_typed = {**auto_typed, **{c['name']: c['to'] for c in corrections}}
    print(f"\n6. RETYPE ({'applied' if apply else 'would'}) — {len(auto_typed)} untyped + {len(corrections)} mistyped: " +
          ", ".join(f"{k}→{v}" for k, v in list(all_typed.items())[:8]))

    counts = type_maintenance(drv, apply)
    print(f"\n7. TYPE USAGE    " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))

    snap = health_snapshot(drv, apply, {"merged": len(merges), "distilled": len(distills),
                                        "retyped": len(all_typed), "decayed": n_decay,
                                        "summaries_refreshed": n_summarized})
    print(f"\n8. HEALTH        {json.dumps(snap)}")

    PROPOSALS_PATH.unlink(missing_ok=True)  # no longer used; clean up if it exists

    if apply:
        write_dream_journal(snap, applied=all_typed, n_decay=n_decay, communities=comm_status,
                            merges=merges, distills=distills, corrections=[],
                            n_summarized=n_summarized)
        lines = ["🌙 Memory consolidation complete."]
        if merges:
            lines.append("• Merged " + ", ".join(
                f"{[d['name'] for d in m['duplicates']]} → {m['canonical']}" for m in merges[:4])
                + (f" (+{len(merges)-4} more)" if len(merges) > 4 else ""))
        if distills:
            lines.append("• Distilled " + ", ".join(
                f"{d['subject']} ({d['count']}×→1)" for d in distills[:4])
                + (f" (+{len(distills)-4} more)" if len(distills) > 4 else ""))
        if all_typed:
            n_t = len(all_typed)
            sample = ", ".join(f"{k}→{v}" for k, v in list(all_typed.items())[:3])
            lines.append(f"• Re-typed {n_t}: {sample}{'…' if n_t > 3 else ''}")
        if n_summarized:
            lines.append(f"• Refreshed {n_summarized} summaries")
        if n_decay:
            lines.append(f"• Faded {n_decay} old fact(s) to outdated")
        if n_pruned:
            lines.append(f"• Pruned {n_pruned} dead entity node(s)")
        # State line: show deltas vs before this run so it's easy to see what changed
        ent_delta = f" ({snap['entities'] - prev_snap['entities']:+d})" if prev_snap else ""
        fct_delta = f" ({snap['current_facts'] - prev_snap['current_facts']:+d})" if prev_snap else ""
        lines.append(
            f"• {snap['entities']} entities{ent_delta} · "
            f"{snap['current_facts']} live / {snap['outdated_facts']} faded facts{fct_delta} · "
            f"{snap['orphans']} orphans · "
            f"{snap['typed_pct']}% typed · {comm_status}")
        sent = notify_matrix("\n".join(lines))
        print(f"\n→ Matrix notify: {'sent' if sent else 'unavailable'}")
    else:
        print("\nDRY-RUN — nothing changed.")

    drv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
