#!/usr/bin/env python3
"""One-time / periodic cleanup for the SILAS Graphiti knowledge graph.

Removes the three kinds of junk identified in the graph and uses the SAME
filter logic the live plugin applies at capture time (``_filters.py``), so a
cleanup pass and ongoing ingestion can never disagree about what counts as
junk:

  1. SECRETS    — redacts secret values out of Episodic node content/name and
                  deletes any fact-edge whose text contains a secret.
  2. TRANSIENT  — deletes fact-edges that are one-time inbox/email/calendar/
                  notification observations (not durable knowledge).
  3. DUPLICATES — collapses exact-duplicate fact-edges, keeping the newest.

DRY-RUN BY DEFAULT: prints exactly what it would change and touches nothing.
Pass --apply to perform the deletions/redactions.

Connection config is read from the same env vars as the plugin (loads
~/.hermes/.env if present):
  GRAPHITI_NEO4J_URI / _USER / _PASSWORD (or NEO4J_PASSWORD), GRAPHITI_GROUP_ID

Usage:
  venv/bin/python plugins/memory/graphiti/cleanup_graph.py            # dry run
  venv/bin/python plugins/memory/graphiti/cleanup_graph.py --apply    # execute
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# import the shared filters (same dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _filters as F  # noqa: E402

from neo4j import GraphDatabase  # noqa: E402


def _load_dotenv() -> None:
    """Best-effort load of ~/.hermes/.env so creds are available standalone."""
    for cand in (Path.home() / ".hermes" / ".env", Path.cwd() / ".env"):
        if not cand.is_file():
            continue
        for line in cand.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _config():
    return {
        "uri": os.environ.get("GRAPHITI_NEO4J_URI", "bolt://127.0.0.1:8687"),
        "user": os.environ.get("GRAPHITI_NEO4J_USER", "neo4j"),
        "password": os.environ.get("GRAPHITI_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", ""),
        "gid": os.environ.get("GRAPHITI_GROUP_ID", "silas"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry run)")
    ap.add_argument("--deep", action="store_true",
                    help="also prune dev/ops-noise facts, ephemeral menu recommendations, "
                         "and orphaned noise-name entities (filenames, endpoints, CLI tools)")
    args = ap.parse_args()

    _load_dotenv()
    cfg = _config()
    if not cfg["password"]:
        print("ERROR: no Neo4j password in env (GRAPHITI_NEO4J_PASSWORD / NEO4J_PASSWORD)", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"=== Graphiti cleanup [{mode}] group={cfg['gid']} {cfg['uri']} ===\n")

    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    gid = cfg["gid"]
    edge_deletes: set[str] = set()          # edge uuids to delete
    episode_redactions: list[dict] = []     # {uuid, name, content}
    episode_deletes: set[str] = set()       # Episodic node uuids to delete whole
    entity_deletes: set[str] = set()        # Entity node uuids to delete (orphan sweep)

    with driver.session() as s:
        edges = [r.data() for r in s.run(
            "MATCH ()-[e:RELATES_TO]->() WHERE e.group_id=$g "
            "RETURN e.uuid AS uuid, e.fact AS fact, toString(e.created_at) AS created_at", g=gid)]
        episodes = [r.data() for r in s.run(
            "MATCH (n:Episodic) WHERE n.group_id=$g "
            "RETURN n.uuid AS uuid, n.name AS name, n.content AS content", g=gid)]

        # 1. SECRETS ----------------------------------------------------------
        print("--- 1. SECRETS ---")
        for ep in episodes:
            new_content, c1 = F.redact_secrets(ep.get("content") or "")
            new_name, c2 = F.redact_secrets(ep.get("name") or "")
            if c1 or c2:
                episode_redactions.append({"uuid": ep["uuid"], "name": new_name, "content": new_content})
                print(f"  redact episode {ep['uuid'][:8]}  {ep.get('name','')[:60]!r}")
        sec_edges = [e for e in edges if F.contains_secret(e.get("fact") or "")]
        for e in sec_edges:
            edge_deletes.add(e["uuid"])
            print(f"  delete fact-edge (secret): {e['fact']!r}")
        if not episode_redactions and not sec_edges:
            print("  (none)")

        # 2. TRANSIENT --------------------------------------------------------
        print("\n--- 2. TRANSIENT OBSERVATIONS ---")
        trans = [e for e in edges if F.is_transient_fact(e.get("fact") or "")]
        for e in trans:
            edge_deletes.add(e["uuid"])
            print(f"  delete: {e['fact']!r}")
        if not trans:
            print("  (none)")

        # 3. META — the store describing itself / its own contents ------------
        print("\n--- 3. META-FACTS (memory talking about itself) ---")
        meta = [e for e in edges if F.is_meta_fact(e.get("fact") or "") and e["uuid"] not in edge_deletes]
        for e in meta:
            edge_deletes.add(e["uuid"])
            print(f"  delete: {e['fact']!r}")
        if not meta:
            print("  (none)")

        # 4. DUPLICATES -------------------------------------------------------
        print("\n--- 4. EXACT DUPLICATES (keep newest) ---")
        by_fact: dict[str, list[dict]] = defaultdict(list)
        for e in edges:
            if e["uuid"] in edge_deletes:   # already slated for deletion
                continue
            by_fact[(e.get("fact") or "").strip()].append(e)
        dup_count = 0
        for fact, group in by_fact.items():
            if not fact or len(group) < 2:
                continue
            group.sort(key=lambda x: x.get("created_at") or "", reverse=True)  # newest first
            for e in group[1:]:
                edge_deletes.add(e["uuid"])
                dup_count += 1
            print(f"  {len(group)}x -> keep 1, delete {len(group) - 1}: {fact!r}")
        if dup_count == 0:
            print("  (none)")

        # 5. NOISE EPISODES — harness/system notices & bare auth-URL pastes ---
        print("\n--- 5. NOISE EPISODES (harness notices / auth-URL pastes) ---")
        for ep in episodes:
            if F.is_noise_turn(ep.get("content") or "") or F.is_noise_turn(ep.get("name") or ""):
                episode_deletes.add(ep["uuid"])
                print(f"  delete episode {ep['uuid'][:8]}  {ep.get('name','')[:60]!r}")
        if not episode_deletes:
            print("  (none)")
        # don't bother redacting an episode we're deleting outright
        episode_redactions = [r for r in episode_redactions if r["uuid"] not in episode_deletes]

        # 6 + 7. DEEP — dev/ops noise, ephemeral recs, orphan noise entities ---
        if args.deep:
            print("\n--- 6. DEV/OPS NOISE + EPHEMERAL RECOMMENDATIONS ---")
            deep = [e for e in edges
                    if e["uuid"] not in edge_deletes
                    and (F.is_dev_noise_fact(e.get("fact") or "")
                         or F.is_ephemeral_recommendation(e.get("fact") or ""))]
            for e in deep:
                edge_deletes.add(e["uuid"])
                print(f"  delete: {e['fact']!r}")
            if not deep:
                print("  (none)")

            print("\n--- 7. ORPHAN NOISE ENTITIES (factless filenames/endpoints/tools) ---")
            orphan_ents = [r.data() for r in s.run(
                "MATCH (n:Entity) WHERE n.group_id=$g AND NOT (n)-[:RELATES_TO]-() "
                "RETURN n.uuid AS uuid, n.name AS name", g=gid)]
            for ent in orphan_ents:
                if F.is_noise_entity_name(ent.get("name") or ""):
                    entity_deletes.add(ent["uuid"])
                    print(f"  delete entity: {ent['name']!r}")
            if not entity_deletes:
                print("  (none)")

        # SUMMARY -------------------------------------------------------------
        print(f"\n=== SUMMARY: {len(edge_deletes)} fact-edge(s) to delete, "
              f"{len(episode_redactions)} episode(s) to redact, "
              f"{len(episode_deletes)} noise episode(s) to delete, "
              f"{len(entity_deletes)} orphan entity(ies) to delete ===")

        if not args.apply:
            print("\nDRY-RUN — nothing changed. Re-run with --apply to execute.")
            driver.close()
            return 0

        # APPLY ---------------------------------------------------------------
        for r in episode_redactions:
            s.run("MATCH (n:Episodic {uuid:$u}) SET n.content=$c, n.name=$nm",
                  u=r["uuid"], c=r["content"], nm=r["name"])
        if edge_deletes:
            s.run("MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $u DELETE e",
                  u=list(edge_deletes))
        if episode_deletes:
            s.run("MATCH (n:Episodic) WHERE n.uuid IN $u DETACH DELETE n",
                  u=list(episode_deletes))
        if entity_deletes:
            s.run("MATCH (n:Entity) WHERE n.uuid IN $u DETACH DELETE n",
                  u=list(entity_deletes))
        print(f"\nAPPLIED: redacted {len(episode_redactions)} episode(s), "
              f"deleted {len(edge_deletes)} fact-edge(s), "
              f"deleted {len(episode_deletes)} noise episode(s), "
              f"deleted {len(entity_deletes)} orphan entity(ies).")

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
