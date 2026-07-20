#!/usr/bin/env python3
"""Restore the SILAS Graphiti graph from a backup JSON dump.

The consolidation pass and the manual backups write rotating JSON dumps to
~/graphiti-backups/. This script rebuilds the live graph from one of them —
the recovery path for a bad consolidation, a corruption event, or a regretted
merge. Faithfully restores node labels (so types survive) and converts temporal
fields back to real datetimes (so the bi-temporal queries keep working).

SAFE: lists backups by default and changes nothing. --apply wipes the current
group and restores; it dumps the CURRENT state to a *-pre-restore.json first, so
even a restore is reversible.

  restore.py                      # list available backups
  restore.py --apply              # restore the most recent backup
  restore.py --apply --file PATH  # restore a specific backup
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase  # noqa: E402

HOME = Path.home()
BACKUP_DIR = HOME / "graphiti-backups"
TEMPORAL_KEYS = {"created_at", "valid_at", "invalid_at", "expired_at", "last_recalled_at"}


def _load_dotenv():
    for cand in (HOME / ".hermes" / ".env", Path.cwd() / ".env"):
        if cand.is_file():
            for line in cand.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(k, d=""):
    return os.environ.get(k, d)


def _driver():
    return GraphDatabase.driver(
        _env("GRAPHITI_NEO4J_URI", "bolt://127.0.0.1:8687"),
        auth=(_env("GRAPHITI_NEO4J_USER", "neo4j"), _env("GRAPHITI_NEO4J_PASSWORD") or _env("NEO4J_PASSWORD", "")))


def _gid():
    return _env("GRAPHITI_GROUP_ID", "silas")


def _dump(drv, suffix: str) -> str:
    g = _gid()
    out = {"entities": [], "episodics": [], "relates_to": [], "mentions": []}
    with drv.session() as s:
        out["entities"] = [r.data()["n"] for r in s.run("MATCH (n:Entity {group_id:$g}) RETURN properties(n) AS n", g=g)]
        out["episodics"] = [r.data()["n"] for r in s.run("MATCH (n:Episodic {group_id:$g}) RETURN properties(n) AS n", g=g)]
        out["relates_to"] = [r.data() for r in s.run("MATCH (a:Entity)-[e:RELATES_TO {group_id:$g}]->(b:Entity) RETURN a.uuid AS a,b.uuid AS b,properties(e) AS e", g=g)]
        out["mentions"] = [r.data() for r in s.run("MATCH (a)-[m:MENTIONS]->(b:Entity {group_id:$g}) RETURN a.uuid AS a,b.uuid AS b,properties(m) AS m", g=g)]
    BACKUP_DIR.mkdir(exist_ok=True)
    path = BACKUP_DIR / f"silas-graph-{datetime.now():%Y%m%d-%H%M%S}-{suffix}.json"
    path.write_text(json.dumps(out, default=str))
    return str(path)


def _split_temporal(props: dict) -> tuple[dict, dict]:
    plain = {k: v for k, v in props.items() if k not in TEMPORAL_KEYS}
    temporal = {k: v for k, v in props.items() if k in TEMPORAL_KEYS and v}
    return plain, temporal


def _create_node(s, labels: list[str], props: dict):
    plain, temporal = _split_temporal(props)
    s.run("CALL apoc.create.node($labels, $props) YIELD node RETURN node", labels=labels, props=plain)
    for k, v in temporal.items():
        try:
            s.run(f"MATCH (n {{uuid:$u}}) SET n.`{k}` = datetime($v)", u=plain.get("uuid"), v=v)
        except Exception:
            pass  # leave as-is if not a parseable datetime


def _create_edge(s, rtype: str, a_uuid: str, b_uuid: str, props: dict):
    plain, temporal = _split_temporal(props)
    s.run(f"MATCH (a {{uuid:$a}}), (b {{uuid:$b}}) CREATE (a)-[e:{rtype}]->(b) SET e = $props",
          a=a_uuid, b=b_uuid, props=plain)
    euid = plain.get("uuid")
    if euid:
        for k, v in temporal.items():
            try:
                s.run(f"MATCH ()-[e:{rtype} {{uuid:$u}}]->() SET e.`{k}` = datetime($v)", u=euid, v=v)
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="perform the restore (default: just list backups)")
    ap.add_argument("--file", help="specific backup file to restore (default: most recent)")
    args = ap.parse_args()
    _load_dotenv()
    if not (_env("GRAPHITI_NEO4J_PASSWORD") or _env("NEO4J_PASSWORD")):
        print("ERROR: no Neo4j password in env", file=sys.stderr)
        return 2

    backups = sorted(BACKUP_DIR.glob("silas-graph-*.json"), reverse=True)
    if not args.apply:
        print(f"Available backups in {BACKUP_DIR} (newest first):")
        for b in backups[:20]:
            try:
                d = json.loads(b.read_text())
                print(f"  {b.name}  —  {len(d.get('entities', []))} entities, {len(d.get('relates_to', []))} facts")
            except Exception:
                print(f"  {b.name}  —  (unreadable)")
        print("\nRun with --apply [--file PATH] to restore (current state is backed up first).")
        return 0

    src = Path(args.file) if args.file else (backups[0] if backups else None)
    if not src or not src.is_file():
        print("ERROR: no backup file to restore from", file=sys.stderr)
        return 2
    data = json.loads(src.read_text())

    drv = _driver()
    g = _gid()
    pre = _dump(drv, "pre-restore")
    print(f"Backed up CURRENT state → {pre}")
    print(f"Restoring from {src.name} "
          f"({len(data['entities'])} entities, {len(data['relates_to'])} facts)…")

    with drv.session() as s:
        s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=g)
        for ent in data["entities"]:
            labels = ent.get("labels") or ["Entity"]
            if "Entity" not in labels:
                labels = ["Entity"] + list(labels)
            _create_node(s, list(labels), ent)
        for ep in data["episodics"]:
            _create_node(s, ["Episodic"], ep)
        for rt in data["relates_to"]:
            _create_edge(s, "RELATES_TO", rt["a"], rt["b"], rt["e"])
        for mn in data["mentions"]:
            _create_edge(s, "MENTIONS", mn["a"], mn["b"], mn["m"])
        ents = s.run("MATCH (n:Entity {group_id:$g}) RETURN count(n) AS c", g=g).single()["c"]
        facts = s.run("MATCH ()-[e:RELATES_TO {group_id:$g}]->() RETURN count(e) AS c", g=g).single()["c"]
    drv.close()
    print(f"Restored: {ents} entities, {facts} facts. "
          f"(Communities are rebuilt by the next consolidation pass.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
