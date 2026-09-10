#!/usr/bin/env python3
"""Time-based fading for the SILAS Graphiti knowledge graph.

Graphiti is bi-temporal but only invalidates a fact when a *contradicting*
fact arrives. Event-bound facts ("User plans to dine at Fonda San Miguel
tonight", "the reservation is for tomorrow at 7pm") are never contradicted, so
they sit flagged ``current`` forever — clutter that a real memory would let
fade once the event has passed.

This pass finds CURRENT fact-edges whose text references a time that, relative
to when the fact was recorded, is now in the past, and marks them OUTDATED
(sets ``invalid_at`` / ``expired_at``). It NEVER deletes: the fact stays in the
graph as history (visible under the "outdated" filter), exactly like a memory
that faded from the foreground but isn't erased.

Precision over recall — only high-confidence relative terms (today/tonight/this
morning|afternoon|evening, tomorrow, this weekend) and explicit ISO dates are
matched, each with a ~1-day safety margin so nothing expires early across the
America/Chicago ↔ UTC offset. Weekday names ("on Friday") are intentionally
skipped as directionally ambiguous.

DRY-RUN BY DEFAULT. Pass --apply to write. Same connection/env contract as
cleanup_graph.py.

Usage:
  venv/bin/python plugins/memory/graphiti/temporal_expiry.py            # dry run
  venv/bin/python plugins/memory/graphiti/temporal_expiry.py --apply    # execute
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from neo4j import GraphDatabase  # noqa: E402

# Relative day terms → whole-day offset from the fact's anchor date.
_REL_OFFSETS = {
    "yesterday": -1,
    "today": 0, "tonight": 0, "this morning": 0,
    "this afternoon": 0, "this evening": 0,
    "tomorrow": 1,
}
_REL_RE = re.compile(
    r"(?i)\b(yesterday|tonight|today|this morning|this afternoon|this evening|tomorrow)\b"
)
_WEEKEND_RE = re.compile(r"(?i)\bthis weekend\b")
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# ~1 day grace so a Chicago-evening event never expires early in UTC.
_GRACE = timedelta(days=1)


def _load_dotenv() -> None:
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


def _parse_anchor(s: str | None) -> datetime | None:
    """Parse a Neo4j datetime string into an aware UTC datetime."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _end_of_day(d: datetime) -> datetime:
    return d.replace(hour=23, minute=59, second=59, microsecond=0)


def event_expiry(fact: str, anchor: datetime) -> datetime | None:
    """Return the UTC instant after which *fact* is stale, or None if it carries
    no resolvable time reference. ``anchor`` is when the fact was recorded."""
    if not fact:
        return None

    # 1. Explicit ISO date wins (unambiguous).
    m = _ISO_DATE_RE.search(fact)
    if m:
        try:
            d = datetime(int(m[1]), int(m[2]), int(m[3]), tzinfo=timezone.utc)
            return _end_of_day(d) + _GRACE
        except ValueError:
            pass

    # 2. "this weekend" → end of the Sunday in the anchor's week.
    if _WEEKEND_RE.search(fact):
        days_to_sun = 6 - anchor.weekday()  # Mon=0 .. Sun=6
        return _end_of_day(anchor + timedelta(days=days_to_sun)) + _GRACE

    # 3. Relative day terms.
    rm = _REL_RE.search(fact)
    if rm:
        offset = _REL_OFFSETS.get(rm.group(1).lower())
        if offset is not None:
            return _end_of_day(anchor + timedelta(days=offset)) + _GRACE

    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="perform changes (default: dry run)")
    args = ap.parse_args()

    _load_dotenv()
    cfg = _config()
    if not cfg["password"]:
        print("ERROR: no Neo4j password in env (GRAPHITI_NEO4J_PASSWORD / NEO4J_PASSWORD)", file=sys.stderr)
        return 2

    mode = "APPLY" if args.apply else "DRY-RUN"
    now = datetime.now(timezone.utc)
    print(f"=== Temporal expiry [{mode}] group={cfg['gid']}  now={now.isoformat()} ===\n")

    driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    gid = cfg["gid"]
    to_expire: list[dict] = []

    with driver.session() as s:
        rows = [r.data() for r in s.run(
            "MATCH ()-[e:RELATES_TO]->() "
            "WHERE e.group_id=$g AND e.invalid_at IS NULL AND e.expired_at IS NULL "
            "RETURN e.uuid AS uuid, e.fact AS fact, "
            "       coalesce(toString(e.valid_at), toString(e.created_at)) AS anchor",
            g=gid)]

        for r in rows:
            anchor = _parse_anchor(r.get("anchor"))
            if anchor is None:
                continue
            expiry = event_expiry(r.get("fact") or "", anchor)
            if expiry is not None and now > expiry:
                to_expire.append({"uuid": r["uuid"], "fact": r["fact"], "expiry": expiry})

        if not to_expire:
            print("No event-bound facts have passed their time. (none)")
            driver.close()
            return 0

        for t in to_expire:
            print(f"  expire (event passed {t['expiry'].date()}): {t['fact']!r}")
        print(f"\n=== SUMMARY: {len(to_expire)} current fact(s) to mark outdated ===")

        if not args.apply:
            print("\nDRY-RUN — nothing changed. Re-run with --apply to execute.")
            driver.close()
            return 0

        for t in to_expire:
            s.run(
                "MATCH ()-[e:RELATES_TO {uuid:$u}]->() "
                "SET e.invalid_at = datetime($iv), e.expired_at = datetime($ex)",
                u=t["uuid"], iv=t["expiry"].isoformat(), ex=now.isoformat())
        print(f"\nAPPLIED: marked {len(to_expire)} fact(s) outdated.")

    driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
