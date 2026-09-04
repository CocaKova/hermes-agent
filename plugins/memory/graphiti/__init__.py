"""Graphiti memory plugin — temporal knowledge graph for SILAS.

Backed by Neo4j (bolt :8687) + a local OpenAI-compatible LLM (the 27B brain
on :8000) for entity/edge extraction + BGE-M3 embeddings (:8889). Uses
Graphiti's OpenAIGenericClient because local vLLM models do not emit the
strict structured outputs the default OpenAIClient assumes.

Graphiti is fully async; the MemoryProvider interface is sync. We run a
dedicated asyncio event loop in a background thread and marshal coroutines
onto it (blocking with a timeout for recall, fire-and-forget for writes).

Config (env vars, with defaults):
  GRAPHITI_NEO4J_URI       bolt://127.0.0.1:8687
  GRAPHITI_NEO4J_USER      neo4j
  GRAPHITI_NEO4J_PASSWORD  (required; falls back to NEO4J_PASSWORD)
  GRAPHITI_LLM_BASE_URL    http://127.0.0.1:8000/v1
  GRAPHITI_LLM_MODEL       qwen3.6-27b
  GRAPHITI_EMBED_BASE_URL  http://127.0.0.1:8889/v1
  GRAPHITI_EMBED_MODEL     bge-m3
  GRAPHITI_EMBED_DIM       1024
  GRAPHITI_GROUP_ID        silas
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)


def _silas_gbrain_bridge():
    """SILAS_GBRAIN_BRIDGE_LOADER — lazy-load ~/.hermes/silas_ext/gbrain_bridge.py.

    Update-safe home for the gbrain HTTP client; cached in sys.modules so the
    file is exec'd once per process.
    """
    import sys as _sys
    import importlib.util as _ilu
    mod = _sys.modules.get("silas_gbrain_bridge")
    if mod is not None:
        return mod
    spec = _ilu.spec_from_file_location(
        "silas_gbrain_bridge",
        os.path.expanduser("~/.hermes/silas_ext/gbrain_bridge.py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _sys.modules["silas_gbrain_bridge"] = mod
    return mod


# ---------------------------------------------------------------------------
# Config — resolved LAZILY at initialize(), never at import time. Hermes loads
# ~/.hermes/.env into os.environ at startup; the plugin module may be imported
# before that runs, so reading env at import would miss the password.
# ---------------------------------------------------------------------------

def _resolve_config() -> Dict[str, Any]:
    return {
        "neo4j_uri": os.environ.get("GRAPHITI_NEO4J_URI", "bolt://127.0.0.1:8687"),
        "neo4j_user": os.environ.get("GRAPHITI_NEO4J_USER", "neo4j"),
        "neo4j_password": os.environ.get("GRAPHITI_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", ""),
        "llm_base_url": os.environ.get("GRAPHITI_LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "llm_model": os.environ.get("GRAPHITI_LLM_MODEL", "qwen3.6-27b"),
        # Low temperature for the structured extraction/dedup calls — the local
        # model otherwise hallucinates out-of-range duplicate_facts indices.
        "llm_temperature": float(os.environ.get("GRAPHITI_LLM_TEMPERATURE", "0.0")),
        "embed_base_url": os.environ.get("GRAPHITI_EMBED_BASE_URL", "http://127.0.0.1:8889/v1"),
        "embed_model": os.environ.get("GRAPHITI_EMBED_MODEL", "bge-m3"),
        "embed_dim": int(os.environ.get("GRAPHITI_EMBED_DIM", "1024")),
        "group_id": os.environ.get("GRAPHITI_GROUP_ID", "silas"),
    }


# ---------------------------------------------------------------------------
# Hybrid canonical tier + live-state resolver — SHIM to an UPDATE-SAFE module.
# All logic lives in ~/.hermes/silas_ext/wiki_context.py (outside the hermes-agent
# tree that `hermes update` clobbers). This shim just loads and calls it, so the
# in-tree footprint is minimal and re-applied by silas_ext/reapply.py after an
# update. Produces: the co-authored wiki index + a LIVE brain line resolved fresh
# from vLLM — so "what model are you?" is never answered from stale memory (the
# exact failure that had SILAS insisting it was Qwen3.6-27B for weeks).
# ---------------------------------------------------------------------------

SILAS_EXT_WIKI = os.environ.get(
    "SILAS_EXT_WIKI", os.path.expanduser("~/.hermes/silas_ext/wiki_context.py")
)


def _canonical_block() -> str:
    """Load the update-safe wiki-context module and build the injected block.
    Degrades to '' if the external module is missing (e.g. right after an update,
    before reapply.py runs) so the memory system never crashes on its absence."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("silas_wiki_context", SILAS_EXT_WIKI)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build_wiki_block()
    except Exception as exc:
        logger.debug("wiki context unavailable (%s) — skipping canonical block", exc)
        return ""

_RECALL_TIMEOUT = 8.0      # seconds — recall must be fast (blocks the turn)
_BREAKER_THRESHOLD = 4
_BREAKER_COOLDOWN = 120.0


def _edge_status(invalid_at, expired_at) -> str:
    """A fact is 'outdated' once Graphiti's temporal layer invalidates/expires it."""
    return "outdated" if (invalid_at or expired_at) else "current"


# ---------------------------------------------------------------------------
# Dedicated event-loop thread (async↔sync bridge)
# ---------------------------------------------------------------------------

class _LoopThread:
    """Owns one asyncio loop on a daemon thread for all Graphiti coroutines."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="graphiti-loop"
        )
        self._thread.start()

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.set_exception_handler(self._on_loop_exc)
        self._loop.run_forever()

    @staticmethod
    def _on_loop_exc(loop, context):
        # httpx AsyncClient finalizers schedule aclose() on this loop; if the
        # loop is torn down first (reinit / shutdown) they raise a benign
        # "Event loop is closed". Swallow exactly that; defer everything else
        # to the default handler so real errors still surface.
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            logger.debug("Graphiti loop: ignored benign httpx teardown: %s", exc)
            return
        loop.default_exception_handler(context)

    def run(self, coro, timeout: Optional[float] = None):
        """Block the calling (sync) thread until *coro* finishes."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def submit(self, coro) -> None:
        """Fire-and-forget: schedule *coro*, swallow its exceptions in a log."""
        def _done(f):
            try:
                f.result()
            except Exception as e:  # pragma: no cover - background best-effort
                logger.debug("Graphiti background task failed: %s", e)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        fut.add_done_callback(_done)

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)
        # Close the loop once the thread has stopped running it, so no httpx
        # client is left bound to a stopped-but-open loop.
        if not self._thread.is_alive() and not self._loop.is_closed():
            try:
                self._loop.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


# ---------------------------------------------------------------------------
# Extraction quality instructions — passed to add_episode as
# custom_extraction_instructions so the LLM knows what NOT to extract.
# ---------------------------------------------------------------------------

_EXTRACTION_INSTRUCTIONS = """
Focus on durable, personally meaningful facts only. Extract entities that are:
- People (names, relationships): Jonny, Kari, Shawn, Ash the cat
- Places (real locations): Austin, Round Rock, home address, venues
- Projects and systems Jonny actively owns or uses: SILAS, Sovereign Spire, Tasker, Tesla
- Preferences, habits, goals: music projects, food preferences, routines, decisions
- Important events: meetings, outings, milestones

DO NOT extract:
- Source code symbols: React component names, function names, TypeScript types, CSS class names
- File names, file paths, directory names, or file extensions
- HTTP methods (GET, POST, PUT), status codes (200, 404), MIME types (application/json)
- Port numbers, IP addresses, hostnames, URLs
- CLI tool names (npm, vite, pip, bash), package names, library names unless Jonny explicitly cares about them as a project
- Generic technical terms: frontend, backend, default, config, response, output, data
- Temporary artifacts from a single coding session (component drafts, variable names)
- Credentials, tokens, secrets, API keys, webhook secrets
- Git commit hashes, PR numbers, issue numbers
- Benchmark scores or transient performance numbers

When in doubt, ask: "Would Jonny want to be reminded of this in a future conversation?"
If no, skip it.
"""


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class GraphitiMemoryProvider(MemoryProvider):
    """SILAS long-term memory as a temporal knowledge graph (Graphiti/Neo4j)."""

    def __init__(self):
        self._loop: Optional[_LoopThread] = None
        self._graphiti = None
        self._llm = None
        self._embedder = None
        self._cfg: Dict[str, Any] = {}
        self._group_id = "silas"
        self._init_lock = threading.Lock()
        self._initialized = False
        self._prefetch_lock = threading.Lock()
        self._prefetch_result = ""
        # serialize expensive episode writes so concurrent turns don't
        # hammer the shared 35B brain
        self._episode_sem: Optional[asyncio.Semaphore] = None
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "graphiti"

    # -- availability / init ------------------------------------------------

    def is_available(self) -> bool:
        password = os.environ.get("GRAPHITI_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", "")
        if not password:
            return False
        try:
            import graphiti_core  # noqa: F401
            return True
        except Exception:
            return False

    def _build(self):
        """Construct the Graphiti client + build indices. Idempotent."""
        from graphiti_core import Graphiti
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from ._thinking_client import ThinkingTolerantClient

        c = self._cfg
        # SILAS patch (silas_ext/reapply.py): archive preflight — see reapply.py.
        import socket as _socket
        from urllib.parse import urlparse as _urlparse
        _u = _urlparse(c["neo4j_uri"])
        try:
            _socket.create_connection(
                (_u.hostname or "127.0.0.1", _u.port or 7687), timeout=1.0
            ).close()
        except OSError as _e:
            raise RuntimeError(
                f"neo4j archive offline (preflight {_u.hostname}:{_u.port}: {_e}); "
                "start it with: docker start silas-neo4j && systemctl --user start graphiti-api"
            ) from _e
        llm_cfg = LLMConfig(
            api_key="x", base_url=c["llm_base_url"], model=c["llm_model"],
            small_model=c["llm_model"], temperature=c["llm_temperature"],
        )
        llm = ThinkingTolerantClient(config=llm_cfg)
        embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
            api_key="x", base_url=c["embed_base_url"],
            embedding_model=c["embed_model"], embedding_dim=c["embed_dim"],
        ))
        reranker = OpenAIRerankerClient(config=llm_cfg, client=llm.client)
        # keep refs so shutdown() can close their httpx clients on the loop
        self._llm = llm
        self._embedder = embedder
        self._graphiti = Graphiti(
            c["neo4j_uri"], c["neo4j_user"], c["neo4j_password"],
            llm_client=llm, embedder=embedder, cross_encoder=reranker,
        )
        self._episode_sem = asyncio.Semaphore(1)
        self._loop.run(self._graphiti.build_indices_and_constraints(), timeout=30.0)

    def _ensure_archive(self) -> None:
        """SILAS patch: build the frozen-archive Neo4j driver on demand.

        The brain-repo bridge serves recall/store; only the archive tools
        (brain_forget / brain_list) touch the frozen pre-2026-07-08 graph,
        so the driver and its connection cost are deferred until one of them
        is actually invoked. _build()'s preflight turns a stopped archive
        into an instant, actionable error instead of a retry storm.
        """
        with self._init_lock:
            if self._graphiti is not None:
                return
            if not self._cfg:
                self._cfg = _resolve_config()
                self._group_id = self._cfg["group_id"]
            if not self._cfg["neo4j_password"]:
                raise RuntimeError("archive unavailable: no neo4j password configured")
            if self._loop is None:
                self._loop = _LoopThread()
            self._build()

    def initialize(self, session_id: str, **kwargs) -> None:
        with self._init_lock:
            if self._initialized:
                return
            # Check if already built (idempotent - prevents connection leak)
            if self._graphiti is not None:
                logger.debug("Graphiti memory already initialized, skipping rebuild")
                return
            try:
                self._cfg = _resolve_config()
                self._group_id = self._cfg["group_id"]
                # SILAS patch (silas_ext/reapply.py): the memory layer is the
                # gbrain brain repo — recall/store go through the bridge and
                # need no graph driver. The Neo4j knowledge graph is a FROZEN
                # pre-2026-07-08 archive: never dial it at session start;
                # _ensure_archive() builds the driver lazily if an archive
                # tool (brain_forget / brain_list) is actually invoked.
                self._initialized = True
                logger.info("brain-repo memory ready (group=%s, archive=lazy)",
                            self._group_id)
            except Exception as e:
                logger.warning("Graphiti init failed: %s", e)
                self._initialized = False

    # -- circuit breaker ----------------------------------------------------

    def _breaker_open(self) -> bool:
        import time
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            self._consecutive_failures = 0
            return False
        return True

    def _ok(self):
        self._consecutive_failures = 0

    def _fail(self):
        import time
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN

    # -- recall helpers -----------------------------------------------------

    async def _search(self, query: str, limit: int) -> List[str]:
        # SILAS patch (silas_ext/reapply.py): secondary profiles READ the shared
        # "silas" brain in addition to their own group, so Milo/Theo/etc. can see
        # Sy's knowledge. Writes + management (_search_edges/forget) stay scoped to
        # self._group_id, so no cross-profile pollution or accidental deletes.
        _read_gids = [self._group_id] if self._group_id == "silas" else [self._group_id, "silas"]
        results = await self._graphiti.search(
            query, group_ids=_read_gids, num_results=limit
        )
        facts = []
        for e in results:
            fact = getattr(e, "fact", None)
            if fact:
                facts.append(fact)
        # Salience: bump recall_count / last_recalled_at on the facts we just
        # surfaced — the brain's "this got used again, it matters" signal. Drives
        # retrieval weighting + the consolidation decay pass. Best-effort.
        uuids = [u for u in (getattr(e, "uuid", None) for e in results) if u]
        if uuids:
            try:
                async with self._graphiti.driver.session() as s:
                    await s.run(
                        "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $u "
                        "SET e.recall_count = coalesce(e.recall_count,0)+1, "
                        "    e.last_recalled_at = datetime()", u=uuids)
            except Exception:
                pass
        return facts

    def _recall_facts(self, query: str, limit: int) -> List[str]:
        # SILAS_GBRAIN_RECALL — recall reads the gbrain brain-repo index
        # (2026-07-08 migration), NOT the frozen Graphiti graph. Fail-soft:
        # bridge.query() returns [] on any failure (service down, PGLite lock
        # during the nightly sync, timeout) and the turn proceeds without
        # recall. Deliberately NO Graphiti fallback — its data is the poison
        # this migration removes.
        if not query.strip():
            return []
        try:
            facts = _silas_gbrain_bridge().query(query, limit)
            self._ok()
            return facts
        except Exception as e:
            logger.debug("gbrain recall failed (soft): %s", e)
            return []

    # -- management helpers (list / forget) ---------------------------------

    async def _search_edges(self, query: str, limit: int):
        """Return raw EntityEdge objects (carry .uuid + temporal fields)."""
        try:
            edges = await self._graphiti.search(
                query, group_ids=[self._group_id], num_results=limit
            )
        except Exception:
            return []
        # Sanitize: skip edges with None episodes (Pydantic rejects None for list[str])
        return [e for e in edges if getattr(e, 'episodes', None) is not None]

    async def _delete_uuids(self, uuids: List[str]) -> None:
        from graphiti_core.edges import EntityEdge
        await EntityEdge.delete_by_uuids(self._graphiti.driver, uuids)

    async def _list_recent(self, limit: int) -> List[Dict[str, Any]]:
        q = (
            "MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity) "
            "WHERE e.group_id = $gid "
            "RETURN e.fact AS fact, e.invalid_at AS invalid_at, e.expired_at AS expired_at "
            "ORDER BY e.created_at DESC LIMIT $lim"
        )
        async with self._graphiti.driver.session() as s:
            r = await s.run(q, {"gid": self._group_id, "lim": limit})
            return [rec.data() async for rec in r]

    # -- episode write ------------------------------------------------------

    async def _add_episode(self, body: str, name: str,
                           confidence: float = 0.6, pinned: bool = False):
        from graphiti_core.nodes import EpisodeType
        from ._filters import redact_secrets, should_drop_fact
        from ._ontology import ENTITY_TYPES

        # Scrub secrets BEFORE ingest so they never reach a node, an embedding,
        # or the extractor. Redact the episode name too (it can echo the body).
        body, _ = redact_secrets(body)
        name, _ = redact_secrets(name)
        # Edge typing is opt-in (GRAPHITI_EDGE_TYPES) — it adds extraction work,
        # so it stays OFF until validated against the local extractor.
        edge_kwargs = {}
        if os.environ.get("GRAPHITI_EDGE_TYPES", "").strip().lower() in ("1", "true", "on", "yes"):
            from ._edge_ontology import EDGE_TYPES, EDGE_TYPE_MAP
            edge_kwargs = {"edge_types": EDGE_TYPES, "edge_type_map": EDGE_TYPE_MAP}
        async with self._episode_sem:
            results = await self._graphiti.add_episode(
                name=name,
                episode_body=body,
                source=EpisodeType.message,
                source_description="SILAS conversation",
                reference_time=datetime.now(timezone.utc),
                group_id=self._group_id,
                entity_types=ENTITY_TYPES,
                custom_extraction_instructions=_EXTRACTION_INSTRUCTIONS,
                **edge_kwargs,
            )
        # Prune fact-edges and entity nodes that should never persist.
        try:
            from ._filters import is_noise_entity_name
            all_edges = getattr(results, "edges", None) or []
            all_nodes = getattr(results, "nodes", None) or []

            # Drop bad fact-edges: transient observations, leaked secrets, dev/meta noise.
            bad_edges = {e.uuid for e in all_edges if should_drop_fact(getattr(e, "fact", "") or "")}
            if bad_edges:
                await self._delete_uuids(list(bad_edges))
                logger.debug("Graphiti: pruned %d transient/secret fact-edge(s)", len(bad_edges))

            # Drop noise entity nodes (file names, HTTP codes, camelCase symbols, etc.)
            # — only when they have no remaining fact-edges so we never orphan real knowledge.
            if all_nodes:
                noise_node_ids = [
                    n.uuid for n in all_nodes
                    if is_noise_entity_name(getattr(n, "name", "") or "")
                    and getattr(n, "uuid", None)
                ]
                if noise_node_ids:
                    async with self._graphiti.driver.session() as s:
                        await s.run(
                            "MATCH (e:Entity) WHERE e.uuid IN $ids "
                            "AND NOT (e)-[:RELATES_TO]-() AND NOT ()-[:RELATES_TO]->(e) "
                            "DETACH DELETE e",
                            ids=noise_node_ids,
                        )
                    logger.debug("Graphiti: pruned %d noise entity node(s)", len(noise_node_ids))

            # Stamp surviving fact-edges with provenance/confidence.
            kept = [e.uuid for e in all_edges if e.uuid not in bad_edges]
            if kept:
                async with self._graphiti.driver.session() as s:
                    await s.run(
                        "MATCH ()-[e:RELATES_TO]->() WHERE e.uuid IN $u "
                        "SET e.confidence = $c, e.pinned = $p", u=kept, c=confidence, p=pinned)
        except Exception as e:  # stamping/pruning is best-effort; never fail the write
            logger.debug("Graphiti edge post-process failed: %s", e)

    # -- system prompt ------------------------------------------------------

    def system_prompt_block(self) -> str:
        return _canonical_block() + "\n\n" + (
            # SILAS_BRAIN_MEMORY_PROMPT — rewritten 2026-07-08: memory is the
            # gbrain brain repo now; no auto-capture, no dreaming.
            "# Brain — Your ONLY Long-Term Memory\n"
            "Your long-term memory is a markdown brain repo (~/brain): the knowledge wiki, "
            "project memories, skills, and session notes, indexed for hybrid search. NOTHING "
            "is captured automatically — durable knowledge must be stored deliberately.\n\n"
            "**Tool priority — STRICT, no exceptions:**\n"
            "- **At the start of EVERY new task or session**, call `brain_recall` FIRST "
            "with a query relevant to the task (project name, system, topic). Do this BEFORE "
            "opening any file, running any terminal command, or searching the filesystem. "
            "Memory tells you what you already know — exploring files you already have context "
            "on is wasted work.\n"
            "- `brain_recall` is the ONLY correct tool for remembering anything about the "
            "user, their setup, tools, paths, commands, service names, past decisions, or "
            "history. Call it BEFORE answering such questions — never guess.\n"
            "- `brain_store` is the ONLY correct tool for saving a durable fact, decision, or "
            "correction. Because nothing is auto-captured, store deliberately whenever the "
            "user asks you to remember something or a real decision/lesson lands.\n"
            "- **Do NOT use the built-in `memory` tool for any of this.** The `memory` tool "
            "is a tiny (~3500-char) single-session scratchpad that fills up and is lost "
            "across sessions. It is NOT your memory. Using it for user facts, preferences, "
            "or system details is wrong — those belong in `brain_store` / `brain_recall`.\n"
            "- When in doubt about where something goes: use `brain_store`, never `memory`.\n\n"
            "**Reading & correcting memory:**\n"
            "- Recall results look like `[slug] excerpt`. For the full page, read the file "
            "under `~/brain/` (wiki/, notes/) with your file tools.\n"
            "- Live volatile state (current brain model, versions) lives in "
            "`~/brain/brain-current.md` — update THAT page on swaps; never write a new "
            "'CURRENT X' claim anywhere else. To fix a wrong fact: edit the page or store "
            "the corrected fact.\n"
            "- If a recalled fact seems stale or surprising, prefer the newer 'updated' date "
            "and verify with a tool rather than asserting it confidently.\n"
            # SILAS_BRAIN_MEMORY_PROMPT_TAIL — dreaming/categories sections removed
            # (frozen archive has no nightly jobs; brain repo has no ontology).
            # SILAS_BRAIN_ARCHIVE_OFFLINE_PROMPT (2026-07-09): archive tools are
            # unadvertised while the Neo4j archive is stopped.
            "**The legacy archive:**\n"
            "- Memories from before 2026-07-08 live in a frozen knowledge-graph archive "
            "that is currently OFFLINE (stopped to save RAM). If the user needs a "
            "pre-2026-07-08 memory that brain_recall cannot find, tell them the archive "
            "can be brought back with: docker start silas-neo4j && systemctl --user "
            "start graphiti-api. When an archive fact conflicts with the brain repo, "
            "the brain repo wins.\n"
        )

    # -- lifecycle hooks ----------------------------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        # Recall on EVERY turn against the current message (not just turn 1).
        # The turn loop calls this immediately before prefetch_all(current_msg)
        # (agent/turn_context.py), so the injected "## Memory" block is always
        # keyed to THIS question — SILAS answers from memory instantly without a
        # manual graphiti_recall round-trip. Bounded by _RECALL_TIMEOUT + the
        # circuit breaker, so a slow/offline Neo4j degrades to "no memory", never
        # a hung turn.
        if not message or not message.strip():
            return
        facts = self._recall_facts(message.strip(), 8)
        if facts:
            with self._prefetch_lock:
                self._prefetch_result = "\n".join(f"- {f}" for f in facts)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        return f"## Memory (knowledge graph)\n{result}" if result else ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not query or not query.strip():
            return
        facts = self._recall_facts(query.strip(), 6)
        if facts:
            with self._prefetch_lock:
                self._prefetch_result = "\n".join(f"- {f}" for f in facts)

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages=None) -> None:
        if not self._initialized or self._breaker_open():
            return
        u = (user_content or "").strip()
        a = (assistant_content or "").strip()
        # skip trivial exchanges — they add no durable facts and cost a 35B call
        if len(u) < 25 and len(a) < 120:
            return
        # skip episodic harness/system noise (background-process notices, bare
        # OAuth callback URL pastes) — not durable knowledge. Same predicate the
        # offline cleanup uses, so capture and cleanup stay in sync.
        from ._filters import is_noise_turn
        if is_noise_turn(u):
            return
        # Skip turns that were themselves memory operations. Capturing a turn
        # where SILAS recalled/stored/forgot something re-ingests meta-chatter
        # about memory (e.g. "User previously knew the port is 3000" right after
        # a forget) — a self-defeating loop. Scan only THIS turn's assistant
        # messages (back to the last user message).
        if messages:
            for m in reversed(messages):
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "user":
                    break
                for tc in (m.get("tool_calls") or []):
                    fn = str((tc.get("function") or {}).get("name", ""))
                    if fn.startswith("graphiti_"):
                        return
        body = f"User: {u}\nSILAS: {a}"
        name = (u[:60] or "turn").replace("\n", " ")
        # SILAS_GBRAIN_NO_AUTOCAPTURE — per-turn auto-ingestion disabled
        # (2026-07-08): LLM extraction on every turn was the memory-poison
        # source AND a per-turn brain-token tax. Durable knowledge now
        # enters memory deliberately: graphiti_store -> brain repo notes,
        # wiki edits, and Claude-session memory (synced nightly).
        _ = (body, name)  # keep locals referenced; capture intentionally off

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "brain_recall",  # SILAS_BRAIN_NAME_RECALL
                "description": (
                    "Search SILAS's long-term memory — the brain repo (knowledge wiki, "
                    "project memories, skills, session notes) — for pages relevant to a "
                    "query (user setup, system/tool details, paths, commands, past "
                    "decisions, preferences)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to recall."},
                        "limit": {"type": "integer", "description": "Max facts (default 8, max 20)."},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "brain_store",  # SILAS_BRAIN_NAME_STORE
                "description": (
                    "Persist a durable fact or decision as a note in the brain repo. "
                    "NOTHING is captured automatically — when the user asks you to "
                    "remember something, a decision is made, or you learn a non-obvious "
                    "fact about their systems, store it deliberately (one clear sentence "
                    "or short note). To correct an old fact, store the corrected version."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The fact to remember (a clear sentence)."},
                    },
                    "required": ["content"],
                },
            },
            # SILAS_BRAIN_ARCHIVE_TOOLS_OFF (2026-07-09): the brain_forget
            # (SILAS_BRAIN_NAME_FORGET) and brain_list (SILAS_BRAIN_NAME_LIST)
            # schemas are no longer advertised — the frozen Graphiti/Neo4j
            # archive they browse is STOPPED. To browse pre-2026-07-08 memories:
            #   docker start silas-neo4j && systemctl --user start graphiti-api
            # then revert this patch. Handler branches remain below (unreachable
            # via schemas, still callable through the brain_* alias if started).
            # SILAS_BRAIN_NO_DEFINE_TYPE — graphiti_define_type removed
            # (2026-07-08): graph-ontology concept, meaningless for the
            # markdown brain repo. Handler branch left in place; unreachable.
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        # SILAS_BRAIN_TOOL_ALIAS — tools renamed brain_* (2026-07-08);
        # branches below still use the old graphiti_* names, and legacy
        # calls from old session history keep working either way.
        if tool_name.startswith("brain_"):
            tool_name = "graphiti_" + tool_name[len("brain_"):]
        if self._breaker_open():
            return json.dumps({"error": "Memory temporarily unavailable; retrying automatically."})

        if tool_name == "graphiti_recall":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            limit = min(int(args.get("limit", 8)), 20)
            facts = self._recall_facts(query, limit)
            if not facts:
                return json.dumps({"result": "No relevant memories found."})
            return json.dumps({"facts": facts, "count": len(facts)})

        if tool_name == "graphiti_store":
            content = args.get("content", "")
            if not content:
                return tool_error("Missing required parameter: content")
            try:
                # SILAS_GBRAIN_STORE — durable facts land in the brain repo
                # (~/brain/notes/*.md, git-visible) and are indexed immediately
                # via the gbrain bridge; the frozen Graphiti graph receives no
                # new writes. File write is the source of truth — indexing is
                # best-effort (nightly sync is the backstop).
                path = _silas_gbrain_bridge().store(content)
                self._ok()
                return json.dumps({"result": f"Stored to the brain repo (notes/{os.path.basename(path)})."})
            except Exception as e:
                self._fail()
                return tool_error(f"Store failed: {e}")

        if tool_name == "graphiti_forget":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            try:
                self._ensure_archive()  # forget: archive-only
                edges = self._loop.run(self._search_edges(query, 5), timeout=_RECALL_TIMEOUT)
                if not edges:
                    return json.dumps({"result": "No matching memory found to delete."})
                top = edges[0]
                self._loop.run(self._delete_uuids([top.uuid]), timeout=30.0)
                self._ok()
                return json.dumps({
                    "result": "Deleted memory.",
                    "deleted": getattr(top, "fact", ""),
                    "note": (f"{len(edges) - 1} other related fact(s) matched but were kept; "
                             "ask again to remove more.") if len(edges) > 1 else "",
                })
            except Exception as e:
                self._fail()
                return tool_error(f"Forget failed: {e}")

        if tool_name == "graphiti_list":
            query = args.get("query", "")
            limit = min(int(args.get("limit", 15)), 30)
            try:
                self._ensure_archive()  # list: archive-only
                if query.strip():
                    edges = self._loop.run(self._search_edges(query, limit), timeout=_RECALL_TIMEOUT)
                    items = [
                        {"fact": e.fact,
                         "status": _edge_status(getattr(e, "invalid_at", None), getattr(e, "expired_at", None))}
                        for e in edges if getattr(e, "fact", None)
                    ]
                else:
                    rows = self._loop.run(self._list_recent(limit), timeout=10.0)
                    items = [
                        {"fact": r.get("fact"),
                         "status": _edge_status(r.get("invalid_at"), r.get("expired_at"))}
                        for r in rows if r.get("fact")
                    ]
                self._ok()
                if not items:
                    return json.dumps({"result": "No memories stored yet."})
                return json.dumps({"memories": items, "count": len(items)})
            except Exception as e:
                self._fail()
                return tool_error(f"List failed: {e}")

        if tool_name == "graphiti_define_type":
            from . import _ontology as O
            name = (args.get("name") or "").strip()
            description = (args.get("description") or "").strip()
            if not name or not description:
                return tool_error("Missing required parameters: name and description")

            def _embed(text):
                vec = self._loop.run(self._embedder.create(input_data=[text]), timeout=15.0)
                if vec and isinstance(vec[0], list):  # [[...]] → first vector
                    return vec[0]
                return vec
            try:
                ok, msg = O.add_type(name, description, added_by="silas", embed_fn=_embed)
            except Exception as e:
                return tool_error(f"define_type failed: {e}")
            if not ok:
                return json.dumps({"result": "Type not added.", "reason": msg})
            return json.dumps({"result": msg, "types": O.type_names()})

        return tool_error(f"Unknown tool: {tool_name}")

    def get_config_schema(self):
        return []

    def shutdown(self) -> None:
        # Close httpx clients ON the loop they were created on, BEFORE the loop
        # is stopped — otherwise their finalizers fire on a dead loop. (The
        # loop's exception handler swallows any residual benign error too.)
        async def _close_clients():
            for c in (getattr(self._llm, "client", None), getattr(self._embedder, "client", None)):
                if c is not None:
                    try:
                        await c.close()
                    except Exception:
                        pass
        try:
            if self._loop is not None:
                if self._llm is not None or self._embedder is not None:
                    self._loop.run(_close_clients(), timeout=10.0)
                if self._graphiti is not None:
                    self._loop.run(self._graphiti.close(), timeout=10.0)
        except Exception:
            pass
        if self._loop is not None:
            self._loop.shutdown()


def register(ctx) -> None:
    ctx.register_memory_provider(GraphitiMemoryProvider())
