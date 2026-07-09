"""Shared content filters for the Graphiti memory plugin.

Single source of truth for what must NOT become durable knowledge:

  - secrets (app passwords, API keys, tokens) — redacted from episode bodies
    *before* ingest so they never reach a node, an embedding, or the
    extractor, and dropped if they slip through as a fact-edge.
  - transient observations (inbox/email/calendar snapshots) — one-time
    reports, not semantic facts. Pruned from the edges an episode produces.

Both the live plugin (__init__.py) and the offline cleanup script import from
here so capture-time filtering and after-the-fact cleanup stay identical.

All matching is deterministic regex — no LLM call, no latency, no flakiness.
Patterns favor precision: it is better to keep a borderline fact than to
silently shred a legitimate one.
"""

from __future__ import annotations

import re
from typing import List, Tuple

REDACTION = "[REDACTED-SECRET]"

# ---------------------------------------------------------------------------
# Secrets — high-precision value shapes. Anything matching here is a credential
# and is redacted out of episode bodies (and dropped as a fact).
# ---------------------------------------------------------------------------

_SECRET_VALUE_PATTERNS: List[re.Pattern] = [
    # Google app password: four groups of four lowercase letters ("hbri tgcs ..").
    re.compile(r"\b(?:[a-z]{4}[ -]){3}[a-z]{4}\b"),
    # OpenAI / Anthropic style keys.
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    # GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_) and fine-grained PATs.
    re.compile(r"\bgh[posur]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # AWS access key id.
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # Slack tokens.
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    # JWTs.
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    # Tesla / generic OAuth codes: NA_<alphanum 20+> (Tesla fleet auth codes).
    re.compile(r"\bNA_[A-Za-z0-9]{20,}\b"),
]

# Labeled secrets: "app password: <value>", "api key = <value>". Redacts the
# value, not the label. Requires an explicit separator so prose like
# "reset your password" is left alone.
_SECRET_LABELED_PATTERN = re.compile(
    r"(?i)\b(app[\s-]?password|application[\s-]?password|api[\s_-]?key|"
    r"access[\s_-]?token|secret[\s_-]?key|client[\s_-]?secret|bearer)\b"
    r"\s*[:=]\s*\S{6,}"
)

# OAuth/auth callback URL query secrets: "...?code=NA_...", "&state=...",
# "&access_token=...". Single-use auth codes, state nonces and tokens are
# credentials AND noise. Anchored on a query separator (?/&) + a known
# sensitive param name so ordinary prose ("exit code 143") is never touched.
# Keeps the param name, redacts only the value (group 1 is "?code=" / "&state=").
_OAUTH_URL_PATTERN = re.compile(
    r"(?i)([?&](?:code|state|access[_-]?token|id[_-]?token|refresh[_-]?token|"
    r"client[_-]?secret|session[_-]?state|token)=)[^&\s#\"']{4,}"
)


# A secret-shaped token appearing shortly AFTER a credential cue word — catches
# "app password for my account: hbritgcspkfiagyu" (the no-space form the 4x4
# pattern misses). The cue locates the window; a token only qualifies if it
# *looks* like a secret (see _looks_secret_token), so ordinary prose after the
# word "password" ("authentication failed", "generated specifically for…") is
# NOT redacted.
_SECRET_CUE = re.compile(r"(?i)\b(?:password|passwd|api[\s_-]?key|secret|token|credential)\b")
_CANDIDATE_TOKEN = re.compile(r"\b[A-Za-z0-9]{12,}\b")
_CUE_WINDOW = 40


def _looks_secret_token(tok: str) -> bool:
    """True for high-entropy-ish tokens, False for plain English words."""
    if len(tok) >= 16:                      # gmail app passwords are 16 chars
        return True
    has_digit = any(c.isdigit() for c in tok)
    has_upper = any(c.isupper() for c in tok)
    has_lower = any(c.islower() for c in tok)
    return has_digit or (has_upper and has_lower)


def _near_cue_secret_spans(text: str) -> List[Tuple[int, int]]:
    """Spans of secret-shaped tokens within _CUE_WINDOW chars after a cue word."""
    spans: List[Tuple[int, int]] = []
    for cue in _SECRET_CUE.finditer(text):
        window = text[cue.end(): cue.end() + _CUE_WINDOW]
        for tok in _CANDIDATE_TOKEN.finditer(window):
            if _looks_secret_token(tok.group(0)):
                spans.append((cue.end() + tok.start(), cue.end() + tok.end()))
                break  # only the first qualifying token per cue
    return spans


def redact_secrets(text: str) -> Tuple[str, bool]:
    """Return (redacted_text, found). Replaces any secret value with REDACTION."""
    if not text:
        return text, False
    found = False

    def _label_sub(m: re.Match) -> str:
        nonlocal found
        found = True
        # keep the label + separator, redact only the value
        label = m.group(0)
        sep_idx = max(label.rfind(":"), label.rfind("="))
        return f"{label[:sep_idx + 1]} {REDACTION}"

    out = _SECRET_LABELED_PATTERN.sub(_label_sub, text)

    def _oauth_sub(m: re.Match) -> str:
        nonlocal found
        found = True
        return f"{m.group(1)}{REDACTION}"  # keep "?code=" / "&state=", drop value

    out = _OAUTH_URL_PATTERN.sub(_oauth_sub, out)

    # near-cue spans (right-to-left so indices stay valid as we splice)
    spans = _near_cue_secret_spans(out)
    if spans:
        found = True
        for start, end in sorted(spans, reverse=True):
            out = out[:start] + REDACTION + out[end:]

    for pat in _SECRET_VALUE_PATTERNS:
        out, n = pat.subn(REDACTION, out)
        if n:
            found = True
    return out, found


def contains_secret(text: str) -> bool:
    if not text:
        return False
    if (_SECRET_LABELED_PATTERN.search(text) or _OAUTH_URL_PATTERN.search(text)
            or _near_cue_secret_spans(text)):
        return True
    return any(p.search(text) for p in _SECRET_VALUE_PATTERNS)


# ---------------------------------------------------------------------------
# Transient observations — one-time inbox/email/calendar/notification reports.
# These are episodic, not semantic; they should never become durable facts.
#
# Deliberately does NOT match durable facts like "has an email address X" or
# "uses Gmail" — only the act of receiving / a snapshot of inbox state.
# ---------------------------------------------------------------------------

_TRANSIENT_FACT_PATTERNS: List[re.Pattern] = [
    # Bank/account balance snapshots: "has a balance of $1,234" — always stale, always fetched live.
    re.compile(r"(?i)\bbalance\s+of\s+\$[\d,]+"),
    re.compile(r"(?i)\b(?:account|checking|savings)\s+(?:balance|amount)\b.*\$[\d,]+"),
    re.compile(r"(?i)\breceived\s+(?:a|an|the|\d+|some|new|another)\s+"
               r"(?:email|e-mail|message|text|sms|notification|alert|reminder|call)"),
    re.compile(r"(?i)\b(?:email|e-mail|message|notification|alert)\s+from\b"),
    re.compile(r"(?i)\bemail\s+about\b"),
    re.compile(r"(?i)\binbox\b.*\b(?:has|have|contains?|received|shows?|holds?)\b"),
    # "has 3 unread emails", "has new messages" — requires a quantity/unread
    # qualifier so durable facts like "has an email address" are left alone.
    re.compile(r"(?i)\bhas\s+(?:\d+|a few|some|several|new|unread)\s+(?:unread\s+|new\s+)?"
               r"(?:email|e-mail|message|notification|alert)s?\b"),
    re.compile(r"(?i)\bhas\s+an?\s+event\s+scheduled\b"),
    re.compile(r"(?i)\b(?:unread|new)\s+(?:email|e-mail|message|notification)s?\b"),
]


def is_transient_fact(fact: str) -> bool:
    if not fact:
        return False
    return any(p.search(fact) for p in _TRANSIENT_FACT_PATTERNS)


# ---------------------------------------------------------------------------
# Meta-facts — the memory system describing ITSELF or its own contents. These
# arise when the user and SILAS discuss the graph / cleanup / hallucinations,
# and the extractor mints "facts" out of that meta-talk (a self-reinforcing
# pollution loop: talking about the junk creates new junk).
#
# Tuned to catch commentary ABOUT the store ("X is a hallucinated entity",
# "auto-capture pulled in…", "connected with N edges") WITHOUT matching real
# project facts about graphiti-viz (ports, stack, theme), which never use this
# vocabulary.
# ---------------------------------------------------------------------------

_META_FACT_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)hallucinat"),                         # hallucinated/-ion/-e
    re.compile(r"(?i)\bauto[\s-]?capture\b"),              # the capture mechanism
    re.compile(r"(?i)\bfictitious entity\b"),
    re.compile(r"(?i)\bconnected with \d+ edges?\b"),
    re.compile(r"(?i)\bMCP\b.{0,30}\b(?:graph|capture|entit|fact|node)"),
    re.compile(r"(?i)\b(?:fact[- ]edge|RELATES_TO|episodic node|entity node|knowledge graph)\b"),
    re.compile(r"(?i)does(?:n['’]?t| not)\s+have\s+(?:any\s+)?(?:persistent\s+)?memor"),
]


def is_meta_fact(fact: str) -> bool:
    if not fact:
        return False
    return any(p.search(fact) for p in _META_FACT_PATTERNS)


# ---------------------------------------------------------------------------
# Dev / ops noise — enumerations of build/runtime internals that the extractor
# mints into "facts" during coding & ops sessions: ComfyUI node-name lists,
# package-version inventories, optional-package warnings, custom-node load
# chatter. These are throwaway operational detail, not durable knowledge.
#
# Deliberately NARROW so durable config facts survive: "ComfyUI runs on port
# 8188 for user cocakova" and "Spire Express server runs on port 3000" do NOT
# match (no node-name / warning / version-installed / custom-node vocabulary).
# ---------------------------------------------------------------------------

_DEV_NOISE_FACT_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bis an? .{1,40} node name\b"),                 # "X is a ComfyUI node name"
    re.compile(r"(?i)\bwarnings?\b.{0,30}\b(?:optional )?package\b"), # "warnings for the optional package onnx"
    re.compile(r"(?i)\bhas .{1,40} version [\w.]+ installed\b"),      # "has PyTorch version 2.12.0 installed"
    re.compile(r"(?i)\bloads the .{1,40} custom node\b"),             # "loads the X custom node"
    re.compile(r"(?i)\buses \w+ as its context implementation\b"),    # SQLiteImpl noise
    # SILAS procedural session steps — what SILAS did/intends to do in one turn,
    # not durable facts. "SILAS integrates/uses/supports X" (capabilities) are
    # present-tense and do NOT match these patterns.
    re.compile(
        r"(?i)^SILAS\s+(?:"
        r"shut\s+down\b|restart(?:ed|s)?\b|"
        r"check(?:ed|s)\s+the\b|"
        r"request(?:ed|s)?\b.{0,30}\b(?:oauth|auth(?:orization)?|access)\b|"
        r"generat(?:ed|es|ing)\s+(?:a\s+)?(?:fresh|new)\b|"
        r"notif(?:ied|ies)\s+(?:user|jonny)\b|"
        r"is\s+(?:facilitating|attempting|trying|checking|restarting|shutting)\b|"
        r"intends?\s+to\b|"
        r"reports?\s+that\b|"
        r"attempt(?:ed|s|ing)?\s+to\b"
        r")"
    ),
]


def is_dev_noise_fact(fact: str) -> bool:
    if not fact:
        return False
    return any(p.search(fact) for p in _DEV_NOISE_FACT_PATTERNS)


# ---------------------------------------------------------------------------
# Ephemeral recommendations — SILAS suggesting menu items / options in
# conversation ("recommends Tacos de Coliflor as an entree for …"). These are
# in-the-moment suggestions tied to a one-off event, not durable facts about
# the user. Anchored on "recommend/suggest … as a <menu-role>" so durable
# advice ("recommends using NVFP4") is NOT matched.
# ---------------------------------------------------------------------------

_EPHEMERAL_REC_PATTERN = re.compile(
    r"(?i)\b(?:recommends?|suggests?|proposes?)\b.{0,80}?\bas an?\s+"
    r"(?:starter|appetizer|entree|entrée|main course|main|side dish|side|"
    r"dessert|drink|beverage|cocktail|dish|course|option)\b"
)


def is_ephemeral_recommendation(fact: str) -> bool:
    if not fact:
        return False
    return bool(_EPHEMERAL_REC_PATTERN.search(fact))


# ---------------------------------------------------------------------------
# Noise entity NAMES — entities the extractor pulled out of coding/ops turns:
# filenames, source paths, API endpoints, bare CLI tools, port-only tokens,
# harness process ids. Used by the offline orphan-sweep (cleanup_graph.py),
# which only ever removes such an entity when it has NO fact-edges, so a real
# entity that happens to match is never lost.
# ---------------------------------------------------------------------------

_NOISE_NAME_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\.(?:tsx?|jsx?|py|json|sh|md|css|html?|ya?ml|toml|cfg|ini|lock|txt|log)$"),
    re.compile(r"^/"),                       # API endpoints / absolute paths: "/api/graph", "/health"
    re.compile(r"(?:^|/)src/"),              # source paths: "src/index.ts"
    re.compile(r"^:?\d{2,5}$"),              # port-only tokens: ":3001", "8188"
    re.compile(r"(?i)^proc_[0-9a-f]+$"),     # harness process ids
    # React/TS component and function names: camelCase or PascalCase identifiers
    # that are pure code symbols, not real-world entities.
    re.compile(r"^[a-z][a-zA-Z0-9]{3,}(?:[A-Z][a-zA-Z0-9]+)+$"),   # camelCase: formatCurrency, addEpisode
    re.compile(r"^[A-Z][a-zA-Z0-9]+(?:Page|Component|Dashboard|Chart|Layout|View|Modal|Panel|Hook|Context|Provider|Reducer|Store|Service|Controller|Handler|Wrapper|Container|Widget|Card)$"),
    # HTTP noise: methods, status codes, MIME types, bare numeric codes
    re.compile(r"^(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS|TRACE)$"),
    re.compile(r"^\d{3}(?:/\d{3})?$"),       # "200", "401/403"
    re.compile(r"^[a-z]+/[a-z+.-]+$"),       # MIME types: "application/json", "text/html"
    # Git short hashes: 7-12 hex chars (not a real-world entity)
    re.compile(r"^[0-9a-f]{7,12}$"),
    # Leet/obfuscated tokens: mixed alphanum with digits replacing letters (e.g. t4sk3r_s3cr3t)
    re.compile(r"(?i)^[a-z0-9]{4,}(?:[_-][a-z0-9]{2,}){2,}$"),     # snake_case tokens: t4sk3r_s3cr3t_c04k0v4
    # Bare URL fragments / webhook paths
    re.compile(r"^(?:https?://|wss?://)"),
    # "port NNNN" spelled out
    re.compile(r"(?i)^port\s+\d{2,5}$"),
    # % template variables
    re.compile(r"^%[A-Z_]+$"),
]

_NOISE_NAME_EXACT = {
    "node", "npm", "npx", "tsx", "ts-node", "pip", "pip3", "venv", "curl",
    "wget", "git", "bash", "sh", "vite", "webpack", "browser", "test server",
    # Generic coding/ops terms that never carry personal meaning
    "default", "frontend", "backend", "graph", "user", "response", "request",
    "output", "input", "error", "result", "data", "value", "config", "true", "false",
    # HTTP noise words
    "post", "get", "put", "delete", "patch",
}


def is_noise_entity_name(name: str) -> bool:
    if not name:
        return False
    n = name.strip()
    if n.lower() in _NOISE_NAME_EXACT:
        return True
    return any(p.search(n) for p in _NOISE_NAME_PATTERNS)


def should_drop_fact(fact: str) -> bool:
    """A fact-edge that should never persist: transient, secret, meta, dev-noise,
    or an ephemeral menu/option recommendation."""
    return (is_transient_fact(fact) or contains_secret(fact) or is_meta_fact(fact)
            or is_dev_noise_fact(fact) or is_ephemeral_recommendation(fact))


# ---------------------------------------------------------------------------
# Noise turns — whole conversation turns that are episodic harness/system noise,
# not knowledge. Unlike the fact-edge filters above (which prune what the
# extractor *derives*), this drops the EPISODE itself so the extractor never
# runs on it. Two classes seen polluting the graph:
#   1. Harness/background-process notices: "[IMPORTANT: Background process
#      proc_... exited (exit code 143, SIGTERM)]" (see
#      tools/process_registry.py:format_process_notification).
#   2. Bare OAuth/auth callback URL pastes: "http://localhost:3000/auth/tesla/
#      callback?code=NA_..." — one-time auth artifacts, not durable facts.
# Used at capture time (skip the write) and by cleanup_graph.py (delete nodes).
# ---------------------------------------------------------------------------

_PROCESS_NOTICE_RE = re.compile(
    r"\[IMPORTANT:\s*Background process\b"
    r"|\[IMPORTANT:[^\]]*\b(?:exit code|SIGTERM|SIGKILL|terminated|exited)\b",
    re.I,
)

_AUTH_URL_RE = re.compile(
    r"https?://\S*?(?:/auth\b|/callback\b|oauth)\S*?[?&]"
    r"(?:code|state|token|session_state)=",
    re.I,
)

# Automated verification prompts (e.g. from Claude Code) that the sender has
# explicitly self-labeled as tests to ignore. These were extracting encyclopedia
# trivia into the graph (a "write an essay about the Greek alphabet" test spawned
# ~20 Zeus/Euboea/Etruscan entities). Only drop when the turn LABELS ITSELF a
# test/ignore — a genuine request that merely contains the word "test" is left
# alone (it needs the "ignore"/"from Claude" cue too).
_AUTOMATED_TEST_RE = re.compile(
    r"\bautomated test from claude\b"
    r"|\btest from claude code\b"
    r"|\(\s*(?:last |one more |another |final )?automated test\b"
    r"|\btest from claude[^)]*\bignore\b"
    r"|\bignore[^)]*\btest from claude\b",
    re.I,
)


def _user_portion(text: str) -> str:
    """The user's text from an episode body 'User: <u>\\nSILAS: <a>'.

    Episodes are stored with that framing; capture-time callers pass the raw
    user message (no framing). Handle both: strip a leading 'User:' label and
    cut at the assistant turn if present.
    """
    m = re.match(r"(?is)^\s*user:\s*(.*?)(?:\nsilas:|\Z)", text)
    return (m.group(1) if m else text).strip()


def is_noise_turn(text: str) -> bool:
    """True for episodic harness/system noise that must not become a memory."""
    if not text:
        return False
    if _PROCESS_NOTICE_RE.search(text):
        return True
    up = _user_portion(text)
    # Self-labeled automated test/verification prompts (and the essays they ask
    # for) — never durable knowledge.
    if _AUTOMATED_TEST_RE.search(up):
        return True
    # "essentially just an auth URL" — the URL plus at most a couple of words
    # ("here: <url>"). A URL buried in a real sentence is left alone.
    if len(up.split()) <= 3 and re.search(r"https?://", up):
        if _AUTH_URL_RE.search(up):
            return True
        # OIDC redirect with no /auth path: an issuer + a state/code nonce.
        low = up.lower()
        if "iss=" in low and re.search(r"[?&](?:state|code)=", low):
            return True
    return False
