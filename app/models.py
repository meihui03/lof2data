"""Which of the supported models can this app actually offer right now?

Two models are offered - minimax-m3:cloud and qwen3-vl:8b-instruct - and this
module decides which of them are selectable at any moment. Kept out of app.py
so the UI file stays about UI: everything that talks to the Ollama daemon, and
every decision about whether a model is offerable, lives here.

The interesting decision is what counts as "rate limited". hybrid/extractor.py's
call_llm() distinguishes two kinds of HTTP 429 from Ollama cloud:

  - a genuine transient burst limit, which it retries with backoff and gets
    through - so the model IS usable and must stay selectable, and
  - a "session usage limit" 429, which is a quota that won't clear on the
    timescale of a retry, so call_llm fails fast on it - the model is NOT
    usable until the quota window resets.

Only the second one disables a model here. Treating every 429 as fatal would
lock the user out of a model that works fine.

Local models are never probed. There's no quota to hit, and if the daemon
answered discovery it can serve them; a probe would only add page-load
latency (a cold local model has to load into memory first, which is far
slower than a cloud round-trip).
"""

import requests

OLLAMA_HOST = "http://localhost:11434"

AVAILABLE = "available"
RATE_LIMITED = "rate_limited"
NOT_INSTALLED = "not_installed"
UNREACHABLE = "unreachable"

# The two models this app offers, deliberately a fixed list rather than
# everything Ollama has installed: these are the two the A/B benchmark in
# hybrid/ab_benchmark.py actually measured on the LOF forms (see
# hybrid/RECOMMENDATION.md), one cloud and one local. Other installed models
# may be vision-capable but are unmeasured on this document type, so they
# aren't offered. Add one here once it's been benchmarked.
SUPPORTED_MODELS = [
    {"name": "minimax-m3:cloud", "is_cloud": True, "label": "cloud"},
    {"name": "qwen3-vl:8b-instruct", "is_cloud": False, "label": "local"},
]

# The model hybrid/extractor.py defaults to, and what this app used before it
# had a selector - so the default selection preserves existing behaviour.
PREFERRED_DEFAULT = "minimax-m3:cloud"

# Substring that marks the non-retryable quota 429 (see module docstring).
# Matches the condition extractor.call_llm raises RuntimeError on.
_QUOTA_MARKER = "session usage limit"

BLOCK_REASONS = {
    RATE_LIMITED: "cloud usage limit reached - try again after the quota window resets",
    NOT_INSTALLED: "not installed - run `ollama pull` for it first",
    UNREACHABLE: "unreachable - the daemon did not answer for this model",
}
_SESSION_REASON = "hit its usage limit during a run in this session - use Re-check once it resets"


def is_cloud(name):
    """Ollama tags hosted models with a ':cloud' suffix. Only that suffix
    means hosted - a local model with 'cloud' elsewhere in its name is still
    local and must not be probed as though it had a quota."""
    return str(name).endswith(":cloud")


def classify_probe(status_code, body_text):
    """Turns one probe response into AVAILABLE / RATE_LIMITED / UNREACHABLE.
    Pure, so the rate-limit rule can be tested without a daemon."""
    if status_code == 200:
        return AVAILABLE
    if status_code == 429:
        if _QUOTA_MARKER in (body_text or "").lower():
            return RATE_LIMITED
        return AVAILABLE  # transient burst limit; call_llm retries through it
    return UNREACHABLE


def looks_rate_limited(stderr):
    """Did a failed extraction fail because of the cloud quota? Reads the
    subprocess stderr that run_hybrid_extraction surfaces."""
    return _QUOTA_MARKER in (stderr or "").lower()


def partition(supported, probe_results, session_rate_limited=frozenset()):
    """Splits the supported models into (usable_names, [(name, reason)]).

    A model with no probe result is treated as usable - local models are
    never probed by design, and refusing to offer something that was never
    checked would be wrong. session_rate_limited holds models that 429'd
    during a real extraction in this session; they stay blocked even when
    the cached probe still says fine, since the cache can be up to its TTL
    out of date."""
    usable, blocked = [], []
    for model in supported:
        name = model["name"]
        if name in session_rate_limited:
            blocked.append((name, _SESSION_REASON))
            continue
        status = probe_results.get(name, AVAILABLE)
        if status == AVAILABLE:
            usable.append(name)
        else:
            blocked.append((name, BLOCK_REASONS[status]))
    return usable, blocked


def pick_default(usable):
    """The production model when it's offerable, otherwise the first thing
    that is. None when nothing is."""
    if PREFERRED_DEFAULT in usable:
        return PREFERRED_DEFAULT
    return usable[0] if usable else None


# ------------------------------------------------------------------ network

def installed_model_names(host=OLLAMA_HOST, timeout=10):
    """Names of every model the daemon has. Raises requests.RequestException
    if Ollama isn't reachable at all - the caller turns that into a "start
    Ollama" message rather than an empty, unexplained dropdown."""
    tags = requests.get(f"{host}/api/tags", timeout=timeout)
    tags.raise_for_status()
    return {entry["name"] for entry in tags.json().get("models", []) if entry.get("name")}


def probe(name, host=OLLAMA_HOST, timeout=20):
    """One-token round-trip to check a cloud model is actually serving.
    num_predict=1 keeps it to ~1s and a couple of hundred tokens."""
    try:
        response = requests.post(
            f"{host}/api/chat",
            json={
                "model": name,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
    except requests.RequestException:
        return UNREACHABLE
    return classify_probe(response.status_code, response.text)


def available_models(host=OLLAMA_HOST, session_rate_limited=frozenset()):
    """Checks the two supported models and probes the cloud one.

    Returns (usable_names, [(name, reason)]). Propagates
    requests.RequestException from the installed-models lookup, which means
    Ollama itself is down - a different problem from "this model is out of
    quota" and worth a different message."""
    installed = installed_model_names(host)
    results = {}
    for model in SUPPORTED_MODELS:
        name = model["name"]
        if name not in installed:
            results[name] = NOT_INSTALLED
        elif model["is_cloud"]:
            results[name] = probe(name, host)
    return partition(SUPPORTED_MODELS, results, session_rate_limited)
