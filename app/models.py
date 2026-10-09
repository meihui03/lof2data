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

A Claude model (claude-...) is called through Anthropic's API, not Ollama, so
it is checked differently: offered only when ANTHROPIC_API_KEY is set, and
probed with a free model lookup that spends no tokens. Running out of prepaid
credit cannot be seen that way - it surfaces on the first real call, and takes
the model out of the selector the same way a spent Ollama quota does.

Local models are never probed. There's no quota to hit, and if the daemon
answered discovery it can serve them; a probe would only add page-load
latency (a cold local model has to load into memory first, which is far
slower than a cloud round-trip).
"""

import os

import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

AVAILABLE = "available"
RATE_LIMITED = "rate_limited"
PAYMENT_REQUIRED = "payment_required"
NOT_INSTALLED = "not_installed"
UNREACHABLE = "unreachable"
NO_API_KEY = "no_api_key"
BAD_API_KEY = "bad_api_key"

# The two models this app offers, deliberately a fixed list rather than
# everything Ollama has installed: these are the two the A/B benchmark in
# hybrid/ab_benchmark.py actually measured on the LOF forms (see
# hybrid/RECOMMENDATION.md), one cloud and one local. Other installed models
# may be vision-capable but are unmeasured on this document type, so they
# aren't offered. Add one here once it's been benchmarked.
SUPPORTED_MODELS = [
    {"name": "minimax-m3:cloud", "is_cloud": True, "label": "cloud"},
    {"name": "qwen3-vl:8b-instruct", "is_cloud": False, "label": "local"},
    # Not part of the Ollama A/B benchmark: added to try Claude on the LOF
    # forms. is_cloud stays False - that flag means Ollama cloud, with its
    # quota probe; Claude has its own check (probe_claude).
    {"name": "claude-sonnet-5-5", "is_cloud": False, "label": "claude api"},
]

# The model hybrid/extractor.py defaults to, and what this app used before it
# had a selector - so the default selection preserves existing behaviour.
PREFERRED_DEFAULT = "minimax-m3:cloud"

# Substrings that mark the two non-retryable cloud refusals (see module
# docstring). Both match text extractor.call_llm raises RuntimeError on.
_QUOTA_MARKER = "session usage limit"
# Ollama cloud's 402 body: "this model requires a subscription or extra
# usage, upgrade for access at https://ollama.com/upgrade".
_PAYMENT_MARKER = "requires a subscription"
# The Claude API's out-of-prepaid-credit error: "Your credit balance is too
# low to access the Anthropic API."
_CREDIT_MARKER = "credit balance is too low"

BLOCK_REASONS = {
    RATE_LIMITED: "cloud usage limit reached - try again after the quota window resets",
    PAYMENT_REQUIRED: ("cloud allowance spent - this model now needs a subscription "
                       "or extra usage: https://ollama.com/settings"),
    NOT_INSTALLED: "not installed - run `ollama pull` for it first",
    UNREACHABLE: "unreachable - the daemon did not answer for this model",
    NO_API_KEY: "set ANTHROPIC_API_KEY in .env to use the Claude API",
    BAD_API_KEY: "ANTHROPIC_API_KEY was rejected - check the key in .env",
}
_SESSION_REASON = ("became unavailable during a run in this session (usage limit or "
                   "subscription required) - use Re-check once that's resolved")


def is_cloud(name):
    """Ollama tags hosted models with a ':cloud' suffix. Only that suffix
    means hosted - a local model with 'cloud' elsewhere in its name is still
    local and must not be probed as though it had a quota."""
    return str(name).endswith(":cloud")


def is_claude(name):
    """Claude models are named by their API id, e.g. claude-sonnet-5-5."""
    return str(name).startswith("claude-")


def label_for(name):
    """Where a model runs, for the selector: local, Ollama cloud, or the
    Claude API."""
    if is_claude(name):
        return "claude api"
    return "cloud" if is_cloud(name) else "local"


def classify_claude_probe(api_key, status_code):
    """Turns a Claude model lookup into AVAILABLE / NO_API_KEY / BAD_API_KEY
    / UNREACHABLE. status_code is None when the lookup never got an HTTP
    answer. Pure, so the rules can be tested without a key."""
    if not api_key:
        return NO_API_KEY
    if status_code == 200:
        return AVAILABLE
    if status_code == 401:
        return BAD_API_KEY
    return UNREACHABLE


def classify_probe(status_code, body_text):
    """Turns one probe response into AVAILABLE / RATE_LIMITED /
    PAYMENT_REQUIRED / UNREACHABLE. Pure, so the rules can be tested without
    a daemon."""
    if status_code == 200:
        return AVAILABLE
    if status_code == 402:
        # The cloud allowance is spent and the model now needs a paid plan.
        # Classified on the status code alone - 402 is unambiguous, and the
        # body wording is Ollama's to change. This used to fall through to
        # UNREACHABLE, which told the user the daemon hadn't answered when it
        # had answered instantly with the actual, actionable reason.
        return PAYMENT_REQUIRED
    if status_code == 429:
        if _QUOTA_MARKER in (body_text or "").lower():
            return RATE_LIMITED
        return AVAILABLE  # transient burst limit; call_llm retries through it
    return UNREACHABLE


def looks_rate_limited(stderr):
    """Did a failed extraction fail because the cloud model is out of
    allowance - either a hard quota 429 or a 402 needing a paid plan? Reads
    the subprocess stderr that run_hybrid_extraction surfaces.

    Both count, because the consequence is the same: the model is dead for
    now and must come out of the selector. Matching only the 429 wording let
    a 402 through, so the user could pick the same dead model again and lose
    another full OCR pass to it."""
    text = (stderr or "").lower()
    return _QUOTA_MARKER in text or _PAYMENT_MARKER in text or _CREDIT_MARKER in text


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


def claude_client(timeout=20):
    """The SDK client the availability check uses - built the same way as the
    engine's (extractor._claude_client), so a key that passes here works there.
    ANTHROPIC_WORKSPACE_ID names the workspace for an organization-level key,
    which the API otherwise refuses."""
    import anthropic
    workspace = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"), timeout=timeout,
                               max_retries=1, default_headers=headers)


def probe_claude(name, timeout=20):
    """Checks the key and that this account can use the model, by looking
    the model up - free, no tokens. The SDK is imported here so the app
    still starts for someone who never installed it."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return classify_claude_probe(api_key, None)
    try:
        claude_client(timeout).models.retrieve(name)
        status = 200
    except ImportError:
        status = None
    except Exception as e:  # anthropic.APIStatusError carries the HTTP status
        status = getattr(e, "status_code", None)
    return classify_claude_probe(api_key, status)


def available_models(host=OLLAMA_HOST, session_rate_limited=frozenset()):
    """Checks the supported models: probes the Ollama cloud one, and looks up
    the Claude one with the API key.

    Returns (usable_names, [(name, reason)]). Propagates
    requests.RequestException from the installed-models lookup, which means
    Ollama itself is down - a different problem from "this model is out of
    quota" and worth a different message."""
    installed = installed_model_names(host)
    results = {}
    for model in SUPPORTED_MODELS:
        name = model["name"]
        if is_claude(name):
            results[name] = probe_claude(name)
        elif name not in installed:
            results[name] = NOT_INSTALLED
        elif model["is_cloud"]:
            results[name] = probe(name, host)
    return partition(SUPPORTED_MODELS, results, session_rate_limited)
