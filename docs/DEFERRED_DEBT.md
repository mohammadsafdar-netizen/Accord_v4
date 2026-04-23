# Deferred Technical Debt

Items identified during production review (2026-04-22) that are known, understood, and
deliberately deferred. Each entry includes the risk, the deferral reason, and a trigger
condition for when it becomes load-bearing.

---

## D1 — SQLite blocks the asyncio event loop

**Files:** `accord_ai/core/store.py`, `accord_ai/conversation/controller.py`, `accord_ai/api.py`

**Risk:** Every `store.get_session()`, `store.append_message()`, `store.update_submission()`,
etc. is called directly from async handlers without `asyncio.to_thread()`. SQLite is
synchronous I/O. Under concurrent API load, each blocked call stalls the event loop and
prevents other in-flight coroutines from making progress.

**Current exposure:** Low. The API is single-tenant demo traffic; `concurrency=1` in eval.
Uvicorn runs with a single worker process. No measured latency cliff yet.

**Deferral reason:** Correct fix is either (a) `asyncio.to_thread()` wrapping every store
call, or (b) migrate to an async-native store (aiosqlite, SQLAlchemy async). Both require
touching ~30 call sites across controller + API. Risk of regression is real; benefit is
zero until concurrent load materializes.

**Trigger to fix:** Any of: (1) API exposed to >1 concurrent tenant, (2) p95 latency
metrics show >200ms on store calls, (3) uvicorn moved to multi-worker mode.

---

## D2 — WAL mode not documented on per-thread connections

**File:** `accord_ai/core/store.py:229–237`

**Risk:** `PRAGMA journal_mode=WAL` is set once at `__init__` on the initializer
connection. WAL is a per-file flag in SQLite — all subsequent connections to the same
file automatically use WAL. The `_conn()` per-thread lazy connection does not re-assert
the PRAGMA, which is correct behavior but surprising to a reader.

**Current exposure:** None. WAL is sticky on the file.

**Deferral reason:** No bug, only a documentation gap. Fix is a single comment.

**Fix:** Add comment to `_conn()`: _"WAL is per-database-file in SQLite — no need to
re-assert per connection; the flag was set at init time."_

---

## D3 — CORS default `"*"` silently disables credentials in production

**Files:** `accord_ai/config.py:124`, `accord_ai/api.py:2399–2404`

**Risk:** Default `allowed_origins="*"` is detected at startup and `allow_credentials`
is set to `False`. Any browser client sending `Authorization: Bearer` with
`withCredentials=true` will receive a CORS rejection. The warning log at startup
is the only signal — an operator who misses it ships a broken auth integration.

**Current exposure:** Low. Auth is disabled in dev; CORS wildcard is intentional for
ngrok/tunnel demos. No browser clients with credentials in production yet.

**Deferral reason:** Changing the default would break existing demo deployments that
rely on wildcard. Requires coordinating with any frontend consumers before changing.

**Fix when triggered:** Change default to `""` (empty string forces explicit config) or
elevate the startup log from `WARNING` to `ERROR` so it surfaces in alerting.

**Trigger:** Any browser-based frontend that sends credentialed requests, or first
multi-tenant production deployment with per-tenant API keys.

---

## D4 — `/sessions` and `/messages` API limits (partially fixed 2026-04-22)

**File:** `accord_ai/api.py`

**Status:** `/sessions` capped at 500; `/messages` capped at 1000 (fixed). The default
`limit=50` on `/sessions` is still a bare `int` with no `ge=1` annotation — a caller
can pass `limit=0` or `limit=-1` and the store will execute `LIMIT 0` or `LIMIT -1`
(SQLite treats negative LIMIT as no limit).

**Fix:** Add `ge=1` to the `/sessions` `limit` parameter.

---

## D5 — `TRACKER` singleton non-atomic under concurrent load

**File:** `accord_ai/llm/json_validity_tracker.py`

**Risk:** `total_attempts += 1` and the other counter increments are not atomic at the
CPython bytecode level (LOAD / BINARY_ADD / STORE). Under concurrent asyncio load with
multi-coroutine concurrency, counter values can be lost. The module docstring documents
this limitation.

**Current exposure:** None. The tracker is used only in eval runs (`concurrency=1`). The
production extraction path imports the tracker but any production load is single-process
uvicorn — the asyncio event loop is cooperative, so GIL-level races do not occur in
practice without actual threading.

**Deferral reason:** Adding `threading.Lock` adds complexity to a diagnostic-only module.
Eval use (the intended use) is always `concurrency=1`.

**Trigger:** Step 25 matrix runs with `concurrency>1`, or if the tracker is ever wired
into a production monitoring path.

---

## D6 — `/answer` endpoint missing auth dependency

**Source:** MEMORY.md (identified in prior code review, not yet resolved)

**Risk:** The `/answer` endpoint may be missing the auth `Depends(_auth_tenant)` call
that gates other endpoints. If so, unauthenticated callers can submit turns to any
session ID they can guess.

**Status:** Not re-verified in this review pass. Needs explicit check.

**Trigger:** Any production deployment with `ACCORD_AUTH_DISABLED=false`.

---

## D7 — `DPOManager._threshold` private attribute in public API response

**File:** `accord_ai/api.py:2042`

**Risk:** Minor encapsulation violation. `mgr._threshold` accesses a private attribute
from outside its class. If `DPOManager` is refactored and `_threshold` is renamed, the
API response silently breaks.

**Fix:** Add a `.threshold` property to `DPOManager`.

---

## Review provenance

Identified by `superpowers:code-reviewer` subagent, 2026-04-22, during Step 25.A
variance diagnostic wait. Items D1–D5 were newly identified; D6 was pre-existing from
MEMORY.md. Fixes applied same session: D4 (partial), plus Critical items from the
review (mode-resolution bug, field-count bug, FREE mode fallback parser).
