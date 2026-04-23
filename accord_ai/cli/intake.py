"""Intake CLI — interactive REPL + batch scripting + session management.

Interactive:
    python -m accord_ai.cli.intake
    python -m accord_ai.cli.intake --tenant acme
    python -m accord_ai.cli.intake --session-id abc123   # resume existing

Batch:
    python -m accord_ai.cli.intake --script messages.txt

Session management:
    python -m accord_ai.cli.intake --list                # list all
    python -m accord_ai.cli.intake --list --tenant acme  # tenant-filtered

REPL commands:
    /help      — show commands
    /status    — show current submission state (via explain)
    /finalize  — finalize session and exit
    /quit      — exit without finalizing

Startup: the CLI verifies the LLM endpoint is reachable before entering
the REPL. If not, prints an actionable error with the startup command.
Use --no-health-check to skip (tests, CI).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import openai

from accord_ai.app import IntakeApp, build_intake_app
from accord_ai.config import Settings
from accord_ai.conversation.controller import TurnResult
from accord_ai.conversation.explainer import explain


_HELP_TEXT = """\
REPL commands:
  /help       show this help
  /status     show current submission state
  /finalize   finalize the session and exit
  /quit       exit without finalizing
"""


async def _check_llm_health(
    settings: Settings, *, timeout_s: float = 5.0
) -> Optional[str]:
    """Return None if healthy, else an error detail string."""
    client = openai.AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key="sk-health-check",
        timeout=timeout_s,
        max_retries=0,
    )
    try:
        await client.models.list()
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def _print_health_error(settings: Settings, detail: str) -> None:
    print(f"[error] LLM endpoint not reachable at {settings.llm_base_url}", file=sys.stderr)
    print(f"[error] {detail}", file=sys.stderr)
    print("[error] Start it first:", file=sys.stderr)
    print("[error]   cd /path/to/accord_ai_v3 && ./run.sh vllm", file=sys.stderr)
    print(
        "[error]   # or set LLM_BASE_URL to an OpenAI-compat endpoint you've started",
        file=sys.stderr,
    )


def list_sessions(app: IntakeApp, *, tenant: Optional[str] = None) -> None:
    """Print active+finalized sessions, newest first. Tenant-filtered if given."""
    sessions = app.store.list_sessions(tenant=tenant)
    if not sessions:
        print("[no sessions]")
        return
    print(f"{'session_id':36}  {'status':10}  {'tenant':16}  updated_at")
    print("-" * 90)
    for s in sessions:
        t = s.tenant or "-"
        print(f"{s.session_id}  {s.status:10}  {t:16}  {s.updated_at.isoformat()}")


async def run_interactive(
    app: IntakeApp,
    *,
    session_id: Optional[str] = None,
    tenant: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Interactive REPL. Session resume, /commands, graceful engine errors."""
    # Session — new or resumed
    if session_id:
        existing = app.store.get_session(session_id, tenant=tenant)
        if existing is None:
            print(f"[error] session not found: {session_id}", file=sys.stderr)
            return
        if existing.status != "active":
            print(
                f"[error] session {session_id} is {existing.status!r} — can't resume",
                file=sys.stderr,
            )
            return
        sid = session_id
        print(f"[session {sid} resumed]")
    else:
        sid = app.store.create_session(tenant=tenant)
        print(f"[session {sid} created]")

    # Initial greeting — responder reads current (possibly empty) submission
    session = app.store.get_session(sid, tenant=tenant)
    if session is not None:
        initial_verdict = app.judge.evaluate(session.submission)
        greeting = await app.responder.respond(
            submission=session.submission,
            verdict=initial_verdict,
        )
        print(f"\nAssistant: {greeting}\n")

    while True:
        try:
            user_msg = input("You: ").strip()
        except EOFError:
            print("\n[eof — exiting]")
            return

        if not user_msg:
            continue

        cmd = user_msg.lower()

        if cmd in ("/quit", "/q", "/exit"):
            print("[quit]")
            return

        if cmd == "/help":
            print(_HELP_TEXT)
            continue

        if cmd == "/status":
            session = app.store.get_session(sid, tenant=tenant)
            if session is None:
                print("[error] session disappeared", file=sys.stderr)
                return
            print("\n[current state]")
            print(explain(session.submission))
            print()
            continue

        if cmd == "/finalize":
            try:
                app.store.finalize_session(sid, tenant=tenant)
                print(f"[session {sid} finalized]")
                return
            except ValueError as e:
                print(f"[cannot finalize: {e}]")
                continue

        # Normal turn — catch engine + unexpected errors, keep session alive
        try:
            result = await app.controller.process_turn(
                session_id=sid, user_message=user_msg, tenant=tenant,
            )
        except openai.APIError as e:
            print(f"\n[engine error] {type(e).__name__}: {e}")
            print("[session still active — try again]\n")
            continue
        except Exception as e:
            print(f"\n[unexpected error] {type(e).__name__}: {e}")
            print("[session still active — try again]\n")
            continue

        print(f"\nAssistant: {result.assistant_message}\n")

        if debug:
            print(
                f"[debug] verdict.passed={result.verdict.passed} "
                f"failed_paths={list(result.verdict.failed_paths)}"
            )

        if result.is_complete:
            print("[intake complete]")
            try:
                confirm = input("Finalize now? [y/N]: ").strip().lower()
            except EOFError:
                confirm = ""
            if confirm in ("y", "yes"):
                app.store.finalize_session(sid, tenant=tenant)
                print(f"[session {sid} finalized]")
                return
            else:
                print("[continuing — more turns allowed]")


async def run_scripted(
    app: IntakeApp,
    messages: List[str],
    *,
    tenant: Optional[str] = None,
    max_turns: Optional[int] = None,
    print_explainer: bool = True,
    debug: bool = False,
) -> List[TurnResult]:
    """Run messages through a fresh session. Returns TurnResults."""
    sid = app.store.create_session(tenant=tenant)
    print(f"[session {sid} created]")

    results: List[TurnResult] = []
    for i, msg in enumerate(messages):
        if max_turns is not None and i >= max_turns:
            print(f"[max-turns {max_turns} reached]")
            break

        print(f"\nTurn {i + 1}: {msg}")
        result = await app.controller.process_turn(
            session_id=sid, user_message=msg, tenant=tenant,
        )
        results.append(result)
        print(f"Assistant: {result.assistant_message}")

        if debug:
            print(
                f"[debug] verdict.passed={result.verdict.passed} "
                f"failed_paths={list(result.verdict.failed_paths)}"
            )

        if result.is_complete:
            print(f"\n[intake complete after turn {i + 1}]")
            break

    if print_explainer:
        session = app.store.get_session(sid, tenant=tenant)
        if session is not None:
            print("\n[final state]")
            print(explain(session.submission))

    return results


def _load_script_messages(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="accord-intake",
        description="Accord AI insurance intake CLI.",
    )
    parser.add_argument("--script", type=str, default=None,
                        help="batch mode: read user messages from file")
    parser.add_argument("--session-id", type=str, default=None,
                        help="resume an existing session by id (interactive mode)")
    parser.add_argument("--tenant", type=str, default=None,
                        help="tenant identifier")
    parser.add_argument("--max-turns", type=int, default=50,
                        help="safety cap on batch-mode turn count")
    parser.add_argument("--list", action="store_true",
                        help="list sessions and exit (optionally --tenant-filtered)")
    parser.add_argument("--debug", action="store_true",
                        help="print verdict diagnostics after each turn")
    parser.add_argument("--no-health-check", action="store_true",
                        help="skip LLM endpoint health check at startup")
    args = parser.parse_args(argv)

    settings = Settings()

    # Listing doesn't need the LLM — skip health check
    if args.list:
        app = build_intake_app(settings)
        list_sessions(app, tenant=args.tenant)
        return 0

    # Health check (unless explicitly skipped)
    if not args.no_health_check:
        detail = asyncio.run(_check_llm_health(settings))
        if detail is not None:
            _print_health_error(settings, detail)
            return 1

    app = build_intake_app(settings)

    if args.script:
        script_path = Path(args.script)
        if not script_path.exists():
            print(f"[error] script file not found: {script_path}", file=sys.stderr)
            return 1
        messages = _load_script_messages(script_path)
        asyncio.run(run_scripted(
            app, messages,
            tenant=args.tenant,
            max_turns=args.max_turns,
            debug=args.debug,
        ))
    else:
        try:
            asyncio.run(run_interactive(
                app,
                session_id=args.session_id,
                tenant=args.tenant,
                debug=args.debug,
            ))
        except KeyboardInterrupt:
            print("\n[interrupted]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
