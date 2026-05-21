"""
main_agent.py — Orchestrator agent.

Loop:
  1. Run test_agent.py → collect errors
  2. If no errors → done
  3. Call Claude API with file contents + errors → receive fixes
  4. Apply fixes (with .bak backup)
  5. Repeat up to MAX_ITERATIONS

Requires:  pip install anthropic
           ANTHROPIC_API_KEY env variable
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Error: 'anthropic' package not found. Run: pip install anthropic")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent
TEST_AGENT   = PROJECT_ROOT / "test_agent.py"
MAX_ITERATIONS = 5
MODEL = "claude-sonnet-4-6"


# ── Subprocess helpers ────────────────────────────────────────────────────────

def run_test_agent() -> tuple[dict, int]:
    """Run test_agent.py; return (parsed_report, exit_code)."""
    result = subprocess.run(
        [sys.executable, str(TEST_AGENT)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError:
        raw = (result.stdout + result.stderr).strip()
        report = {
            "errors": [{"file": "?", "type": "AgentRunError", "line": None,
                        "message": raw or "test_agent produced no output", "text": ""}],
            "warnings": [], "passed": [], "files_checked": [],
            "summary": {"total_files": 0, "passed": 0, "errors": 1, "warnings": 0},
        }
    return report, result.returncode


# ── File I/O ──────────────────────────────────────────────────────────────────

def read_file(rel_path: str) -> str:
    try:
        return (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
    except Exception as e:
        return f"[Cannot read: {e}]"


def backup_and_write(rel_path: str, content: str) -> None:
    full = PROJECT_ROOT / rel_path
    if full.exists():
        shutil.copy2(full, full.with_suffix(full.suffix + ".bak"))
    full.write_text(content, encoding="utf-8")
    print(f"  [fixed]  {rel_path}")


# ── Claude API call ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python code fixer embedded in an automated repair loop.
You receive a set of Python files with reported errors.
You must return ONLY a valid JSON array — no explanations, no markdown fences — with this structure:

[
  {"file": "relative/path.py", "content": "...complete fixed file content..."},
  ...
]

Rules:
- Include ONLY files you actually changed.
- Preserve all existing logic; fix only what is reported.
- For ImportError on a local name: check if the name was renamed or if the import path is wrong.
- For torch.load DeprecationWarning: add weights_only=False to keep current behaviour.
- Never add, remove, or rename features beyond what is required to fix the errors."""


def build_user_message(errors: list[dict], warnings: list[dict]) -> str:
    # Group errors by file
    by_file: dict[str, list[dict]] = {}
    for e in errors:
        by_file.setdefault(e["file"], []).append(e)
    # Also include files that only have warnings (if they exist)
    for w in warnings:
        if w["file"] not in by_file:
            by_file.setdefault(w["file"], [])  # will show warnings only

    sections: list[str] = []
    for rel_path, file_errors in by_file.items():
        content = read_file(rel_path)
        err_block = "\n".join(
            f"  Line {e.get('line', '?')}: [{e['type']}] {e['message']}"
            for e in file_errors
        ) or "  (no errors — warnings only)"
        file_warnings = [w for w in warnings if w["file"] == rel_path]
        warn_block = ""
        if file_warnings:
            warn_block = "\nWarnings:\n" + "\n".join(
                f"  Line {w.get('line', '?')}: [{w['type']}] {w['message']}"
                for w in file_warnings
            )
        sections.append(
            f"### {rel_path}\n"
            f"Errors:\n{err_block}"
            f"{warn_block}\n\n"
            f"```python\n{content}\n```"
        )

    return "Fix the following Python files:\n\n" + "\n\n---\n\n".join(sections)


def call_claude(client: anthropic.Anthropic, errors: list[dict], warnings: list[dict]) -> list[dict]:
    user_msg = build_user_message(errors, warnings)

    response = client.messages.create(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    fixes: list[dict] = json.loads(raw)
    return fixes


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    print("=" * 60)
    print("MAIN AGENT — Automated Code Test & Fix Loop")
    print(f"Model : {MODEL}  |  Max iterations : {MAX_ITERATIONS}")
    print("=" * 60)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n[Iteration {iteration}/{MAX_ITERATIONS}] Running test_agent.py …")
        report, exit_code = run_test_agent()
        s = report.get("summary", {})
        print(f"  Files: {s.get('total_files', '?')}  |  "
              f"Passed: {s.get('passed', '?')}  |  "
              f"Errors: {s.get('errors', '?')}  |  "
              f"Warnings: {s.get('warnings', '?')}")

        errors   = report.get("errors", [])
        warnings = report.get("warnings", [])

        if not errors:
            print("\nAll checks passed — no errors found.")
            if warnings:
                print(f"  {len(warnings)} non-blocking warning(s) remain.")
            print("Done.")
            sys.exit(0)

        # Print error summary
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            print(f"    [{e['type']}] {e['file']}:{e.get('line', '?')} — {e['message']}")

        if iteration == MAX_ITERATIONS:
            print(f"\nMax iterations reached ({MAX_ITERATIONS}). Errors remain — manual review needed.")
            sys.exit(1)

        # Ask Claude to fix
        affected_files = len({e["file"] for e in errors})
        print(f"\n  Sending {len(errors)} error(s) across {affected_files} file(s) to Claude …")
        try:
            fixes = call_claude(client, errors, warnings)
        except json.JSONDecodeError as exc:
            print(f"  Claude returned unparseable JSON: {exc}")
            sys.exit(1)
        except anthropic.APIError as exc:
            print(f"  Claude API error: {exc}")
            sys.exit(1)

        if not fixes:
            print("  Claude returned no fixes — stopping.")
            sys.exit(1)

        print(f"  Applying {len(fixes)} fix(es) (originals backed up as *.bak):")
        for fix in fixes:
            path = fix.get("file", "").strip()
            content = fix.get("content", "").strip()
            if path and content:
                backup_and_write(path, content)
            else:
                print(f"  [skip] Malformed fix entry (no file or content): {list(fix.keys())}")

    print("\nLoop ended.")


if __name__ == "__main__":
    main()
