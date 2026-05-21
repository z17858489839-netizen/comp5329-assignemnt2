"""
test_agent.py — Static analysis + syntax checking for all project Python files.

Checks performed (no ML deps required):
  1. py_compile  — byte-compile syntax check via subprocess
  2. AST parse   — pinpoints exact line/column of syntax errors
  3. AST static  — undefined local imports, bare excepts, common typos
  4. Cross-file  — verifies locally imported names actually exist in source

Output: JSON to stdout, human summary to stderr.
Exit 0 = all clear, Exit 1 = errors found.
"""

import ast
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent
SKIP_FILES = {"test_agent.py", "main_agent.py"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_source(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def make_error(file: str, kind: str, line: int | None, msg: str, text: str = "") -> dict:
    return {"file": file, "type": kind, "line": line, "message": msg, "text": text}


def make_warning(file: str, kind: str, line: int | None, msg: str) -> dict:
    return {"file": file, "type": kind, "line": line, "message": msg}


# ── Check 1: py_compile via subprocess ───────────────────────────────────────

def check_py_compile(path: Path) -> list[dict]:
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return []
    return [make_error(rel(path), "SyntaxError", None, result.stderr.strip())]


# ── Check 2: AST parse (detailed location) ───────────────────────────────────

def check_ast_parse(path: Path, source: str) -> tuple[ast.Module | None, list[dict]]:
    try:
        tree = ast.parse(source, filename=str(path))
        return tree, []
    except SyntaxError as e:
        err = make_error(rel(path), "SyntaxError", e.lineno, e.msg, e.text or "")
        return None, [err]
    except Exception as e:
        return None, [make_error(rel(path), type(e).__name__, None, str(e))]


# ── Check 3: AST static analysis ─────────────────────────────────────────────

KNOWN_TYPOS: dict[str, str] = {
    "tourch": "torch",
    "numppy": "numpy",
    "scikitlearn": "sklearn",
    "PIL.image": "PIL.Image",
    "transofrmers": "transformers",
}

def check_ast_static(path: Path, tree: ast.Module) -> list[dict]:
    issues: list[dict] = []
    f = rel(path)

    for node in ast.walk(tree):
        # Bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append(make_warning(
                f, "BareExcept", node.lineno,
                "Bare 'except:' swallows all exceptions including SystemExit/KeyboardInterrupt",
            ))

        # Import typos
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module] if node.module else [])
            )
            for name in names:
                if name in KNOWN_TYPOS:
                    issues.append(make_warning(
                        f, "TypoWarning", node.lineno,
                        f"Possible typo: '{name}' — did you mean '{KNOWN_TYPOS[name]}'?",
                    ))

        # torch.load without weights_only (deprecation in torch >= 2.0)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "torch"):
            kwarg_names = {kw.arg for kw in node.keywords}
            if "weights_only" not in kwarg_names:
                issues.append(make_warning(
                    f, "DeprecationWarning", node.lineno,
                    "torch.load() without weights_only=True is deprecated in torch >= 2.0; "
                    "add weights_only=False to suppress or weights_only=True for safety",
                ))

    return issues


# ── Check 4: Cross-file local import validation ───────────────────────────────

def collect_exported_names(path: Path) -> set[str]:
    """Return all top-level names defined in a Python file."""
    source = read_source(path)
    if source is None:
        return set()
    try:
        tree = ast.parse(source)
    except Exception:
        return set()
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname if alias.asname else alias.name.split(".")[0])
    return names


def module_path_for(module_str: str) -> Path | None:
    """Convert a dotted local module string to a Path within PROJECT_ROOT."""
    parts = module_str.split(".")
    candidate = PROJECT_ROOT.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    pkg = PROJECT_ROOT.joinpath(*parts, "__init__.py")
    if pkg.exists():
        return pkg
    return None


def check_local_imports(path: Path, tree: ast.Module, all_exports: dict[str, set[str]]) -> list[dict]:
    issues: list[dict] = []
    f = rel(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        # Only check imports that look local (no dots meaning package-relative, or known local packages)
        local_root = module.split(".")[0] if module else ""
        if local_root not in ("config", "models", "utils", "training", "evaluation", "experiments"):
            continue
        mod_path = module_path_for(module)
        if mod_path is None:
            continue  # can't resolve — skip
        exported = all_exports.get(str(mod_path.relative_to(PROJECT_ROOT)), set())
        if not exported:
            continue  # file not parsed yet — skip
        for alias in node.names:
            name = alias.name
            if name == "*":
                continue
            if name not in exported:
                issues.append(make_error(
                    f, "ImportError", node.lineno,
                    f"'{name}' not found in '{module}' — available: {sorted(exported)}",
                ))
    return issues


# ── Runner ────────────────────────────────────────────────────────────────────

def collect_python_files() -> list[Path]:
    files = []
    for p in sorted(PROJECT_ROOT.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        if p.name in SKIP_FILES:
            continue
        files.append(p)
    return files


def run_all_checks() -> dict[str, Any]:
    files = collect_python_files()
    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "files_checked": [rel(f) for f in files],
        "errors": [],
        "warnings": [],
        "passed": [],
    }

    # First pass: parse all files and collect exported names
    parsed_trees: dict[str, ast.Module | None] = {}
    all_exports: dict[str, set[str]] = {}
    for path in files:
        source = read_source(path)
        if source is None:
            report["errors"].append(make_error(rel(path), "ReadError", None, "Cannot read file"))
            continue
        tree, parse_errors = check_ast_parse(path, source)
        parsed_trees[rel(path)] = tree
        if parse_errors:
            report["errors"].extend(parse_errors)
        if tree is not None:
            all_exports[rel(path)] = collect_exported_names(path)

    # Second pass: full checks
    for path in files:
        f = rel(path)
        tree = parsed_trees.get(f)

        # 1. py_compile
        compile_errors = check_py_compile(path)
        report["errors"].extend(compile_errors)
        if compile_errors:
            continue  # skip further checks if syntax is broken

        if tree is None:
            continue

        # 2. AST static
        static_warnings = check_ast_static(path, tree)
        report["warnings"].extend(static_warnings)

        # 3. Cross-file local imports
        import_errors = check_local_imports(path, tree, all_exports)
        report["errors"].extend(import_errors)

        if not compile_errors and not import_errors:
            report["passed"].append(f)

    report["summary"] = {
        "total_files": len(files),
        "passed":   len(report["passed"]),
        "errors":   len(report["errors"]),
        "warnings": len(report["warnings"]),
    }
    return report


def main():
    report = run_all_checks()
    print(json.dumps(report, indent=2))

    s = report["summary"]
    print(f"\n{'='*55}", file=sys.stderr)
    print(f"Files checked : {s['total_files']}", file=sys.stderr)
    print(f"Passed        : {s['passed']}", file=sys.stderr)
    print(f"Errors        : {s['errors']}", file=sys.stderr)
    print(f"Warnings      : {s['warnings']}", file=sys.stderr)
    print(f"{'='*55}", file=sys.stderr)

    if report["errors"]:
        print("\nErrors:", file=sys.stderr)
        for e in report["errors"]:
            print(f"  [{e['type']}] {e['file']}:{e.get('line','?')} — {e['message']}",
                  file=sys.stderr)

    sys.exit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
