"""Check source size, Python syntax and local Markdown targets."""

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def source_errors(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    label = str(path.relative_to(ROOT))
    errors = []
    if len(text.splitlines()) > 300:
        errors.append(f"{label}: exceeds 300 lines")
    if path.suffix == ".py":
        tree = ast.parse(text, filename=label)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.end_lineno - node.lineno + 1 > 90:
                    errors.append(f"{label}:{node.lineno}: function exceeds 90 lines")
    return errors


def link_errors(path: Path) -> list[str]:
    errors = []
    for match in re.finditer(r"\[[^\]\n]+\]\(([^)\n]+)\)", path.read_text(encoding="utf-8")):
        target = match.group(1).split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        if not (path.parent / target).exists():
            errors.append(f"{path.relative_to(ROOT)}: missing link {target}")
    return errors


def main() -> None:
    errors = []
    paths = []
    for directory in ("src", "bench", "tests", "scripts", "examples"):
        paths.extend((ROOT / directory).rglob("*.rs"))
        paths.extend((ROOT / directory).rglob("*.py"))
    for path in sorted(paths):
        errors.extend(source_errors(path))
    documents = list(ROOT.glob("*.md")) + list((ROOT / "docs").rglob("*.md"))
    for path in sorted(documents):
        errors.extend(link_errors(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Structure OK: {len(paths)} source files and {len(documents)} documents")


if __name__ == "__main__":
    main()
