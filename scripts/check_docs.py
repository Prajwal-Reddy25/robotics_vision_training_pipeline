"""Validate repository Markdown/HTML file links and simple heading anchors offline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def prose(text):
    lines, fence = [], None
    for line in text.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            char = marker[1][0]
            fence = None if fence == char else char if fence is None else fence
        elif fence is None:
            lines.append(line)
    return "\n".join(lines)


def anchors(text):
    result, counts = set(), {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*$", prose(text), flags=re.MULTILINE):
        slug = re.sub(r"<[^>]+>", "", heading).lower()
        slug = re.sub(r"[^\w\- ]", "", slug).replace(" ", "-")
        n = counts.get(slug, 0)
        counts[slug] = n + 1
        result.add(f"{slug}-{n}" if n else slug)
    result.update(re.findall(r'\bid=["\']([^"\']+)["\']', text))
    return result


def check(root):
    root = Path(root).resolve()
    files = [root / name for name in ("README.md", "CHANGELOG.md", "ROADMAP.md")]
    files += sorted((root / "docs").rglob("*.md"))
    failures, checked = [], 0
    for path in files:
        text = prose(path.read_text(encoding="utf-8"))
        targets = re.findall(r"\]\(([^\n)]+)\)", text)
        targets += re.findall(r'\b(?:src|href)=["\']([^"\']+)["\']', text)
        for raw in targets:
            raw = raw.strip().split(' "', 1)[0].strip("<>")
            url = urlsplit(raw)
            if url.scheme or url.netloc:
                continue  # External availability/terms are not an offline claim.
            checked += 1
            target = ((path.parent / unquote(url.path)) if url.path else path).resolve()
            label = f"{path.relative_to(root)}: {raw}"
            if not target.is_relative_to(root) or not target.is_file():
                failures.append(label + " (missing or outside repository)")
            elif url.fragment and target.suffix == ".md":
                if unquote(url.fragment) not in anchors(target.read_text(encoding="utf-8")):
                    failures.append(label + " (missing heading anchor)")
    return {"markdown_files": len(files), "local_links_checked": checked, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = check(args.repo_root)
    for error in report["failures"]:
        print(error)
    print(
        f"Checked {report['markdown_files']} Markdown files and "
        f"{report['local_links_checked']} local links; {len(report['failures'])} failures"
    )
    return bool(report["failures"])


if __name__ == "__main__":
    raise SystemExit(main())
