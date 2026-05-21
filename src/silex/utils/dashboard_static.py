"""Resolve the directory FastAPI StaticFiles should use for the pre-built dashboard."""

from __future__ import annotations

from pathlib import Path


def resolve_dashboard_dir(*, packaged_aria_pkg: Path, repo_root_from_scripts: Path) -> Path | None:
    """Pick dashboard static root: packaged web_dist/, else aria-ui/out/ in repo checkout."""
    # 1. Packaged under aria
    wd = packaged_aria_pkg / "web_dist"
    if (wd / "index.html").is_file():
        return wd

    # 2. Packaged under vyn_app (next to web_server or passed repo_root)
    vyn_wd = repo_root_from_scripts / "vyn_app" / "web_dist"
    if (vyn_wd / "index.html").is_file():
        return vyn_wd

    # Also support nested layout under products/vyn/src/vyn_app/web_dist
    vyn_wd_alt = repo_root_from_scripts / "products" / "vyn" / "src" / "vyn_app" / "web_dist"
    if (vyn_wd_alt / "index.html").is_file():
        return vyn_wd_alt

    # 3. Development checkout - scan up to find the repository root containing packages/aria-ui
    repo_root = None
    current = repo_root_from_scripts.resolve()
    for _ in range(10):
        if (current / "packages" / "aria-ui").is_dir():
            repo_root = current
            break
        if current.parent == current:
            break
        current = current.parent

    if repo_root:
        out = repo_root / "packages" / "aria-ui" / "out"
        if (out / "index.html").is_file():
            return out

    # Legacy fallback for tests
    legacy_out = repo_root_from_scripts / "aria-ui" / "out"
    if (legacy_out / "index.html").is_file():
        return legacy_out

    return None
