"""Build the from-scratch notebooks of every Hale repository from their Markdown sources.

Layout (per repository)
-----------------------
``notebooks/_from_scratch_kit/``
    ``stepviz/``   the picture/animation kit (canonical copy in HaleBlocks, synced everywhere)
    ``sources/``   one ``<original notebook stem>.md`` per from-scratch notebook
    ``build.py``   this script (also synced)

A source named ``resnet.md`` is the twin of the repository's ``.../resnet.ipynb``; it is built to
``.../from_scratch/resnet_from_scratch.ipynb`` *next to* the original (a ``_step_by_step`` suffix is
dropped: ``pi0_step_by_step.md`` -> ``pi0_from_scratch.ipynb``). Sources in HaleBlocks'
``sources/shared/`` are for notebooks that exist in several repos (the backbone notebooks): they are
executed once and the executed notebook is copied next to every original with the same name.

Source format: ```` ```python ```` blocks are executed code cells, ```` ```py ```` blocks are
display-only code, ```` ```mermaid ```` blocks stay in markdown, and a line holding only ``+++``
starts a new markdown cell.

Usage (run from anywhere, with any Python that has nbformat + nbclient)::

    python HaleBlocks/notebooks/_from_scratch_kit/build.py                 # everything
    python .../build.py --repo Halos-lift                                  # one repository
    python .../build.py resnet vit                                         # by source stem
    python .../build.py --sync-only                                        # just sync kit + indexes

Each notebook runs with that repository's own virtualenv (falling back to Hale-VLM's, which has
PyTorch and Jupyter) and ``PYTHONPATH=<repo>/src:HaleBlocks/src`` so the optional "same as the
library?" cross-check cells can import the real implementation. A failing cell fails the build.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import nbformat
from jupyter_client.kernelspec import KernelSpec
from jupyter_client.manager import KernelManager
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

KIT = Path(__file__).resolve().parent
HALEBLOCKS = KIT.parents[1]
MUSIC = HALEBLOCKS.parent
REPOS = ["HaleBlocks", "Hale-VLM", "Hale-VLA", "HALE-WAM", "HaloDiffusionLLM", "Halos-lift"]
KIT_REL = Path("notebooks") / "_from_scratch_kit"
FALLBACK_VENV = "Hale-VLM"

CODE_BLOCK = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


@dataclass
class Job:
    """One source and every place its executed notebook must be written."""

    source: Path
    run_repo: str  # repository whose venv/PYTHONPATH executes it
    targets: List[Path]  # output .ipynb paths (first one is the execution working directory)


# ----------------------------------------------------------------------------------- parsing
def markdown_cells(text: str) -> List[nbformat.NotebookNode]:
    cells = []
    for chunk in re.split(r"^\+\+\+[ \t]*$", text, flags=re.MULTILINE):
        chunk = chunk.strip("\n")
        if chunk.strip():
            cells.append(nbformat.v4.new_markdown_cell(chunk))
    return cells


def parse_source(text: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    cursor = 0
    for m in CODE_BLOCK.finditer(text):
        nb.cells.extend(markdown_cells(text[cursor:m.start()]))
        nb.cells.append(nbformat.v4.new_code_cell(m.group(1).rstrip("\n")))
        cursor = m.end()
    nb.cells.extend(markdown_cells(text[cursor:]))
    return nb


# ----------------------------------------------------------------------------------- discovery
def tracked_notebooks(repo: str) -> List[Path]:
    """Original notebooks tracked by git (from-scratch outputs excluded)."""
    root = MUSIC / repo
    out = subprocess.run(["git", "-C", str(root), "ls-files", "*.ipynb"], capture_output=True,
                         text=True, check=True).stdout.split()
    return [root / p for p in out if "/from_scratch/" not in p and "_from_scratch_kit" not in p]


def output_name(stem: str) -> str:
    return re.sub(r"_step_by_step$", "", stem) + "_from_scratch.ipynb"


def plan(repos: List[str], stems: Optional[List[str]]) -> List[Job]:
    """Match every source to the originals it mirrors."""
    originals: Dict[str, Dict[str, List[Path]]] = {r: {} for r in REPOS}
    for repo in REPOS:
        for p in tracked_notebooks(repo):
            originals[repo].setdefault(p.stem, []).append(p)
    jobs: List[Job] = []
    shared_dir = HALEBLOCKS / KIT_REL / "sources" / "shared"
    for src in sorted(shared_dir.glob("*.md")):
        targets = [orig.parent / "from_scratch" / output_name(src.stem)
                   for repo in REPOS for orig in originals[repo].get(src.stem, [])]
        if not targets:
            print(f"  warn: shared source {src.name} matches no notebook", file=sys.stderr)
            continue
        jobs.append(Job(src, "HaleBlocks", targets))
    for repo in REPOS:
        src_dir = MUSIC / repo / KIT_REL / "sources"
        for src in sorted(src_dir.rglob("*.md")):
            if src.parent.name == "shared" and repo == "HaleBlocks":
                continue
            found = originals[repo].get(src.stem, [])
            if not found:
                print(f"  warn: {repo}: {src.name} matches no notebook", file=sys.stderr)
                continue
            jobs.append(Job(src, repo, [o.parent / "from_scratch" / output_name(src.stem)
                                        for o in found]))
    if repos:
        jobs = [j for j in jobs if j.run_repo in repos
                or any(str(t).startswith(str(MUSIC / r)) for r in repos for t in j.targets)]
    if stems:
        jobs = [j for j in jobs if any(j.source.stem.startswith(s) for s in stems)]
    return jobs


# ----------------------------------------------------------------------------------- running
def python_for(repo: str) -> str:
    """The repo's own venv if it has PyTorch + ipykernel, else the fallback venv."""
    for candidate in (repo, FALLBACK_VENV):
        py = MUSIC / candidate / ".venv" / "bin" / "python"
        if py.exists() and subprocess.run([str(py), "-c", "import torch, ipykernel"],
                                          capture_output=True).returncode == 0:
            return str(py)
    raise RuntimeError(f"no virtualenv with torch + ipykernel for {repo}")


def kernel_for(repo: str) -> KernelManager:
    """A kernel on the chosen interpreter (the user-level ``python3`` spec may point anywhere)."""
    km = KernelManager(kernel_name="python3")
    km._kernel_spec = KernelSpec(argv=[python_for(repo), "-m", "ipykernel_launcher", "-f",
                                       "{connection_file}"], display_name="Python 3",
                                 language="python")
    return km


def execute(job: Job, timeout: int) -> float:
    start = time.perf_counter()
    nb = parse_source(job.source.read_text(encoding="utf-8"))
    workdir = job.targets[0].parent
    workdir.mkdir(parents=True, exist_ok=True)
    # The kernel inherits this process' environment: put the repo's sources first so the guarded
    # cross-check cells import the *current* library code, not a stale installed copy.
    paths = [str(MUSIC / job.run_repo / "src"), str(HALEBLOCKS / "src")]
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    client = NotebookClient(nb, km=kernel_for(job.run_repo), timeout=timeout,
                            resources={"metadata": {"path": str(workdir)}})
    client.execute()
    for target in job.targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, target)
    return time.perf_counter() - start


# ----------------------------------------------------------------------------------- syncing
def sync_kit() -> None:
    """Copy the canonical stepviz + build.py into every other repository's kit folder."""
    for repo in REPOS:
        if repo == "HaleBlocks":
            continue
        dest = MUSIC / repo / KIT_REL
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(KIT / "stepviz", dest / "stepviz", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(KIT / "build.py", dest / "build.py")
        shutil.copy2(KIT / "README.md", dest / "README.md")


def write_indexes() -> None:
    """``notebooks/FROM_SCRATCH.md`` per repo: every original next to its from-scratch twin."""
    for repo in REPOS:
        root = MUSIC / repo
        rows = []
        for orig in sorted(tracked_notebooks(repo)):
            twin = orig.parent / "from_scratch" / output_name(orig.stem)
            if twin.exists():
                rel_o = orig.relative_to(root / "notebooks")
                rel_t = twin.relative_to(root / "notebooks")
                rows.append(f"| [{rel_o}]({rel_o}) | [{rel_t.name}]({rel_t}) |")
        if not rows:
            continue
        text = (
            "# From-scratch notebooks\n\n"
            "Every notebook below has a **from-scratch twin**: the same model or algorithm rebuilt in\n"
            "pure Python / PyTorch, one step at a time, with a picture, a Mermaid diagram and an\n"
            "animation for each step. The twins are generated from Markdown sources in\n"
            "`notebooks/_from_scratch_kit/sources/` (see the kit's README).\n\n"
            "| original | from scratch |\n|---|---|\n" + "\n".join(rows) + "\n"
        )
        (root / "notebooks" / "FROM_SCRATCH.md").write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------------- main
def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="*", help="only sources whose stem starts with these")
    ap.add_argument("--repo", action="append", default=[], help="only jobs touching this repo")
    ap.add_argument("--timeout", type=int, default=1200, help="per-cell timeout (s)")
    ap.add_argument("--sync-only", action="store_true", help="sync kit + indexes, build nothing")
    ap.add_argument("--list", action="store_true", help="print the build plan and exit")
    args = ap.parse_args(argv)

    if MUSIC / "HaleBlocks" / KIT_REL != KIT:
        print("run the canonical copy: HaleBlocks/notebooks/_from_scratch_kit/build.py",
              file=sys.stderr)
        return 2
    sync_kit()
    jobs = plan(args.repo, args.stems or None)
    if args.list:
        for j in jobs:
            print(f"{j.source.relative_to(MUSIC)}  ->  "
                  + ", ".join(str(t.relative_to(MUSIC)) for t in j.targets))
        return 0
    failures = []
    if not args.sync_only:
        for job in jobs:
            try:
                secs = execute(job, args.timeout)
                print(f"  ok   {job.source.stem:<48} {secs:6.1f}s  x{len(job.targets)}")
            except CellExecutionError as err:
                failures.append(job.source.stem)
                print(f"  FAIL {job.source.stem}\n" + "\n".join(str(err).splitlines()[-40:]),
                      file=sys.stderr)
    write_indexes()
    print(f"\n{len(jobs) - len(failures)}/{len(jobs)} built" if not args.sync_only else "synced")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
