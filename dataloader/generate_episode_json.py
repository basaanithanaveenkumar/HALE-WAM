"""Merge LeRobot ``meta/episodes/**/*.parquet`` files into one ``episodes.jsonl``.

Usage::

    python -m dataloader.generate_episode_json --meta_dir /data/airoa-moma/meta \
        --out /data/airoa-moma/episodes.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger


def merge_episode_metadata(meta_dir: Path, out: Path) -> int:
    """Concatenate every episode parquet under ``meta_dir/episodes`` into JSON lines at ``out``.

    Args:
        meta_dir: the dataset's ``meta`` directory.
        out: destination ``.jsonl`` file (overwritten).

    Returns:
        Number of episode rows written.

    Raises:
        FileNotFoundError: if no episode parquet file exists under ``meta_dir``.
    """
    import pandas as pd  # lazy: only this tool needs pandas

    files = sorted(meta_dir.glob("episodes/**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet files under {meta_dir / 'episodes'}")
    frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_json(out, orient="records", lines=True, force_ascii=False)
    logger.info("wrote {} rows to {} ({} bytes)", len(frame), out, out.stat().st_size)
    return len(frame)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--meta_dir", type=Path, required=True, help="dataset meta/ directory")
    parser.add_argument("--out", type=Path, required=True, help="output episodes.jsonl")
    args = parser.parse_args(argv)
    merge_episode_metadata(args.meta_dir, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
