#!/usr/bin/env python3
"""
Prepare a reviewer-friendly checkout.

This script:
- moves raw/full data and heavy artifacts into ./archive (gitignored)
- writes small *sample* versions back into ./dataset so reviewers can run code
  without access to the full restricted dataset.

It is intentionally conservative: it never deletes; it only moves and rewrites
small samples.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def move_path(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    ensure_parent(dst)
    if dry_run:
        print(f"[dry-run] move {src} -> {dst}")
        return
    shutil.move(str(src), str(dst))


def copy_path(src: Path, dst: Path, dry_run: bool) -> None:
    if not src.exists():
        return
    ensure_parent(dst)
    if dry_run:
        print(f"[dry-run] copy {src} -> {dst}")
        return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def write_sample_jsonl(src: Path, dst: Path, n_lines: int, dry_run: bool) -> None:
    if not src.exists():
        return
    ensure_parent(dst)
    if dry_run:
        print(f"[dry-run] sample(jsonl) {src} -> {dst} (lines={n_lines})")
        return
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if i >= n_lines:
                break
            fout.write(line)


def write_sample_csv(src: Path, dst: Path, n_rows: int, dry_run: bool) -> None:
    """
    Keep header + first n_rows data rows.
    """
    if not src.exists():
        return
    ensure_parent(dst)
    if dry_run:
        print(f"[dry-run] sample(csv) {src} -> {dst} (rows={n_rows})")
        return
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            # header is i == 0; keep up to n_rows data lines => total n_rows + 1 lines
            if i > n_rows:
                break
            fout.write(line)


def archive_outputs_checkpoint(repo_root: Path, dry_run: bool) -> None:
    outputs_checkpoint = repo_root / "outputs" / "checkpoint"
    if not outputs_checkpoint.exists():
        return

    archive_checkpoint_full = repo_root / "archive" / "outputs" / "checkpoint_full"
    summary_src = outputs_checkpoint / "summary.csv"

    summary_tmp = None
    if summary_src.exists():
        summary_tmp = repo_root / "archive" / "tmp_checkpoint_summary.csv"
        copy_path(summary_src, summary_tmp, dry_run=dry_run)

    move_path(outputs_checkpoint, archive_checkpoint_full, dry_run=dry_run)

    # Recreate lightweight outputs/checkpoint with summary.csv only.
    new_checkpoint_dir = repo_root / "outputs" / "checkpoint"
    if dry_run:
        print(f"[dry-run] mkdir {new_checkpoint_dir}")
    else:
        new_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if summary_tmp and summary_tmp.exists():
        copy_path(summary_tmp, new_checkpoint_dir / "summary.csv", dry_run=dry_run)
        if not dry_run:
            summary_tmp.unlink(missing_ok=True)


def archive_outputs_jsonl(repo_root: Path, dry_run: bool) -> None:
    outputs_dir = repo_root / "outputs"
    if not outputs_dir.exists():
        return

    for path in outputs_dir.rglob("*.jsonl"):
        # Skip anything already in archive/ (paranoia) and the checkpoint summary dir.
        if "archive" in path.parts:
            continue
        rel = path.relative_to(repo_root)
        dst = repo_root / "archive" / "outputs" / "jsonl" / rel
        move_path(path, dst, dry_run=dry_run)


def archive_dataset_and_write_samples(repo_root: Path, sample_lines: int, sample_rows: int, dry_run: bool) -> None:
    dataset_dir = repo_root / "dataset"
    if not dataset_dir.exists():
        return

    # Move raw spreadsheet sources.
    for xlsx in [dataset_dir / "抖音.xlsx", dataset_dir / "微信视频号.xlsx"]:
        move_path(xlsx, repo_root / "archive" / "dataset" / xlsx.name, dry_run=dry_run)

    # Move unified_3platform full exports and write small samples back.
    for fname, writer in [
        ("unified_3platform.csv", lambda s, d: write_sample_csv(s, d, n_rows=sample_rows, dry_run=dry_run)),
        ("unified_3platform.jsonl", lambda s, d: write_sample_jsonl(s, d, n_lines=sample_lines, dry_run=dry_run)),
    ]:
        src = dataset_dir / fname
        if not src.exists():
            continue
        archived = repo_root / "archive" / "dataset" / fname
        move_path(src, archived, dry_run=dry_run)
        # write sample back
        writer(archived, src)

    # Move lf_data full set and write samples back.
    lf_data = dataset_dir / "lf_data"
    if lf_data.exists():
        archived_lf = repo_root / "archive" / "dataset" / "lf_data_full"
        move_path(lf_data, archived_lf, dry_run=dry_run)

        # Recreate structure and write sample jsonl files; copy small json metadata files.
        if dry_run:
            print(f"[dry-run] recreate {lf_data}")
        else:
            lf_data.mkdir(parents=True, exist_ok=True)

        for src in archived_lf.rglob("*"):
            if src.is_dir():
                continue
            rel = src.relative_to(archived_lf)
            dst = lf_data / rel
            if src.suffix == ".jsonl":
                write_sample_jsonl(src, dst, n_lines=sample_lines, dry_run=dry_run)
            elif src.suffix == ".json":
                copy_path(src, dst, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare reviewer-friendly repo (archive heavy/raw; keep samples).")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files.")
    parser.add_argument("--sample-lines", type=int, default=50, help="Sample lines for JSONL files.")
    parser.add_argument("--sample-rows", type=int, default=50, help="Sample data rows for CSV (header included).")
    args = parser.parse_args()

    # Ensure archive root exists (even if ignored by git).
    archive_root = REPO_ROOT / "archive"
    if args.dry_run:
        print(f"[dry-run] mkdir {archive_root}")
    else:
        archive_root.mkdir(parents=True, exist_ok=True)

    archive_outputs_checkpoint(REPO_ROOT, dry_run=args.dry_run)
    archive_outputs_jsonl(REPO_ROOT, dry_run=args.dry_run)
    archive_dataset_and_write_samples(
        REPO_ROOT, sample_lines=args.sample_lines, sample_rows=args.sample_rows, dry_run=args.dry_run
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

