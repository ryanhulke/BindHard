from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from common import resolve_path, score_bound_pose
from datamodules import CrossDockedLMDBDataset
from common import DEFAULT_GUIDANCE_TARGET, read_label_sidecar, vina_score_metadata
from reconstruct_molecule import build_reference_mol

_WORKER_DATASET: CrossDockedLMDBDataset | None = None
_WORKER_SPLIT: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score CrossDocked LMDB samples with AutoDock Vina score-only mode."
    )
    parser.add_argument(
        "--lmdb",
        default="data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb",
        help="Processed CrossDocked LMDB path.",
    )
    parser.add_argument(
        "--split-path",
        default="data/crossdocked_pose_split_from_name_val1000.pt",
        help="Torch split file path.",
    )
    parser.add_argument(
        "--output",
        default="data/crossdocked_vina_score_labels.jsonl",
        help="Output JSONL sidecar path.",
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val"),
        help="Split(s) to score. Defaults to train and val.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional per-split record cap. 0 means no limit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing sidecar file and skip completed sample keys.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shard count for distributed scoring.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index in [0, num_shards).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Worker process count for parallel scoring.",
    )
    return parser.parse_args()


def load_existing_sample_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    _, records = read_label_sidecar(path)
    return set(records)


def ensure_output_file(path: Path, *, resume: bool) -> None:
    if path.exists() and not resume:
        raise FileExistsError(f"output exists: {path}. Pass --resume to append and skip completed keys.")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(json.dumps(vina_score_metadata()) + "\n", encoding="utf-8")


def init_worker_dataset(lmdb_path: str, split_path: str, split: str) -> None:
    global _WORKER_DATASET, _WORKER_SPLIT
    _WORKER_DATASET = CrossDockedLMDBDataset(
        lmdb_path=lmdb_path,
        split_pt_path=split_path,
        split=split,
        center="protein_mean",
        return_text_fields=True,
    )
    _WORKER_SPLIT = split


def score_split_index(split_index: int) -> dict[str, Any]:
    if _WORKER_DATASET is None:
        raise RuntimeError("worker dataset is not initialized")

    sample = _WORKER_DATASET[split_index]
    sample_key = sample["key"].decode("utf-8")
    record = {
        "kind": "sample",
        "sample_key": sample_key,
        "split": _WORKER_SPLIT,
        "split_index": split_index,
        "status": "error",
        DEFAULT_GUIDANCE_TARGET: None,
        "error": None,
    }

    try:
        with tempfile.TemporaryDirectory(prefix="bindhard_vina_score_") as temp_dir_name:
            value = score_bound_pose(
                protein={
                    "protein_pos": sample["protein_pos"],
                    "protein_element": sample["protein_element"],
                },
                mol=build_reference_mol(sample),
                ligand_pos=sample["ligand_pos"],
                work_dir=temp_dir_name,
            )
        if not math.isfinite(value):
            raise ValueError(f"vina score is not finite: {value}")

        record["status"] = "ok"
        record[DEFAULT_GUIDANCE_TARGET] = value
    except Exception as exc:
        record["error"] = str(exc)

    return record


def collect_split_tasks(
    *,
    dataset: CrossDockedLMDBDataset,
    split: str,
    existing_keys: set[str],
    limit: int,
    num_shards: int,
    shard_index: int,
) -> list[tuple[int, str]]:
    tasks: list[tuple[int, str]] = []
    for split_index in tqdm(range(len(dataset)), desc=f"scan {split}"):
        if split_index % num_shards != shard_index:
            continue

        sample = dataset[split_index]
        sample_key = sample["key"].decode("utf-8")
        if sample_key in existing_keys:
            continue

        tasks.append((split_index, sample_key))
        if limit > 0 and len(tasks) >= limit:
            break
    return tasks


def score_split(
    *,
    split: str,
    lmdb_path: Path,
    split_path: Path,
    output_path: Path,
    existing_keys: set[str],
    limit: int,
    num_shards: int,
    shard_index: int,
    num_workers: int,
) -> dict[str, float | int | None]:
    dataset = CrossDockedLMDBDataset(
        lmdb_path=str(lmdb_path),
        split_pt_path=str(split_path),
        split=split,
        center="protein_mean",
        return_text_fields=True,
    )
    tasks = collect_split_tasks(
        dataset=dataset,
        split=split,
        existing_keys=existing_keys,
        limit=limit,
        num_shards=num_shards,
        shard_index=shard_index,
    )

    scanned = len(tasks)
    ok = 0
    failed = 0
    value_sum = 0.0
    value_min: float | None = None
    value_max: float | None = None

    with output_path.open("a", encoding="utf-8") as out_f:
        if tasks:
            mp_context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max(1, num_workers),
                mp_context=mp_context,
                initializer=init_worker_dataset,
                initargs=(str(lmdb_path), str(split_path), split),
            ) as executor:
                future_to_task = {
                    executor.submit(score_split_index, split_index): sample_key
                    for split_index, sample_key in tasks
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_task),
                    total=len(future_to_task),
                    desc=f"score {split}",
                ):
                    sample_key = future_to_task[future]
                    record = future.result()
                    value = record[DEFAULT_GUIDANCE_TARGET]
                    if record["status"] == "ok" and isinstance(value, float):
                        ok += 1
                        value_sum += value
                        value_min = value if value_min is None else min(value_min, value)
                        value_max = value if value_max is None else max(value_max, value)
                    else:
                        failed += 1

                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    existing_keys.add(sample_key)
        else:
            failed = 0

    return {
        "scanned": scanned,
        "ok": ok,
        "failed": failed,
        "score_min": value_min,
        "score_max": value_max,
        "score_mean": (value_sum / ok) if ok > 0 else None,
    }


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be >= 1")

    lmdb_path = resolve_path(args.lmdb)
    split_path = resolve_path(args.split_path)
    output_path = resolve_path(args.output, allow_missing=True)

    ensure_output_file(output_path, resume=args.resume)
    existing_keys = load_existing_sample_keys(output_path) if args.resume else set()

    print(f"LMDB: {lmdb_path}")
    print(f"Split file: {split_path}")
    print(f"Output: {output_path}")
    print(f"Sharding: {args.shard_index + 1}/{args.num_shards}")
    print(f"Workers: {args.num_workers}")

    for split in args.split:
        stats = score_split(
            split=split,
            lmdb_path=lmdb_path,
            split_path=split_path,
            output_path=output_path,
            existing_keys=existing_keys,
            limit=int(args.limit),
            num_shards=int(args.num_shards),
            shard_index=int(args.shard_index),
            num_workers=int(args.num_workers),
        )
        print(
            f"{split}: scanned={stats['scanned']} ok={stats['ok']} failed={stats['failed']} "
            f"score_mean={stats['score_mean']} range=({stats['score_min']}, {stats['score_max']})"
        )


if __name__ == "__main__":
    main()
