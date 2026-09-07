"""Aggregate ablation results into summary CSVs.

Two directory layouts are read, under whichever root ``--runs-dir`` names:
  <root>/<split_type>/<ablation_name>/<dataset_run>/metrics.json   (current)
  <root>/<ablation_name>/<dataset_run>/metrics.json                (pre-refactor)

``--runs-dir`` defaults to ``ablation/runs``, the pre-refactor tree. Results from
the current architecture live in ``ablation/runs_v2`` and are aggregated with
``python ablation/main.py --runs-dir ablation/runs_v2``, which writes its own
``*_v2.csv`` files -- the two architectures differ (the v1 runs had a transformer
stage) and their numbers must not be pooled into one table.

<dataset_run> is either <dataset> (single seed) or <dataset>_seed<N> (multi-seed).
Each metrics.json must contain at minimum a primary test metric and dataset_name.
When produced by run_experiment with --ablation-name, it also contains:
  ablation_name, seed, ablation_settings (zero_phys_branch, distill_weight, ...).

Classification and regression are written to separate files, because ROC-AUC
(higher is better) and RMSE (lower is better) cannot be ranked in one table:
  ablation/ablation_summary_classification.csv
  ablation/ablation_summary_regression.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import project_path  # noqa: E402
from common.data import SPLIT_TYPES, split_label  # noqa: E402
from common.datasets import DATASETS, TASK_TYPES  # noqa: E402

DEFAULT_RUNS_DIR = PROJECT_ROOT / "ablation" / "runs"


def output_path(task_type: str, runs_dir: Path) -> Path:
    """Summary path for a results tree.

    The default tree keeps the historic filename; any other tree appends its
    directory suffix (``runs_v2`` -> ``_v2``) so two architectures never write
    over each other's table.
    """
    suffix = ""
    if runs_dir.resolve() != DEFAULT_RUNS_DIR.resolve():
        name = runs_dir.name
        suffix = "_" + (name[len("runs_"):] if name.startswith("runs_") else name)
    return PROJECT_ROOT / "ablation" / f"ablation_summary_{task_type}{suffix}.csv"

ABLATION_SETTING_KEYS = [
    "gnn_conv",
    "zero_phys_branch",
    "distill_weight",
    "cross_distill_weight",
    "ih_rank",
    "ih_num_prototypes",
    "use_fingerprint",
    "fp_stage",
    "ih_block_proj",
]


def _strip_seed_suffix(name: str) -> str:
    return re.sub(r"_seed\d+$", "", name)


def _infer_seed(run_dir_name: str) -> int | None:
    m = re.search(r"_seed(\d+)$", run_dir_name)
    return int(m.group(1)) if m else None


def _label(split_protocol: str) -> str:
    """Published name for a protocol; unknown values (e.g. 'random_80_10_10'
    written by older runs) are passed through untouched."""
    try:
        return split_label(split_protocol)
    except ValueError:
        return split_protocol


def _read_run(run_dir: Path, ablation_name: str, split_protocol: str) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    metadata_path = run_dir / "run_metadata.json"
    if not metrics_path.exists():
        return None

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )

    base_dataset = metrics.get("dataset") or _strip_seed_suffix(run_dir.name)
    spec = DATASETS.get(base_dataset)
    # Runs written before regression support only carry test_roc_auc.
    primary = metrics.get("test_metric", metrics.get("test_roc_auc"))
    seed = metrics.get("seed") or _infer_seed(run_dir.name)

    # Ablation settings: prefer explicit field in metrics, fall back to metadata
    abl_settings = metrics.get("ablation_settings", {})
    if not abl_settings:
        hparams = metadata.get("hparams", {})
        mkw = metadata.get("model_kwargs", {})
        abl_settings = {
            "gnn_conv": mkw.get("gnn_conv", "gcn"),
            "zero_phys_branch": mkw.get("zero_phys_branch", False),
            "distill_weight": hparams.get("distill_weight"),
            "cross_distill_weight": hparams.get("cross_distill_weight"),
            "ih_rank": mkw.get("ih_rank"),
            "ih_num_prototypes": mkw.get("ih_num_prototypes"),
            "use_fingerprint": mkw.get("use_fingerprint", False),
            "fp_stage": mkw.get("fp_stage"),
            "ih_block_proj": mkw.get("ih_block_proj", False),
        }

    return {
        "ablation": metrics.get("ablation_name", ablation_name),
        "dataset": base_dataset,
        "task_type": metrics.get("task_type") or (spec.task_type if spec else "classification"),
        "metric_name": metrics.get("metric_name") or (spec.metric_name if spec else "roc_auc"),
        # Older metrics.json files stored the internal key; report the published
        # label so every generated table names the protocol the same way.
        "split_protocol": _label(metrics.get("split_protocol", split_protocol)),
        "seed": seed,
        "test_metric": primary,
        "best_val_metric": metrics.get("best_val_metric", metrics.get("best_val_auc")),
        "test_mae": metrics.get("test_mae"),
        "test_r2": metrics.get("test_r2"),
        "config_source": metrics.get("config_source", "optuna_tuned"),
        **{k: abl_settings.get(k) for k in ABLATION_SETTING_KEYS},
    }


def collect_records(runs_dir: Path) -> list[dict]:
    """Walk both the per-protocol subtrees and the flat pre-refactor layout."""
    if not runs_dir.exists():
        return []

    records: list[dict] = []
    for top_dir in sorted(runs_dir.iterdir()):
        if not top_dir.is_dir():
            continue
        if top_dir.name in SPLIT_TYPES:
            # ablation/runs/<split_type>/<ablation>/<run>/
            for ablation_dir in sorted(top_dir.iterdir()):
                if not ablation_dir.is_dir():
                    continue
                for run_dir in sorted(ablation_dir.iterdir()):
                    record = _read_run(run_dir, ablation_dir.name, top_dir.name)
                    if record is not None:
                        records.append(record)
            continue
        # ablation/runs/<ablation>/<run>/  — legacy label-aware results
        for run_dir in sorted(top_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            record = _read_run(run_dir, top_dir.name, "label_aware_scaffold")
            if record is not None:
                records.append(record)
    return records


def aggregate(records: list[dict]) -> pd.DataFrame:
    """One row per (split protocol, ablation, dataset) cell."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in records:
        key = (r["split_protocol"], r["ablation"], r["dataset"])
        groups.setdefault(key, []).append(r)

    rows: list[dict] = []
    for (split_protocol, ablation, dataset), group in sorted(groups.items()):
        # Sort by seed once, here, so per_seed_metrics lines up with seeds. The
        # runs arrive in directory order (lexicographic: seed10 before seed2),
        # which does not match a numeric sort of the seed column.
        group = sorted(group, key=lambda g: (g["seed"] is None, g["seed"]))
        values = [g["test_metric"] for g in group if g["test_metric"] is not None]
        val_values = [g["best_val_metric"] for g in group if g["best_val_metric"] is not None]
        seeds = [g["seed"] for g in group if g["seed"] is not None]

        mean_metric = float(np.mean(values)) if values else float("nan")
        std_metric = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        mean_val = float(np.mean(val_values)) if val_values else float("nan")

        rep = group[0]
        row = {
            "split_protocol": split_protocol,
            "ablation": ablation,
            "dataset": dataset,
            "task_type": rep["task_type"],
            "metric_name": rep["metric_name"],
            "n_seeds": len(values),
            "seeds": str(seeds),
            "mean_test_metric": round(mean_metric, 4),
            "std_test_metric": round(std_metric, 4),
            "mean_best_val_metric": round(mean_val, 4),
            "per_seed_metrics": str([round(v, 4) for v in values]),
            "config_source": rep.get("config_source"),
            **{k: rep.get(k) for k in ABLATION_SETTING_KEYS},
        }
        if rep["task_type"] == "regression":
            for key in ("test_mae", "test_r2"):
                present = [g[key] for g in group if g.get(key) is not None]
                row[f"mean_{key}"] = round(float(np.mean(present)), 4) if present else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR,
                        help="Results tree to aggregate. Default: ablation/runs "
                             "(pre-refactor). Current runs: ablation/runs_v2.")
    args = parser.parse_args()

    runs_dir = project_path(args.runs_dir)
    records = collect_records(runs_dir)
    if not records:
        raise SystemExit(
            f"No ablation results found under {runs_dir}.\n"
            "Run scripts/seed_expansion.py first (see commands.md)."
        )

    df = aggregate(records)
    for task_type in TASK_TYPES:
        subset = df[df["task_type"] == task_type]
        path = output_path(task_type, runs_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(path, index=False)

        print(f"\n=== {task_type}  ({len(subset)} rows across "
              f"{subset['ablation'].nunique()} ablations, "
              f"{subset['dataset'].nunique()} datasets, "
              f"{subset['split_protocol'].nunique()} split protocols) ===")
        if subset.empty:
            print("(no runs yet)")
        else:
            print(subset[["split_protocol", "ablation", "dataset", "metric_name",
                          "n_seeds", "mean_test_metric", "std_test_metric"]].to_string(index=False))
        print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
