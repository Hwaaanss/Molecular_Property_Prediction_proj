"""Compute Student-t 95% CIs from per-seed metrics.json files.

Automatically discovers all available seeds in each (condition, dataset) cell,
so it works transparently with n=5, n=15, or any other seed count.

Split protocol
  --split-type selects which protocol subtree to read,
      ablation/runs/<split_type>/<condition>/<dataset>_seed<N>/metrics.json
  and is appended to every output filename, so results from different protocols
  are never mixed into one CSV. --legacy instead reads the flat pre-refactor
  layout ablation/runs/<condition>/... which holds the label-aware results.

Task types are never mixed: ROC-AUC (higher better) and RMSE (lower better) do
not belong in one table, so each gets its own CSV.

Outputs under results/artifacts/revision/:
  ablation_summary_with_ci_classification_<suffix>.csv   — conditions × classification sets
  ablation_summary_with_ci_regression_<suffix>.csv       — conditions × regression sets
  random_split_summary_with_ci_classification_random.csv — SIDER/Tox21/ClinTox, read
      from ablation/runs/full_model_random/, which is random-split by definition
      and therefore protocol-tagged by its own name rather than by --split-type

Usage on server (after seed_expansion.py completes):
  conda activate dualgnn
  python scripts/compute_ci.py                                  # deterministic_scaffold, both task types
  python scripts/compute_ci.py --task-type regression
  python scripts/compute_ci.py --split-type random_scaffold
  python scripts/compute_ci.py --legacy                         # old label-aware results

The script prints a headline table showing the new n and CI half-widths so you
can immediately verify the expansion increased statistical power.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import ablation_runs_root, project_path  # noqa: E402
from common.data import SPLIT_TYPE_CHOICES, resolve_split_type, split_label  # noqa: E402
from common.datasets import DATASETS as DATASET_SPECS, TASK_TYPES, datasets_for_task  # noqa: E402
from dual_kd_gnn.model import GNN_CONV_TYPES  # noqa: E402

OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "revision"
# Current grid (<conv>_full_model, <conv>_no_<factors>) plus the pre-refactor
# names, so one command covers either results tree; conditions with no runs are
# skipped silently.
FACTOR_ORDER = ("3d", "infonce", "codebook", "fp")
CURRENT_ABLATIONS = [
    (f"{conv}_full_model" if not removed else f"{conv}_no_" + "_".join(removed))
    for conv in GNN_CONV_TYPES
    for size in range(len(FACTOR_ORDER) + 1)
    for removed in itertools.combinations(FACTOR_ORDER, size)
]
LEGACY_ABLATIONS = ["full_model", "a1_no_phys", "a2_no_infonce", "a4_no_codebook",
                    "a12_no_phys_infonce", "a14_no_phys_codebook", "a24_no_infonce_codebook",
                    "a124_no_all_three", "a3_no_mse_kd", "a5_linear_head"]
ABLATIONS = CURRENT_ABLATIONS + LEGACY_ABLATIONS
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ablation" / "runs_v2"
DATASETS_RANDOM = ["sider", "tox21", "clintox"]
# Secondary metrics reported alongside RMSE for regression cells.
REGRESSION_EXTRAS = ("test_mae", "test_r2")
# dataset -> |test mean - train mean| in train sd; filled in per task type.
_LABEL_SHIFT_CACHE: dict[str, float | None] = {}


def collect_seeds(condition: str, dataset: str, base: Path,
                  keep_seeds: set[int] | None = None) -> dict[int, dict]:
    """Discover all seed folders for one (condition, dataset) cell.

    Returns {seed: metrics_dict}. ``test_metric`` is the protocol-appropriate
    primary score (ROC-AUC or RMSE); runs written before the regression support
    only have ``test_roc_auc``, so that is used as the fallback.

    ``keep_seeds`` restricts the aggregation to specific seeds. Without it every
    seed directory found is used, which mixes seed counts across cells whenever
    a dataset was swept more than once.
    """
    cond_dir = base / condition
    if not cond_dir.exists():
        return {}
    pattern = re.compile(rf"^{re.escape(dataset)}_seed(\d+)$")
    out: dict[int, dict] = {}
    for run_dir in cond_dir.iterdir():
        m = pattern.match(run_dir.name)
        if not m:
            continue
        if keep_seeds is not None and int(m.group(1)) not in keep_seeds:
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        primary = data.get("test_metric", data.get("test_roc_auc"))
        if primary is None:
            continue
        data["_primary"] = float(primary)
        out[int(m.group(1))] = data
    return out


def student_t_ci(aucs: list[float], alpha: float = 0.05) -> tuple[float, float, float, float, float]:
    """Return (mean, std_ddof1, ci_lower, ci_upper, margin)."""
    arr = np.array(aucs, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    if n < 2:
        return mean, 0.0, mean, mean, 0.0
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = float(t_crit * se)
    return mean, sd, mean - margin, mean + margin, margin


def label_shift_sd(dataset: str, split_type: str) -> float | None:
    """|test mean - train mean| in train standard deviations, for regression sets.

    R^2 is measured against the *test* variance, so a split that shifts the label
    distribution can drive it negative even when the model beats every baseline
    available to it. Malaria is the clear case: its CSV is sorted by activity, and
    because ~half its molecules are singleton scaffolds the split's (size, first
    index) tie-break sends the file's leading -- highest-activity -- rows to test.
    Recording the shift next to the metrics keeps that caveat with the numbers.
    """
    spec = DATASET_SPECS[dataset]
    if not spec.is_regression:
        return None
    path = spec.data_path()
    if not path.exists():
        return None
    from common.data import split_indices  # local: keeps the import cost off the CLI path

    frame = pd.read_csv(path)
    targets = list(spec.target_columns) if spec.target_columns else None
    if not targets:
        return None
    train_idx, _, test_idx = split_indices(
        frame, split_type=split_type, target_columns=targets,
        smiles_column=spec.smiles_column, seed=42, task_type=spec.task_type,
    )
    values = frame[targets[0]].to_numpy(dtype=float)
    train_sd = float(np.nanstd(values[train_idx]))
    if train_sd < 1e-8:
        return None
    return float(abs(np.nanmean(values[test_idx]) - np.nanmean(values[train_idx])) / train_sd)


def _summarize(runs: dict[int, dict], dataset: str) -> dict:
    """CI row body shared by the ablation and random-split tables."""
    spec = DATASET_SPECS[dataset]
    values = [run["_primary"] for run in runs.values()]
    mean, sd, lo, hi, margin = student_t_ci(values)
    # Best seed by the metric's own direction: max AUC, min RMSE.
    ordered = sorted(values, reverse=spec.greater_is_better)
    row = {
        "task_type": spec.task_type,
        "metric_name": spec.metric_name,
        "n_seeds": len(values),
        "mean_test_metric": round(mean, 4),
        "std_test_metric": round(sd, 4),
        "ci95_lower": round(lo, 4),
        "ci95_upper": round(hi, 4),
        "ci95_margin": round(margin, 4),
        "per_seed_metrics": str([round(v, 4) for v in ordered]),
        "config_source": next(iter(runs.values())).get("config_source", "optuna_tuned"),
    }
    if spec.is_regression:
        for key in REGRESSION_EXTRAS:
            present = [run[key] for run in runs.values() if run.get(key) is not None]
            row[f"mean_{key}"] = round(float(np.mean(present)), 4) if present else None
        shift = _LABEL_SHIFT_CACHE.get(dataset)
        row["train_test_shift_sd"] = round(shift, 3) if shift is not None else None
        # A large shift shrinks the test variance that R^2 is normalised by, so a
        # negative R^2 there says more about the split than about the model.
        row["r2_reliable"] = None if shift is None else bool(shift < 1.0)
    else:
        # Keep the historical column name so existing readers of the
        # classification table keep working.
        row["mean_test_roc_auc"] = round(mean, 4)
        row["std_test_roc_auc"] = round(sd, 4)
    return row


def build_summary(scope: str, base: Path, split_protocol: str, datasets: list[str],
                  random_condition: str = "full_model",
                  keep_seeds: set[int] | None = None) -> pd.DataFrame:
    """scope in {'ablation', 'random'}."""
    rows = []
    if scope == "ablation":
        for condition in ABLATIONS:
            for dataset in datasets:
                runs = collect_seeds(condition, dataset, base, keep_seeds)
                if not runs:
                    continue
                rows.append({
                    "ablation": condition,
                    "dataset": dataset,
                    "split_protocol": split_protocol,
                    **_summarize(runs, dataset),
                })
    elif scope == "random":
        for dataset in datasets:
            runs = collect_seeds(random_condition, dataset, base, keep_seeds)
            if not runs:
                print(f"  [warn] {random_condition} x {dataset}: no seeds found")
                continue
            rows.append({
                "dataset": dataset,
                "split_protocol": split_protocol,
                **_summarize(runs, dataset),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-type", default="scaffold", choices=SPLIT_TYPE_CHOICES,
                        help="Split protocol to aggregate. Default: scaffold. "
                             "('deterministic_scaffold' is accepted as the same thing.)")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                        help="Results tree. Default: ablation/runs_v2. "
                             "Pass ablation/runs for the pre-refactor results.")
    parser.add_argument("--legacy", action="store_true",
                        help="Read the flat pre-refactor layout ablation/runs/<condition>/... "
                             "(the label-aware results) instead of a protocol subtree.")
    parser.add_argument("--task-type", default="all", choices=[*TASK_TYPES, "all"],
                        help="Which task type to aggregate. Default: all, which writes one "
                             "CSV per task type. Classification and regression are never "
                             "mixed into the same file.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, metavar="N",
                        help="Aggregate only these seeds, e.g. --seeds 0 1 2 3 42. Without it "
                             "every seed directory present is used, which mixes seed counts "
                             "across cells when a dataset was swept more than once. Restricting "
                             "the seeds appends '_<n>seed' to the output filenames so the "
                             "unrestricted tables are left in place.")
    args = parser.parse_args()

    keep_seeds = set(args.seeds) if args.seeds else None
    split_type = resolve_split_type(args.split_type)
    label = split_label(split_type)
    suffix = "legacy_label_aware" if args.legacy else label
    if keep_seeds:
        suffix = f"{suffix}_{len(keep_seeds)}seed"
    # Tag the results tree into the filename, so a v2 run never overwrites the
    # CI table computed from the pre-refactor runs.
    # Relative paths are project-relative, so a documented command means the
    # same thing from any working directory.
    runs_root = project_path(args.runs_root)
    tree_name = runs_root.name
    if not args.legacy and tree_name != "runs":
        suffix = f"{suffix}_{tree_name[len('runs_'):] if tree_name.startswith('runs_') else tree_name}"
    split_protocol = "label_aware_scaffold" if args.legacy else label
    base = (
        ablation_runs_root(split_type, legacy=True) if args.legacy
        else runs_root / split_type
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading: {base}")
    if not base.exists():
        print("  [warn] directory does not exist — no runs for this protocol yet")

    task_types = list(TASK_TYPES) if args.task_type == "all" else [args.task_type]
    written: list[Path] = []

    for task_type in task_types:
        datasets = datasets_for_task(task_type)
        # Computed once per dataset, not once per (condition, dataset) cell.
        if task_type == "regression":
            _LABEL_SHIFT_CACHE.update(
                {name: label_shift_sd(name, split_type) for name in datasets}
            )
        ablation_df = build_summary("ablation", base, split_protocol, datasets,
                                    keep_seeds=keep_seeds)
        path = OUT_DIR / f"ablation_summary_with_ci_{task_type}_{suffix}.csv"
        ablation_df.to_csv(path, index=False)
        written.append(path)

        print(f"\n=== {path.name} (full_model rows only) ===")
        if ablation_df.empty:
            print("(no runs found for this task type / protocol)")
        else:
            full_rows = ablation_df[ablation_df["ablation"] == "full_model"]
            columns = ["dataset", "metric_name", "n_seeds", "mean_test_metric",
                       "std_test_metric", "ci95_lower", "ci95_upper", "ci95_margin"]
            if task_type == "regression":
                columns += [f"mean_{key}" for key in REGRESSION_EXTRAS]
                columns += ["train_test_shift_sd", "r2_reliable"]
            print(full_rows[columns].to_string(index=False) if not full_rows.empty else "(no full_model rows)")
            if task_type == "regression" and not full_rows.empty:
                shifted = full_rows[full_rows["r2_reliable"] == False]  # noqa: E712 - pandas mask
                for _, row in shifted.iterrows():
                    print(f"  [note] {row['dataset']}: train->test label shift is "
                          f"{row['train_test_shift_sd']:.2f} train-sd, which compresses the test "
                          f"variance R^2 is normalised by. Report RMSE ({row['mean_test_metric']:.3f}); "
                          "the negative R^2 reflects the split, not model failure.")

    # ablation/runs/full_model_random/ has only ever held random-split full_model
    # runs on classification sets, so it needs no protocol subtree.
    random_df = build_summary(
        "random", ablation_runs_root(), "random", DATASETS_RANDOM,
        random_condition="full_model_random", keep_seeds=keep_seeds,
    )
    random_suffix = f"random_{len(keep_seeds)}seed" if keep_seeds else "random"
    random_path = OUT_DIR / f"random_split_summary_with_ci_classification_{random_suffix}.csv"
    random_df.to_csv(random_path, index=False)
    written.append(random_path)

    print(f"\n=== {random_path.name} ===")
    if random_df.empty:
        print("(no rows)")
    else:
        print(random_df[["dataset", "n_seeds", "mean_test_metric", "std_test_metric",
                         "ci95_lower", "ci95_upper", "ci95_margin"]].to_string(index=False))

    print("\nSaved:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
