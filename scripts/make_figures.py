"""Validate the ablation result CSVs and draw the figures for one task type.

Runs as steps (3) and (6) of the full pipeline: after a sweep finishes, confirm
the summary CSVs really were written and really carry a standard deviation for
every cell, then produce the figures.

Validation is strict -- it exits non-zero if a CSV is missing, empty, or has a
cell with fewer than two seeds (a std of 0.0 from n=1 is not a std). That way a
half-finished sweep fails loudly here instead of silently producing a figure
with meaningless error bars.

Every figure is written at 300 dpi; PNG by default (DIKAT_FIGURE_FORMATS
adds tiff/pdf).

Usage:
    python scripts/make_figures.py --task-type classification
    python scripts/make_figures.py --task-type regression
    python scripts/make_figures.py --task-type all --split-type deterministic_scaffold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.resources import apply_thread_limits  # noqa: E402

apply_thread_limits()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import SPLIT_TYPE_CHOICES, split_label  # noqa: E402
from common.datasets import DATASETS, TASK_TYPES, datasets_for_task  # noqa: E402
from common.plotting import save_figure  # noqa: E402

ABLATION_DIR = PROJECT_ROOT / "ablation"
REVISION_DIR = PROJECT_ROOT / "results" / "artifacts" / "revision"
FIGURE_DIR = PROJECT_ROOT / "results" / "artifacts" / "figures"

# Consistent condition order and colours across every figure.
#
# Current names are "<conv>_full_model" / "<conv>_no_<factors>"; figures are
# drawn one encoder at a time (--results-tree runs_v2 tables carry all three),
# so the order below is by removal count and the label drops the encoder prefix,
# which is already in the figure title.
FACTOR_ORDER = ("3d", "infonce", "codebook", "fp")
FACTOR_LABELS = {"3d": "3D", "infonce": "NCE", "codebook": "CB", "fp": "FP"}
CONDITION_ORDER = (
    [f"{conv}_full_model" for conv in ("gcn", "gine", "gat")]
    + [f"{conv}_no_" + "_".join(removed)
       for conv in ("gcn", "gine", "gat")
       for size in range(1, len(FACTOR_ORDER) + 1)
       for removed in __import__("itertools").combinations(FACTOR_ORDER, size)]
    # Pre-refactor names, so ablation/runs tables still order sensibly.
    + ["full_model", "a1_no_phys", "a2_no_infonce", "a4_no_codebook",
       "a12_no_phys_infonce", "a14_no_phys_codebook", "a24_no_infonce_codebook",
       "a124_no_all_three"]
)


def _condition_label(condition: str) -> str:
    """'gcn_no_3d_fp' -> '−3D,FP'; 'gcn_full_model' -> 'Full'."""
    body = condition.split("_", 1)[1] if condition.split("_", 1)[0] in ("gcn", "gine", "gat") else condition
    if body == "full_model":
        return "Full"
    if body.startswith("no_"):
        return "−" + ",".join(FACTOR_LABELS.get(f, f) for f in body[3:].split("_"))
    return LEGACY_CONDITION_LABELS.get(condition, condition)


LEGACY_CONDITION_LABELS = {
    "full_model": "Full",
    "a1_no_phys": "−A1",
    "a2_no_infonce": "−A2",
    "a4_no_codebook": "−A4",
    "a12_no_phys_infonce": "−A1,A2",
    "a14_no_phys_codebook": "−A1,A4",
    "a24_no_infonce_codebook": "−A2,A4",
    "a124_no_all_three": "−A1,A2,A4",
}
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(task_type: str, split_type: str, suffix: str,
             ci_command_hint: str, summary_suffix: str = "") -> tuple[pd.DataFrame, list[str]]:
    """Load and check the summary CSVs. Returns (ci_frame, problems).

    The CI table is the figure data source: it carries mean, std and the 95%
    interval for every cell, and — unlike ablation_summary — it honours the seed
    filter, so a figure never mixes seed counts that the table does not.
    """
    problems: list[str] = []
    summary_path = ABLATION_DIR / f"ablation_summary_{task_type}{summary_suffix}.csv"
    ci_path = REVISION_DIR / f"ablation_summary_with_ci_{task_type}_{suffix}.csv"

    if not summary_path.exists():
        problems.append(f"missing {summary_path.relative_to(PROJECT_ROOT)} — run: python ablation/main.py")
    else:
        summary = pd.read_csv(summary_path)
        if summary[summary["split_protocol"] == split_type].empty:
            problems.append(f"{summary_path.name} has no rows for split_protocol={split_type}")

    if not ci_path.exists() or ci_path.stat().st_size <= 1:
        problems.append(
            f"missing or empty {ci_path.relative_to(PROJECT_ROOT)} — run: {ci_command_hint}"
        )
        return pd.DataFrame(), problems

    frame = pd.read_csv(ci_path)
    if frame.empty:
        problems.append(f"{ci_path.name} has no rows")
        return frame, problems

    # Standard deviation must be present and meaningful.
    missing_std = frame[frame["std_test_metric"].isna()]
    for _, row in missing_std.iterrows():
        problems.append(f"{row['ablation']} x {row['dataset']}: std_test_metric is NaN")
    single_seed = frame[frame["n_seeds"] < 2]
    for _, row in single_seed.iterrows():
        problems.append(
            f"{row['ablation']} x {row['dataset']}: n_seeds={row['n_seeds']} "
            "— a standard deviation needs at least 2 seeds"
        )
    if frame["ci95_margin"].isna().any():
        problems.append(f"{ci_path.name} has NaN ci95_margin values")

    expected = set(datasets_for_task(task_type))
    for name in sorted(expected - set(frame["dataset"])):
        problems.append(f"no runs for dataset '{name}'")

    return frame, problems


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _ordered_conditions(frame: pd.DataFrame) -> list[str]:
    present = set(frame["ablation"])
    ordered = [c for c in CONDITION_ORDER if c in present]
    return ordered + sorted(present - set(ordered))


def _ordered_datasets(frame: pd.DataFrame, task_type: str) -> list[str]:
    """Datasets smallest-first, matching the order the sweep runs them in."""
    present = [d for d in datasets_for_task(task_type) if d in set(frame["dataset"])]
    sizes = {}
    for name in present:
        path = DATASETS[name].data_path()
        sizes[name] = sum(1 for _ in path.open()) if path.exists() else 0
    return sorted(present, key=lambda d: sizes.get(d, 0))


def figure_metric_by_dataset(frame: pd.DataFrame, task_type: str, split_type: str,
                             suffix: str) -> None:
    """Grouped bars: one group per dataset, one bar per ablation, +-1 sd."""
    conditions = _ordered_conditions(frame)
    datasets = _ordered_datasets(frame, task_type)
    if not conditions or not datasets:
        return
    metric = frame["metric_name"].iloc[0]

    width = 0.8 / len(conditions)
    positions = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(8, 1.7 * len(datasets)), 5.5))

    for index, condition in enumerate(conditions):
        means, errors = [], []
        for dataset in datasets:
            cell = frame[(frame["ablation"] == condition) & (frame["dataset"] == dataset)]
            means.append(float(cell["mean_test_metric"].iloc[0]) if not cell.empty else np.nan)
            errors.append(float(cell["std_test_metric"].iloc[0]) if not cell.empty else 0.0)
        ax.bar(positions + index * width - 0.4 + width / 2, means, width,
               yerr=errors, capsize=2.5, label=_condition_label(condition),
               color=PALETTE[index % len(PALETTE)],
               error_kw={"linewidth": 0.8, "ecolor": "#333333"})

    ax.set_xticks(positions)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_xlabel("Dataset (smallest to largest)")
    ax.set_ylabel(f"Test {metric.upper()}" + (" (lower is better)" if metric == "rmse" else ""))
    ax.set_title(f"{task_type.capitalize()} ablation — test {metric.upper()} "
                 f"(mean ± 1 sd over seeds, {split_type})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=min(4, len(conditions)), fontsize=8, framealpha=0.9)
    if metric == "roc_auc":
        ax.set_ylim(0.4, 1.0)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"ablation_{task_type}_{suffix}_by_dataset",
                bbox_inches="tight")
    plt.close(fig)


def _reference_for(condition: str) -> str:
    """The full-model condition an ablation is compared against.

    Each encoder series has its own reference: comparing a GINE ablation against
    a GCN full model would fold the encoder swap into the delta.
    """
    conv = condition.split("_", 1)[0]
    return f"{conv}_full_model" if conv in ("gcn", "gine", "gat") else "full_model"


def figure_delta_vs_full(frame: pd.DataFrame, task_type: str, split_type: str,
                         suffix: str) -> None:
    """Change relative to full_model, signed so positive always means worse."""
    conditions = [c for c in _ordered_conditions(frame) if not c.endswith("full_model")]
    datasets = _ordered_datasets(frame, task_type)
    if not conditions or not datasets:
        return
    metric = frame["metric_name"].iloc[0]
    # ROC-AUC: removing a component usually lowers it, so full-ablation is the
    # degradation. RMSE: higher is worse, so ablation-full is the degradation.
    higher_is_better = metric != "rmse"

    matrix = np.full((len(conditions), len(datasets)), np.nan)
    for i, condition in enumerate(conditions):
        for j, dataset in enumerate(datasets):
            full = frame[(frame["ablation"] == _reference_for(condition)) & (frame["dataset"] == dataset)]
            cell = frame[(frame["ablation"] == condition) & (frame["dataset"] == dataset)]
            if full.empty or cell.empty:
                continue
            full_value = float(full["mean_test_metric"].iloc[0])
            cell_value = float(cell["mean_test_metric"].iloc[0])
            matrix[i, j] = full_value - cell_value if higher_is_better else cell_value - full_value

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(datasets)), 0.6 * len(conditions) + 2.5))
    limit = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([d.upper() for d in datasets], rotation=30, ha="right")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([_condition_label(c) for c in conditions])
    for i in range(len(conditions)):
        for j in range(len(datasets)):
            if not np.isnan(matrix[i, j]):
                shade = "white" if abs(matrix[i, j]) > limit * 0.6 else "black"
                ax.text(j, i, f"{matrix[i, j]:+.3f}", ha="center", va="center",
                        color=shade, fontsize=8)
    ax.set_title(f"{task_type.capitalize()} — degradation vs full model "
                 f"({metric.upper()}, {split_type})\npositive = ablation is worse")
    fig.colorbar(image, ax=ax, label=f"Δ {metric.upper()} (worse →)")
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"ablation_{task_type}_{suffix}_delta",
                bbox_inches="tight")
    plt.close(fig)


def figure_ci(frame: pd.DataFrame, task_type: str, split_type: str, suffix: str) -> None:
    """Full-model mean with its 95% CI, one row per dataset (all encoders)."""
    frame = frame[frame["ablation"].str.endswith("full_model")]
    if frame.empty:
        return
    metric = frame["metric_name"].iloc[0]
    frame = frame.sort_values("dataset")

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(frame) + 2.2))
    positions = np.arange(len(frame))
    means = frame["mean_test_metric"].to_numpy(dtype=float)
    lower = means - frame["ci95_lower"].to_numpy(dtype=float)
    upper = frame["ci95_upper"].to_numpy(dtype=float) - means
    ax.errorbar(means, positions, xerr=[lower, upper], fmt="o", capsize=4,
                color="#4C72B0", ecolor="#4C72B0", markersize=6, linewidth=1.5)
    for position, mean, n in zip(positions, means, frame["n_seeds"]):
        ax.text(mean, position + 0.18, f"{mean:.3f} (n={n})", ha="center", fontsize=8)
    ax.set_yticks(positions)
    ax.set_yticklabels([d.upper() for d in frame["dataset"]])
    ax.set_xlabel(f"Test {metric.upper()} (95% CI)")
    ax.set_title(f"{task_type.capitalize()} — full model, Student-t 95% CI ({split_type})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"ablation_{task_type}_{suffix}_ci", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task-type", default="all", choices=[*TASK_TYPES, "all"])
    parser.add_argument("--split-type", default="scaffold", choices=SPLIT_TYPE_CHOICES,
                        help="Default: scaffold ('deterministic_scaffold' means the same).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, metavar="N",
                        help="Read the seed-restricted tables written by "
                             "'compute_ci.py --seeds ...', e.g. --seeds 0 1 2 3 42, and tag "
                             "the figures the same way. Must match the seeds compute_ci used.")
    parser.add_argument("--results-tree", default="runs_v2", metavar="NAME",
                        help="Which results tree the tables came from: 'runs' for the "
                             "pre-refactor summaries, 'runs_v2' (default) for the current "
                             "architecture. Selects the CSV filename suffix.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Report validation problems as warnings and still draw figures, "
                             "instead of exiting non-zero.")
    args = parser.parse_args()
    # Generated tables and figures use the published protocol name throughout.
    args.split_type = split_label(args.split_type)
    suffix = args.split_type
    ci_hint = f"python scripts/compute_ci.py --split-type {args.split_type}"
    if args.seeds:
        suffix = f"{suffix}_{len(set(args.seeds))}seed"
        ci_hint += " --seeds " + " ".join(str(s) for s in args.seeds)
    tree = args.results_tree
    summary_suffix = "" if tree == "runs" else "_" + (tree[len("runs_"):] if tree.startswith("runs_") else tree)
    if summary_suffix:
        suffix = f"{suffix}{summary_suffix}"
        ci_hint += f" --runs-root ablation/{tree}"

    task_types = list(TASK_TYPES) if args.task_type == "all" else [args.task_type]
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    all_problems: list[str] = []

    for task_type in task_types:
        print(f"\n=== {task_type} ({suffix}) ===")
        frame, problems = validate(task_type, args.split_type, suffix, ci_hint, summary_suffix)
        if problems:
            print(f"  {len(problems)} problem(s):")
            for problem in problems:
                print(f"    - {problem}")
            all_problems.extend(f"[{task_type}] {p}" for p in problems)
        else:
            print(f"  CSV validation OK: {len(frame)} cells, "
                  f"{frame['dataset'].nunique()} datasets, "
                  f"{frame['ablation'].nunique()} conditions, "
                  f"n_seeds {int(frame['n_seeds'].min())}-{int(frame['n_seeds'].max())}, "
                  "std present for every cell")

        if frame.empty:
            continue
        figure_metric_by_dataset(frame, task_type, args.split_type, suffix)
        figure_delta_vs_full(frame, task_type, args.split_type, suffix)
        figure_ci(frame, task_type, args.split_type, suffix)

    written = sorted(FIGURE_DIR.glob(f"*_{suffix}_*"))
    print(f"\nFigures in {FIGURE_DIR.relative_to(PROJECT_ROOT)}/ (300 dpi PNG + LZW TIFF):")
    for path in written:
        print(f"  {path.name}  ({path.stat().st_size/1024:.0f} KB)")

    if all_problems and not args.allow_incomplete:
        raise SystemExit(
            f"\n{len(all_problems)} validation problem(s) — see above. "
            "Re-run the missing steps, or pass --allow-incomplete to draw figures anyway."
        )


if __name__ == "__main__":
    main()
