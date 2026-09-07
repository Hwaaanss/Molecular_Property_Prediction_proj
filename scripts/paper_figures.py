"""Figures that are read off the run tree rather than the summary CSVs.

``make_figures.py`` draws everything that the aggregated CSVs can support:
grouped bars, the degradation heatmap, and the full model's confidence
intervals. Three things a paper still needs are not in those CSVs at all,
because they are per-run quantities that only the run directories carry:

  learning curves     per-epoch train/val loss and metric, averaged over seeds,
                      with the stage 1 / stage 2 boundary marked. Reviewers ask
                      whether the two-phase schedule is doing anything; a table
                      of final numbers cannot answer that.
  block norms         how much of the learned quadratic form sits in each block
                      pair. This is the figure the paper's third contribution
                      stands on -- "geometry reaches the score" is a claim about
                      the geo-topo and geo-fp cross blocks specifically, and
                      until it is plotted it is an assertion.
  prototype routing   the task x prototype assignment matrix, which is what
                      turns the codebook from a parameter-count argument into a
                      statement about which tasks share a decision axis.

Two more are drawn here because they summarise the ablation across datasets in
the form conference readers expect: a mean-rank plot and the per-seed paired
delta distribution that the Wilcoxon test is computed from.

Every figure is 300 dpi PNG (set DIKAT_FIGURE_FORMATS to add tiff/pdf).

Usage:
    python scripts/paper_figures.py --runs-root ablation/runs_v5_optuna \
        --split-type scaffold --task-type all --seeds 0 1 2 3 42
"""
from __future__ import annotations

import argparse
import json
import re
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

from common.config import project_path  # noqa: E402
from common.data import SPLIT_TYPE_CHOICES, resolve_split_type, split_label  # noqa: E402
from common.datasets import DATASETS, TASK_TYPES, datasets_for_task  # noqa: E402
from common.plotting import save_figure  # noqa: E402

FIGURE_DIR = PROJECT_ROOT / "results" / "artifacts" / "figures"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ablation" / "runs_v2"

FACTOR_ORDER = ("3d", "infonce", "codebook", "fp")
FACTOR_LABELS = {"3d": "3D", "infonce": "NCE", "codebook": "CB", "fp": "FP"}
CONVS = ("gcn", "gine", "gat")
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
           "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
# Fixed colour per block pair, so the same pair reads the same across figures.
BLOCK_PAIR_COLORS = {
    "geo-geo": "#4C72B0", "topo-topo": "#DD8452", "fp-fp": "#55A868",
    "geo-topo": "#C44E52", "geo-fp": "#8172B3", "topo-fp": "#937860",
}
SEED_RE = re.compile(r"^(?P<dataset>.+)_seed(?P<seed>\d+)$")


def condition_label(condition: str) -> str:
    """'gcn_no_3d_fp' -> '-3D,FP'; 'gcn_full_model' -> 'Full'."""
    head, _, body = condition.partition("_")
    if head not in CONVS:
        head, body = "", condition
    if body == "full_model":
        return "Full"
    if body.startswith("no_"):
        return "−" + ",".join(FACTOR_LABELS.get(f, f) for f in body[3:].split("_"))
    return condition


def reference_for(condition: str) -> str:
    conv = condition.split("_", 1)[0]
    return f"{conv}_full_model" if conv in CONVS else "full_model"


def sort_key(condition: str) -> tuple:
    """Full model first, then by how many factors were removed."""
    head, _, body = condition.partition("_")
    conv_rank = CONVS.index(head) if head in CONVS else len(CONVS)
    if body == "full_model":
        return (conv_rank, 0, "")
    removed = body[3:].split("_") if body.startswith("no_") else []
    return (conv_rank, len(removed), condition)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(runs_root: Path, split_type: str, keep_seeds: set[int] | None) -> pd.DataFrame:
    """One row per run directory found under <runs_root>/<split_type>/."""
    base = runs_root / split_type
    rows: list[dict] = []
    if not base.exists():
        return pd.DataFrame(rows)
    for condition_dir in sorted(base.iterdir()):
        if not condition_dir.is_dir():
            continue
        for run_dir in sorted(condition_dir.iterdir()):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            match = SEED_RE.match(run_dir.name)
            if not match:
                continue
            seed = int(match.group("seed"))
            if keep_seeds is not None and seed not in keep_seeds:
                continue
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            dataset = data.get("dataset") or match.group("dataset")
            spec = DATASETS.get(dataset)
            rows.append({
                "condition": data.get("ablation_name", condition_dir.name),
                "dataset": dataset,
                "seed": seed,
                "task_type": data.get("task_type") or (spec.task_type if spec else "classification"),
                "metric_name": data.get("metric_name") or (spec.metric_name if spec else "roc_auc"),
                "test_metric": data.get("test_metric", data.get("test_roc_auc")),
                "head_diagnostics": data.get("head_diagnostics") or {},
                "run_dir": str(run_dir),
            })
    return pd.DataFrame(rows)


def ordered_datasets(frame: pd.DataFrame, task_type: str) -> list[str]:
    present = set(frame["dataset"])
    names = [d for d in datasets_for_task(task_type) if d in present]
    sizes = {}
    for name in names:
        path = DATASETS[name].data_path()
        sizes[name] = sum(1 for _ in path.open()) if path.exists() else 0
    return sorted(names, key=lambda d: sizes.get(d, 0))


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------

def _stack_histories(run_dirs: list[str], column: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean and sd of one training_log column across seeds, plus stage-1 length.

    Runs stop at different epochs (early stopping in phase 2), so the curves are
    padded to the longest and averaged with nanmean: the tail is then the mean
    over whichever seeds were still running, which is the honest thing to show
    and is why the band widens to the right.
    """
    series: list[np.ndarray] = []
    stage1 = 0
    for run_dir in run_dirs:
        path = Path(run_dir) / "training_log.csv"
        if not path.exists():
            continue
        log = pd.read_csv(path)
        if column not in log.columns or log.empty:
            continue
        series.append(log[column].to_numpy(dtype=float))
        if "phase" in log.columns:
            stage1 = max(stage1, int((log["phase"] == "stage1_gcn_kd").sum()))
    if not series:
        return np.array([]), np.array([]), 0
    length = max(len(s) for s in series)
    padded = np.full((len(series), length), np.nan)
    for index, values in enumerate(series):
        padded[index, :len(values)] = values
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0, ddof=1) if len(series) > 1 else np.zeros(length)
    return mean, std, stage1


def figure_learning_curves(frame: pd.DataFrame, task_type: str, split: str,
                           suffix: str, condition: str) -> None:
    """Loss and metric vs epoch for one condition, one panel per dataset."""
    subset = frame[(frame["task_type"] == task_type) & (frame["condition"] == condition)]
    datasets = ordered_datasets(subset, task_type)
    if not datasets:
        return
    metric = subset["metric_name"].iloc[0]
    metric_label = "RMSE" if metric == "rmse" else metric.replace("_", "-").upper()

    columns = max(1, min(4, len(datasets)))
    rows = int(np.ceil(len(datasets) / columns))
    for kind, (train_col, val_col, ylabel) in {
        "loss": ("train_loss", "val_loss", "Loss"),
        "metric": ("train_metric", "val_metric", metric_label),
    }.items():
        # Floor on the width: with one panel the figure would otherwise be
        # narrower than its own title, and bbox_inches="tight" then stretches
        # the canvas around the text instead of around the plot.
        fig, axes = plt.subplots(rows, columns,
                                 figsize=(max(7.0, 4.2 * columns), 3.1 * rows + 0.5),
                                 squeeze=False)
        drew = False
        for index, dataset in enumerate(datasets):
            axis = axes[index // columns][index % columns]
            run_dirs = subset[subset["dataset"] == dataset]["run_dir"].tolist()
            n_seeds = len(run_dirs)
            for column, color, name in ((train_col, "#4C72B0", "train"),
                                        (val_col, "#C44E52", "val")):
                mean, std, stage1 = _stack_histories(run_dirs, column)
                if mean.size == 0 or np.all(np.isnan(mean)):
                    continue
                drew = True
                epochs = np.arange(1, mean.size + 1)
                axis.plot(epochs, mean, color=color, linewidth=1.4, label=name)
                axis.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.18,
                                  linewidth=0)
                if stage1:
                    axis.axvline(stage1 + 0.5, color="#888888", linestyle="--", linewidth=0.9)
            axis.set_title(f"{dataset.upper()}  (n={n_seeds})", fontsize=10)
            axis.grid(alpha=0.3)
            axis.tick_params(labelsize=8)
            if index % columns == 0:
                axis.set_ylabel(ylabel, fontsize=9)
            if index // columns == rows - 1:
                axis.set_xlabel("Epoch", fontsize=9)
        for index in range(len(datasets), rows * columns):
            axes[index // columns][index % columns].axis("off")
        if not drew:
            plt.close(fig)
            continue
        # Legend inside the first panel, not at figure level: a figure-level
        # legend is placed relative to the canvas and lands on top of the
        # suptitle whenever the grid is small.
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            axes[0][0].legend(handles, labels, fontsize=8, framealpha=0.9, loc="best")
        fig.suptitle(
            f"{condition} — {kind} vs epoch ({task_type}, {split})\n"
            f"mean ± 1 sd over seeds; dashed line = stage 1 → stage 2",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(fig, FIGURE_DIR / f"learning_curve_{kind}_{condition}_{task_type}_{suffix}",
                    bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Block norms of the learned quadratic form
# ---------------------------------------------------------------------------

def figure_block_norms(frame: pd.DataFrame, task_type: str, split: str,
                       suffix: str, condition: str) -> None:
    """Share of the quadratic form carried by each block pair, per dataset."""
    subset = frame[(frame["task_type"] == task_type) & (frame["condition"] == condition)]
    datasets = ordered_datasets(subset, task_type)
    if not datasets:
        return

    shares: dict[str, dict[str, list[float]]] = {}
    for dataset in datasets:
        for _, row in subset[subset["dataset"] == dataset].iterrows():
            fractions = (row["head_diagnostics"] or {}).get("block_norms_fraction") or {}
            for pair, value in fractions.items():
                shares.setdefault(dataset, {}).setdefault(pair, []).append(float(value))
    datasets = [d for d in datasets if shares.get(d)]
    if not datasets:
        print(f"  [skip] block norms: no head_diagnostics in {condition}/{task_type} runs "
              "(they are written by the current seed_expansion.py; older runs have none)")
        return

    pairs = [p for p in BLOCK_PAIR_COLORS if any(p in shares[d] for d in datasets)]
    pairs += sorted({p for d in datasets for p in shares[d]} - set(pairs))

    width = 0.8 / len(pairs)
    positions = np.arange(len(datasets))
    fig, axis = plt.subplots(figsize=(max(7.0, 1.6 * len(datasets)), 4.8))
    for index, pair in enumerate(pairs):
        means = [float(np.mean(shares[d].get(pair, [np.nan]))) for d in datasets]
        errors = [float(np.std(shares[d].get(pair, [0.0]), ddof=1))
                  if len(shares[d].get(pair, [])) > 1 else 0.0 for d in datasets]
        axis.bar(positions + index * width - 0.4 + width / 2, means, width,
                 yerr=errors, capsize=2.0, label=pair,
                 color=BLOCK_PAIR_COLORS.get(pair, PALETTE[index % len(PALETTE)]),
                 error_kw={"linewidth": 0.7, "ecolor": "#333333"})
    axis.set_xticks(positions)
    axis.set_xticklabels([d.upper() for d in datasets])
    axis.set_xlabel("Dataset (smallest to largest)")
    axis.set_ylabel("Share of $\\|W\\|_F$ (block pair / total)")
    axis.set_title(
        f"Where the learned quadratic form lives — {condition}\n"
        f"{task_type}, {split}, mean ± 1 sd over seeds; cross-block bars are the "
        "interaction terms",
        fontsize=10,
    )
    axis.grid(axis="y", alpha=0.3)
    axis.legend(ncol=min(6, len(pairs)), fontsize=8, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"block_norms_{condition}_{task_type}_{suffix}",
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Prototype routing
# ---------------------------------------------------------------------------

MAX_HEATMAP_ROWS = 60


def figure_prototype_assignment(frame: pd.DataFrame, task_type: str, split: str,
                                suffix: str, condition: str) -> None:
    """Task x prototype assignment matrix for the multitask datasets."""
    subset = frame[(frame["task_type"] == task_type) & (frame["condition"] == condition)]
    for dataset in ordered_datasets(subset, task_type):
        runs = subset[subset["dataset"] == dataset].sort_values("seed")
        matrices = []
        for run_dir in runs["run_dir"]:
            path = Path(run_dir) / "assignment_probs.npy"
            if path.exists():
                matrices.append(np.load(path))
        if not matrices:
            continue
        shapes = {m.shape for m in matrices}
        if len(shapes) > 1:                      # differing prototype counts
            matrices = [matrices[0]]
        matrix = np.mean(np.stack(matrices), axis=0)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            continue                             # single-task set: nothing to cluster

        spec = DATASETS.get(dataset)
        labels = list(spec.target_columns) if spec and spec.target_columns else []
        # Order tasks by their dominant prototype, so tasks that share a
        # decision axis end up adjacent -- that grouping is the point of the
        # figure, and unsorted rows hide it completely.
        order = np.lexsort((-matrix.max(axis=1), matrix.argmax(axis=1)))
        truncated = order.size > MAX_HEATMAP_ROWS
        order = order[:MAX_HEATMAP_ROWS]
        shown = matrix[order]

        height = max(3.0, 0.22 * shown.shape[0] + 1.8)
        fig, axis = plt.subplots(figsize=(max(4.2, 0.55 * shown.shape[1] + 2.6), height))
        image = axis.imshow(shown, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_xlabel("Prototype")
        axis.set_xticks(range(shown.shape[1]))
        axis.set_xticklabels([str(i) for i in range(shown.shape[1])], fontsize=8)
        if labels and len(labels) >= matrix.shape[0]:
            axis.set_yticks(range(shown.shape[0]))
            axis.set_yticklabels([str(labels[i])[:28] for i in order], fontsize=6)
        else:
            axis.set_ylabel("Task")
        title = (f"{dataset.upper()} — prototype assignment α "
                 f"({condition}, {split}, mean over {len(matrices)} seed(s))")
        if truncated:
            title += f"\nfirst {MAX_HEATMAP_ROWS} of {matrix.shape[0]} tasks, grouped by dominant prototype"
        axis.set_title(title, fontsize=10)
        fig.colorbar(image, ax=axis, label="assignment weight")
        fig.tight_layout()
        save_figure(fig, FIGURE_DIR / f"prototype_assignment_{dataset}_{condition}_{suffix}",
                    bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Mean rank across datasets
# ---------------------------------------------------------------------------

def figure_condition_rank(frame: pd.DataFrame, task_type: str, split: str,
                          suffix: str, conv: str) -> None:
    """Mean rank of each condition across datasets (1 = best on that dataset)."""
    subset = frame[(frame["task_type"] == task_type)
                   & (frame["condition"].str.startswith(f"{conv}_"))]
    if subset.empty:
        return
    metric = subset["metric_name"].iloc[0]
    ascending = metric == "rmse"          # RMSE: smaller is rank 1
    cell = (subset.groupby(["dataset", "condition"])["test_metric"]
            .mean().reset_index())
    table = cell.pivot(index="dataset", columns="condition", values="test_metric")
    if table.shape[1] < 2:
        return
    ranks = table.rank(axis=1, ascending=ascending)
    conditions = sorted(ranks.columns, key=sort_key)
    means = ranks[conditions].mean(axis=0)
    stds = ranks[conditions].std(axis=0, ddof=1).fillna(0.0)

    order = means.sort_values().index.tolist()
    fig, axis = plt.subplots(figsize=(7.0, 0.34 * len(order) + 2.2))
    positions = np.arange(len(order))
    axis.errorbar(means[order], positions, xerr=stds[order], fmt="o", capsize=3,
                  color="#4C72B0", ecolor="#9BB4D4", markersize=5.5, linewidth=1.2)
    axis.set_yticks(positions)
    axis.set_yticklabels([condition_label(c) for c in order], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel(f"Mean rank across {ranks.shape[0]} datasets (1 = best)")
    axis.set_title(f"{conv.upper()} ablation ranking — {task_type} ({metric.upper()}, {split})",
                   fontsize=10)
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"condition_rank_{conv}_{task_type}_{suffix}",
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Per-seed paired deltas
# ---------------------------------------------------------------------------

def figure_paired_delta(frame: pd.DataFrame, task_type: str, split: str,
                        suffix: str, conv: str) -> None:
    """Distribution of per-seed (full - ablation) deltas, signed so up = worse.

    This is the quantity the Wilcoxon test consumes. Plotting the raw paired
    differences alongside the test makes the n visible: with five seeds per
    dataset the spread, not the p-value, is what a reader can judge.
    """
    subset = frame[(frame["task_type"] == task_type)
                   & (frame["condition"].str.startswith(f"{conv}_"))]
    if subset.empty:
        return
    metric = subset["metric_name"].iloc[0]
    higher_is_better = metric != "rmse"
    lookup = {(r["condition"], r["dataset"], r["seed"]): r["test_metric"]
              for _, r in subset.iterrows()}

    conditions = sorted({c for c in subset["condition"] if not c.endswith("full_model")},
                        key=sort_key)
    series: list[np.ndarray] = []
    for condition in conditions:
        reference = reference_for(condition)
        deltas = []
        for _, row in subset[subset["condition"] == condition].iterrows():
            full = lookup.get((reference, row["dataset"], row["seed"]))
            if full is None or row["test_metric"] is None:
                continue
            value = float(row["test_metric"])
            deltas.append(full - value if higher_is_better else value - full)
        series.append(np.asarray(deltas, dtype=float))
    keep = [i for i, values in enumerate(series) if values.size]
    if not keep:
        return
    conditions = [conditions[i] for i in keep]
    series = [series[i] for i in keep]

    fig, axis = plt.subplots(figsize=(max(7.0, 0.62 * len(conditions) + 2.0), 4.6))
    box = axis.boxplot(series, widths=0.6, showfliers=False, patch_artist=True)
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PALETTE[index % len(PALETTE)])
        patch.set_alpha(0.45)
        patch.set_linewidth(0.9)
    for element in ("medians", "whiskers", "caps"):
        for artist in box[element]:
            artist.set_color("#333333")
            artist.set_linewidth(0.9)
    rng = np.random.default_rng(0)
    for index, values in enumerate(series, start=1):
        jitter = rng.uniform(-0.17, 0.17, size=values.size)
        axis.scatter(np.full(values.size, index) + jitter, values, s=9,
                     color="#333333", alpha=0.55, linewidths=0, zorder=3)
    axis.axhline(0.0, color="#C44E52", linewidth=1.0, linestyle="--")
    axis.set_xticks(range(1, len(conditions) + 1))
    axis.set_xticklabels([condition_label(c) for c in conditions], rotation=45,
                         ha="right", fontsize=8)
    axis.set_ylabel(f"Δ {metric.upper()} vs full model")
    axis.set_title(
        f"{conv.upper()} — per-seed paired degradation, all {task_type} datasets pooled "
        f"({split})\nabove zero = removing the component made it worse; one point = one "
        "(dataset, seed)",
        fontsize=10,
    )
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, FIGURE_DIR / f"paired_delta_{conv}_{task_type}_{suffix}",
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                        help="Results tree written by scripts/seed_expansion.py.")
    parser.add_argument("--split-type", default="scaffold", choices=SPLIT_TYPE_CHOICES)
    parser.add_argument("--task-type", default="all", choices=[*TASK_TYPES, "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None, metavar="N",
                        help="Restrict to these seeds and tag the filenames with the count, "
                             "matching compute_ci.py / make_figures.py.")
    parser.add_argument("--conditions", nargs="+", default=None, metavar="NAME",
                        help="Conditions to draw the per-condition figures (learning curves, "
                             "block norms, prototypes) for. Default: every *_full_model present.")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Warn instead of exiting non-zero when a tree has no runs.")
    args = parser.parse_args()

    split_type = resolve_split_type(args.split_type)
    label = split_label(split_type)
    keep_seeds = set(args.seeds) if args.seeds else None
    suffix = f"{label}_{len(keep_seeds)}seed" if keep_seeds else label
    runs_root = project_path(args.runs_root)
    tree = runs_root.name
    if tree != "runs":
        suffix = f"{suffix}_{tree[len('runs_'):] if tree.startswith('runs_') else tree}"

    frame = load_runs(runs_root, split_type, keep_seeds)
    if frame.empty:
        message = (f"No runs under {runs_root}/{split_type}. "
                   "Run scripts/seed_expansion.py first.")
        if args.allow_incomplete:
            print(f"[warn] {message}")
            return
        raise SystemExit(message)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    conditions = args.conditions or sorted(
        {c for c in frame["condition"] if c.endswith("full_model")}, key=sort_key)
    task_types = list(TASK_TYPES) if args.task_type == "all" else [args.task_type]

    print(f"Runs: {len(frame)} across {frame['condition'].nunique()} conditions, "
          f"{frame['dataset'].nunique()} datasets, seeds={sorted(frame['seed'].unique())}")
    for task_type in task_types:
        if frame[frame["task_type"] == task_type].empty:
            print(f"[warn] no {task_type} runs — skipping")
            continue
        for condition in conditions:
            figure_learning_curves(frame, task_type, label, suffix, condition)
            figure_block_norms(frame, task_type, label, suffix, condition)
            figure_prototype_assignment(frame, task_type, label, suffix, condition)
        for conv in CONVS:
            figure_condition_rank(frame, task_type, label, suffix, conv)
            figure_paired_delta(frame, task_type, label, suffix, conv)

    written = sorted(FIGURE_DIR.glob("*.png"))
    print(f"\n{len(written)} PNG files in {FIGURE_DIR.relative_to(PROJECT_ROOT)}:")
    for path in written:
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
