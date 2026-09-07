"""Revision experiments triggered by Stage 3 peer review.

Three sub-tasks consolidated into one script for server transfer convenience:

  --task b1   MMFF94 conformer embedding failure rate per dataset (Reviewer R3).
              Counts molecules whose physical-branch features degenerate to zero.
              Fast (~5-10 minutes total).

  --task b2   Re-run full_model under random split (MoleculeNet original
              protocol) for SIDER, Tox21, ClinTox to enable apples-to-apples
              comparison with MLFGNN (Reviewer R2).
              Slow (15 runs at ~10-30 min each; ~3-6 GPU hours total).

  --task c1   Paired Wilcoxon signed-rank tests for every (ablation, dataset)
              cell using per-seed AUCs from ablation/runs/ (Reviewer R5).
              Fast (~30 seconds).

  --task all  Run B1, then B2, then C1, in sequence.

Usage on server:

  conda activate dualgnn
  python scripts/revision_experiments.py --task b1
  python scripts/revision_experiments.py --task b2
  python scripts/revision_experiments.py --task b2 --datasets sider          # subset
  python scripts/revision_experiments.py --task b2 --seeds 42                # one seed only
  python scripts/revision_experiments.py --task c1

  python scripts/revision_experiments.py --task all                          # everything

Outputs land in results/artifacts/revision/ and ablation/runs/full_model_random/.
The script is idempotent: B2 skips runs whose metrics.json already exists.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Must run before torch/numpy load (see common/resources).
from common.resources import apply_thread_limits  # noqa: E402

apply_thread_limits()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from common.config import ablation_runs_root, project_path, set_seed, get_device  # noqa: E402
from common.data import (  # noqa: E402
    SPLIT_TYPE_CHOICES,
    MoleculeDualDataset,
    random_split,
    resolve_split_type,
    split_label,
)
from common.datasets import (  # noqa: E402
    DATASETS,
    TASK_TYPES,
    datasets_for_task,
    resolve_target_columns,
)
from common.trainer import Trainer  # noqa: E402
from dual_kd_gnn.model import GNN_CONV_TYPES, DualDistillationModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATASETS_ALL = list(DATASETS.keys())
DATASETS_RANDOM = ["sider", "tox21", "clintox"]  # MLFGNN uses random split for these
SEEDS = [0, 1, 2, 3, 42]
# Current grid: <conv>_full_model and <conv>_no_<factors>, one series per
# encoder. Each ablation pairs against its own encoder's full model -- pairing a
# GINE cell against a GCN reference would confound the ablation with the
# encoder swap, which changes every parameter shape.
FACTOR_ORDER = ("3d", "infonce", "codebook", "fp")
CURRENT_ABLATIONS = [
    f"{conv}_no_" + "_".join(removed)
    for conv in GNN_CONV_TYPES
    for size in range(1, len(FACTOR_ORDER) + 1)
    for removed in itertools.combinations(FACTOR_ORDER, size)
]
# Pre-refactor conditions (transformer-stage architecture). Kept so the same
# command still analyses ablation/runs; cells with no data are skipped, so the
# two generations coexist in one list without interfering.
LEGACY_ABLATIONS = ["a1_no_phys", "a2_no_infonce", "a4_no_codebook",
                    "a12_no_phys_infonce", "a14_no_phys_codebook", "a24_no_infonce_codebook",
                    "a124_no_all_three", "a3_no_mse_kd", "a5_linear_head",
                    "a8_no_fp", "a18_no_phys_fp", "a48_no_codebook_fp", "fp_from_stage1",
                    "fp_no_block_proj", "gine_a8_no_fp"]
ABLATIONS = CURRENT_ABLATIONS + LEGACY_ABLATIONS

DEFAULT_REFERENCE = "full_model"
LEGACY_ABLATION_REFERENCE = {
    "a8_no_fp": "full_model_fp",
    "a18_no_phys_fp": "full_model_fp",
    "a48_no_codebook_fp": "full_model_fp",
    "fp_from_stage1": "full_model_fp",
    "fp_no_block_proj": "full_model_fp",
    "gine_a8_no_fp": "gine_full_model_fp",
}


def reference_for(condition: str) -> str:
    """The condition an ablation is paired against."""
    conv = condition.split("_", 1)[0]
    if conv in GNN_CONV_TYPES and condition.startswith(f"{conv}_no_"):
        return f"{conv}_full_model"
    return LEGACY_ABLATION_REFERENCE.get(condition, DEFAULT_REFERENCE)


OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "revision"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "ablation" / "runs_v2"
RANDOM_RUNS_DIR = PROJECT_ROOT / "ablation" / "runs" / "full_model_random"


# ============================================================================
# Task B1: MMFF94 conformer embedding failure rate per dataset (parallelized)
# ============================================================================

def _embed_one(smi: str) -> str:
    """Classify one SMILES as one of: 'success', 'invalid_smiles',
    'embed_fail', 'mmff_fail'. Module-level for multiprocessing pickling.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "invalid_smiles"
    mol_h = Chem.AddHs(mol)
    try:
        embed_status = AllChem.EmbedMolecule(mol_h, randomSeed=42)
    except Exception:
        embed_status = -1
    if embed_status < 0:
        return "embed_fail"
    try:
        mmff_status = AllChem.MMFFOptimizeMolecule(mol_h)
    except Exception:
        mmff_status = -1
    if mmff_status not in (0, 1):
        return "mmff_fail"
    try:
        _ = mol_h.GetConformer()
        return "success"
    except Exception:
        return "embed_fail"


def task_b1(args) -> None:
    from multiprocessing import Pool, cpu_count
    print("\n[Task B1] MMFF94 conformer embedding failure rate per dataset")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    n_workers = max(1, min(args.b1_workers, cpu_count()))
    print(f"  Using {n_workers} parallel workers")

    for ds_name in args.datasets:
        spec = DATASETS[ds_name]
        df = pd.read_csv(spec.data_path())
        smiles_list = df[spec.smiles_column].tolist()
        n_total = len(smiles_list)

        counts = {"success": 0, "invalid_smiles": 0, "embed_fail": 0, "mmff_fail": 0}
        t0 = time.time()
        progress_every = max(50, n_total // 20)  # 5% increments
        with Pool(processes=n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_embed_one, smiles_list, chunksize=8), start=1):
                counts[result] += 1
                if i % progress_every == 0 or i == n_total:
                    pct = 100.0 * i / n_total
                    print(f"    {ds_name}: {i}/{n_total} ({pct:5.1f}%)  elapsed={time.time()-t0:.0f}s", flush=True)

        elapsed = time.time() - t0
        n_total_fail = counts["invalid_smiles"] + counts["embed_fail"] + counts["mmff_fail"]
        failure_rate = n_total_fail / max(n_total, 1)
        row = {
            "dataset": ds_name,
            "n_total": n_total,
            "n_success": counts["success"],
            "n_invalid_smiles": counts["invalid_smiles"],
            "n_embed_fail": counts["embed_fail"],
            "n_mmff_fail": counts["mmff_fail"],
            "n_total_fail": n_total_fail,
            "failure_rate": round(failure_rate, 4),
            "elapsed_seconds": round(elapsed, 2),
        }
        rows.append(row)
        print(
            f"  [{ds_name:10s}] n={n_total:5d}  success={counts['success']:5d}  "
            f"smiles_invalid={counts['invalid_smiles']:4d}  embed_fail={counts['embed_fail']:4d}  "
            f"mmff_fail={counts['mmff_fail']:4d}  fail_rate={failure_rate:.3f}  ({elapsed:.1f}s)"
        )

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "mmff_failure_rates.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[B1] Saved: {out_path.relative_to(PROJECT_ROOT)}")


# ============================================================================
# Task B2: Re-run full_model under random split (MLFGNN-compatible protocol)
# ============================================================================

def load_best_config(dataset: str) -> dict:
    path = PROJECT_ROOT / "dual_kd_gnn" / "optuna" / f"{dataset}_xkd" / "best_config.json"
    if not path.exists():
        raise FileNotFoundError(f"best_config.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _train_single_random(dataset: str, seed: int, device: torch.device) -> dict:
    """Train one (dataset, seed) configuration under random split."""
    run_dir = RANDOM_RUNS_DIR / f"{dataset}_seed{seed}"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        print(f"  [skip] {dataset}_seed{seed} already exists")
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    num_classes = len(target_columns)
    df = pd.read_csv(data_path)
    n = len(df)
    train_idx, val_idx, test_idx = random_split(n, 0.8, 0.1, seed)
    print(f"  random split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    train_ds = MoleculeDualDataset(data_path, target_columns, train_idx, smiles_column=spec.smiles_column)
    val_ds = MoleculeDualDataset(data_path, target_columns, val_idx, smiles_column=spec.smiles_column)
    test_ds = MoleculeDualDataset(data_path, target_columns, test_idx, smiles_column=spec.smiles_column)

    best_config = load_best_config(dataset)
    model_kwargs = dict(best_config["model_kwargs"])
    hparams = dict(best_config["hparams"])

    model = DualDistillationModel(num_classes=num_classes, **model_kwargs)
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        num_classes=num_classes,
        **hparams,
    )
    t0 = time.time()
    trainer.train()
    test_auc = float(trainer.evaluate(test_ds, batch_size=int(hparams["batch_size"])))
    elapsed = time.time() - t0

    n_params = int(sum(p.numel() for p in model.parameters()))
    metrics = {
        "model_name": "Dual_KD_GNN_random_split",
        "model_slug": "dual_kd_gnn",
        "dataset_name": f"{dataset}_full_random_seed{seed}",
        "split_protocol": "random_80_10_10",
        "target_columns": target_columns,
        "num_targets": num_classes,
        "best_val_auc": float(trainer.best_val_auc),
        "best_epoch": int(trainer.best_epoch),
        "test_roc_auc": test_auc,
        "num_parameters": n_params,
        "uses_dual_features": True,
        "elapsed_seconds": round(elapsed, 2),
        "seed": seed,
        "ablation_name": "full_model_random",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"  done: val={metrics['best_val_auc']:.4f}  test={metrics['test_roc_auc']:.4f}  "
        f"({elapsed/60:.1f} min)"
    )
    return metrics


def task_b2(args) -> None:
    print("\n[Task B2] Re-run full_model under MoleculeNet random split (SIDER, Tox21, ClinTox)")
    device = get_device(args.device)
    print(f"Device: {device}")

    datasets = [d for d in args.datasets if d in DATASETS_RANDOM]
    if not datasets:
        print(f"  [warn] no random-split-applicable datasets in {args.datasets}; "
              f"valid choices: {DATASETS_RANDOM}")
        return

    all_metrics = []
    for ds_name in datasets:
        print(f"\n--- Dataset: {ds_name} ---")
        for seed in args.seeds:
            print(f"\n[{ds_name} seed={seed}]")
            try:
                m = _train_single_random(ds_name, seed, device)
                all_metrics.append(m)
            except Exception as e:
                print(f"  [error] {ds_name}_seed{seed}: {e}")
                continue

    if all_metrics:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        per_dataset = defaultdict(list)
        for m in all_metrics:
            per_dataset[m["dataset_name"].split("_full_random_seed")[0]].append(m["test_roc_auc"])
        for ds, aucs in per_dataset.items():
            arr = np.array(aucs, dtype=float)
            summary_rows.append({
                "dataset": ds,
                "split_protocol": "random_80_10_10",
                "n_seeds": len(arr),
                "mean_test_roc_auc": round(float(arr.mean()), 4),
                "std_test_roc_auc": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
                "per_seed_aucs": str([round(a, 4) for a in aucs]),
            })
        out_path = OUT_DIR / "random_split_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out_path, index=False)
        print(f"\n[B2] Saved summary: {out_path.relative_to(PROJECT_ROOT)}")
        print(f"[B2] Per-run metrics under: {RANDOM_RUNS_DIR.relative_to(PROJECT_ROOT)}/")


# ============================================================================
# Task C1: Paired Wilcoxon signed-rank tests (ablation vs full_model per cell)
# ============================================================================

def _load_per_seed_aucs(condition: str, dataset: str, runs_root: Path) -> dict[int, float]:
    """Read test_roc_auc per seed from <runs_root>/<condition>/<dataset>_seed<N>/metrics.json.

    Automatically discovers ALL seed folders (not just the original 5) so n=15
    seed expansion runs are picked up without modifying this function.
    """
    import re
    base = runs_root / condition
    out: dict[int, float] = {}
    if not base.exists():
        return out
    pattern = re.compile(rf"^{re.escape(dataset)}_seed(\d+)$")
    for run_dir in base.iterdir():
        m = pattern.match(run_dir.name)
        if not m:
            continue
        seed = int(m.group(1))
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        # Runs written before regression support only carry test_roc_auc.
        primary = data.get("test_metric", data.get("test_roc_auc"))
        if primary is not None:
            out[seed] = float(primary)
    return out


def task_c1(args) -> None:
    legacy = getattr(args, "legacy", False)
    split_type = resolve_split_type(args.split_type)
    suffix = "legacy_label_aware" if legacy else split_label(split_type)
    # Same filename rule as compute_ci.py. Without it every results tree writes
    # to wilcoxon_tests_<task>_<split>.csv, so running the v4 tree after the v3
    # tree silently overwrites the v3 table with no trace that it happened --
    # and the seed count is invisible in a file whose n sets the p-value floor.
    keep_seeds = set(args.seeds) if getattr(args, "seeds", None) else None
    if keep_seeds:
        suffix = f"{suffix}_{len(keep_seeds)}seed"
    # Relative paths are project-relative, so a documented command means the
    # same thing from any working directory.
    runs_root_arg = project_path(args.runs_root)
    tree_name = runs_root_arg.name
    if not legacy and tree_name != "runs":
        suffix = f"{suffix}_{tree_name[len('runs_'):] if tree_name.startswith('runs_') else tree_name}"
    runs_root = (
        ablation_runs_root(split_type, legacy=legacy) if legacy
        else runs_root_arg / split_type
    )
    task_types = list(TASK_TYPES) if args.task_type == "all" else [args.task_type]
    print("\n[Task C1] Paired Wilcoxon signed-rank tests (ablation vs full_model)")
    print(f"  split protocol: {suffix}")
    print(f"  reading: {runs_root}")
    if not runs_root.exists():
        print("  [warn] directory does not exist — no runs for this protocol yet")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ablations = [a for a in ABLATIONS if a != "full_model"]
    for task_type in task_types:
        rows = []
        for ablation in ablations:
            reference = reference_for(ablation)
            for dataset in datasets_for_task(task_type):
                spec = DATASETS[dataset]
                full = _load_per_seed_aucs(reference, dataset, runs_root)
                abl = _load_per_seed_aucs(ablation, dataset, runs_root)
                # Restrict to the requested seeds. Without this a dataset swept
                # twice contributes n=15 while a newer one contributes n=5, and
                # the p-values in one table are not on a common footing -- the
                # Wilcoxon floor alone moves from 0.0625 (n=5) to 6.1e-5 (n=15).
                if keep_seeds is not None:
                    full = {s_: v for s_, v in full.items() if s_ in keep_seeds}
                    abl = {s_: v for s_, v in abl.items() if s_ in keep_seeds}
                shared_seeds = sorted(set(full.keys()) & set(abl.keys()))
                if len(shared_seeds) < 3:
                    continue
                full_values = np.array([full[s] for s in shared_seeds], dtype=float)
                abl_values = np.array([abl[s] for s in shared_seeds], dtype=float)
                deltas = full_values - abl_values

                try:
                    stat, pval = wilcoxon(full_values, abl_values, zero_method="wilcox",
                                          alternative="two-sided")
                    stat = float(stat)
                    pval = float(pval)
                except ValueError as e:
                    # All-zero differences trigger ValueError in scipy wilcoxon
                    stat = float("nan")
                    pval = float("nan")
                    print(f"  [warn] {ablation} x {dataset}: wilcoxon error: {e}")

                # A positive delta means full_model scored higher, which is an
                # improvement for ROC-AUC but a regression for RMSE.
                full_is_better = (
                    deltas.mean() > 0 if spec.greater_is_better else deltas.mean() < 0
                )
                row = {
                    "ablation": ablation,
                    "reference": reference,
                    "dataset": dataset,
                    "task_type": task_type,
                    "metric_name": spec.metric_name,
                    "split_protocol": suffix,
                    "n_paired_seeds": len(shared_seeds),
                    "seeds": str(shared_seeds),
                    "mean_full_metric": round(float(full_values.mean()), 4),
                    "mean_ablation_metric": round(float(abl_values.mean()), 4),
                    "mean_delta_full_minus_ablation": round(float(deltas.mean()), 4),
                    "median_delta": round(float(np.median(deltas)), 4),
                    "full_model_better": bool(full_is_better),
                    "wilcoxon_statistic": round(stat, 6) if not np.isnan(stat) else "nan",
                    "p_value": round(pval, 6) if not np.isnan(pval) else "nan",
                    "significant_at_0.05": (not np.isnan(pval)) and pval < 0.05,
                    "significant_at_0.01": (not np.isnan(pval)) and pval < 0.01,
                }
                rows.append(row)
                sig05 = "*" if row["significant_at_0.05"] else " "
                sig01 = "**" if row["significant_at_0.01"] else "  "
                pstr = f"{pval:.4f}" if not np.isnan(pval) else "  nan "
                print(
                    f"  {ablation:24s} x {dataset:9s} ({spec.metric_name:7s})  "
                    f"delta={row['mean_delta_full_minus_ablation']:+.4f}  p={pstr} {sig05}{sig01}"
                )

        out_path = OUT_DIR / f"wilcoxon_tests_{task_type}_{suffix}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        sig05_count = sum(1 for r in rows if r["significant_at_0.05"] is True)
        sig01_count = sum(1 for r in rows if r["significant_at_0.01"] is True)
        print(f"[C1] {task_type}: {len(rows)} tests; {sig05_count} significant at p<0.05, "
              f"{sig01_count} at p<0.01  ->  {out_path.relative_to(PROJECT_ROOT)}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=["b1", "b2", "c1", "all"], required=True)
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ALL,
                        choices=DATASETS_ALL,
                        help="Datasets to process. Default: all 5. B2 ignores BACE and BBBP "
                             "(MLFGNN uses scaffold split for those).")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds to use. B2 runs them; C1 restricts the paired tests to "
                             "them and tags the output filename with the count, so a table "
                             "never mixes cells with different n. Default: 0 1 2 3 42.")
    parser.add_argument("--device", default=None,
                        help="Compute device for B2. Default: auto (cuda > mps > cpu).")
    parser.add_argument("--b1-workers", type=int, default=8,
                        help="Parallel workers for B1 MMFF embedding. Default: 8.")
    parser.add_argument("--split-type", default="scaffold", choices=SPLIT_TYPE_CHOICES,
                        help="C1 only: which split protocol to read from ablation/runs/. "
                             "Default: scaffold ('deterministic_scaffold' means the same).")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                        help="Results tree for task c1. Default: ablation/runs_v2. "
                             "Pass ablation/runs for the pre-refactor results.")
    parser.add_argument("--legacy", action="store_true",
                        help="C1 only: read the flat pre-refactor layout "
                             "ablation/runs/<condition>/... (the label-aware results).")
    parser.add_argument("--task-type", default="all", choices=[*TASK_TYPES, "all"],
                        help="C1 only: which task type to test. Default: all, which writes "
                             "one CSV per task type (never mixed).")
    args = parser.parse_args()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Task: {args.task}")
    print(f"Datasets: {args.datasets}")
    if args.task in ("b2", "all"):
        print(f"Seeds: {args.seeds}")

    if args.task == "b1":
        task_b1(args)
    elif args.task == "b2":
        task_b2(args)
    elif args.task == "c1":
        task_c1(args)
    elif args.task == "all":
        task_b1(args)
        task_b2(args)
        task_c1(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
