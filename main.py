from __future__ import annotations

import argparse
from pathlib import Path

# Must run before torch/numpy load (see common/resources).
from common.resources import apply_thread_limits

apply_thread_limits(verbose=True)

import pandas as pd  # noqa: E402

from common.config import get_project_root  # noqa: E402
from common.datasets import (
    DEFAULT_BENCHMARK_DATASETS,
    available_datasets,
    get_dataset_spec,
    resolve_target_columns,
)
from common.io_utils import ensure_dir
from common.data import resolve_split_type  # noqa: E402
from common.runner import add_general_training_arguments, collect_override_hparams, run_experiment, run_multi_seed
from dual_kd_gnn.main import (
    MODEL_SPEC,
    add_dual_model_arguments,
    collect_dual_hparam_overrides,
    collect_dual_model_kwargs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the dual_kd_gnn model across multiple MoleculeNet datasets.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_BENCHMARK_DATASETS,
        choices=available_datasets(),
        help="Datasets to benchmark sequentially. Defaults to BACE, BBBP, SIDER, Tox21, ClinTox.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=get_project_root() / "results" / "artifacts",
        help="Directory for the aggregated benchmark summary table.",
    )
    add_general_training_arguments(parser)
    add_dual_model_arguments(parser)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    spec = MODEL_SPEC
    model_kwargs = collect_dual_model_kwargs(args)
    hparam_overrides = collect_dual_hparam_overrides(args)
    hparam_overrides.update(
        {key: value for key, value in collect_override_hparams(args).items() if value is not None}
    )

    model_dir = get_project_root() / spec.slug
    summary_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    seeds = args.seeds if args.seeds is not None else [args.seed]
    multi_seed = len(seeds) > 1
    ablation_name = getattr(args, "ablation_name", None)
    split_type = resolve_split_type(getattr(args, "split_type", "scaffold"))

    for name in args.datasets:
        dataset_spec = get_dataset_spec(name)
        data_path = dataset_spec.data_path()
        if not data_path.exists():
            print(
                f"[skip] {name}: dataset file not found at {data_path}. "
                f"Run: python scripts/download_data.py {name}"
            )
            skipped.append(name)
            continue

        target_columns = resolve_target_columns(dataset_spec, data_path)
        print(f"\n{'=' * 72}")
        print(f"Benchmarking {spec.name} on {name} ({len(target_columns)} task(s))")
        if multi_seed:
            print(f"Seeds: {seeds}")
        print(f"{'=' * 72}")

        if multi_seed:
            result = run_multi_seed(
                spec=spec,
                data_path=str(data_path),
                dataset_name=dataset_spec.name,
                seeds=seeds,
                device_name=args.device,
                target_columns=target_columns,
                model_dir=model_dir,
                overrides=hparam_overrides,
                model_kwargs=model_kwargs,
                smiles_column=dataset_spec.smiles_column,
                ablation_name=ablation_name,
                task_type=dataset_spec.task_type,
                split_type=split_type,
            )
            summary_rows.append(
                {
                    "dataset": name,
                    "task_type": dataset_spec.task_type,
                    "metric_name": dataset_spec.metric_name,
                    "num_tasks": len(target_columns),
                    "test_metric_mean": result["mean_test_metric"],
                    "test_metric_std": result["std_test_metric"],
                    "seeds": str(seeds),
                }
            )
        else:
            metrics = run_experiment(
                spec=spec,
                data_path=str(data_path),
                dataset_name=dataset_spec.name,
                seed=seeds[0],
                device_name=args.device,
                target_columns=target_columns,
                model_dir=model_dir,
                overrides=hparam_overrides,
                model_kwargs=model_kwargs,
                smiles_column=dataset_spec.smiles_column,
                ablation_name=ablation_name,
                task_type=dataset_spec.task_type,
                split_type=split_type,
            )
            summary_rows.append(
                {
                    "dataset": name,
                    "task_type": dataset_spec.task_type,
                    "metric_name": dataset_spec.metric_name,
                    "num_tasks": metrics["num_targets"],
                    # test_roc_auc is absent on regression rows; test_metric is
                    # the metric the dataset actually declares.
                    "test_metric": metrics["test_metric"],
                    "best_val_metric": metrics["best_val_metric"],
                    "num_parameters": metrics["num_parameters"],
                    "elapsed_seconds": metrics["elapsed_seconds"],
                }
            )

    if not summary_rows:
        raise SystemExit(
            "No datasets were benchmarked. Download the data first (see commands.md)."
        )

    results_dir = ensure_dir(args.results_dir)
    summary_path = results_dir / "benchmark_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nBenchmark summary (ROC-AUC higher is better; RMSE lower is better)")
    for row in summary_rows:
        metric = str(row["metric_name"]).upper()
        if "test_metric_mean" in row:
            print(
                f"  {str(row['dataset']):<10} "
                f"tasks={row['num_tasks']:<3} "
                f"{metric}={float(row['test_metric_mean']):.4f} ± {float(row['test_metric_std']):.4f}"
                f"  seeds={row['seeds']}"
            )
        else:
            print(
                f"  {str(row['dataset']):<10} "
                f"tasks={row['num_tasks']:<3} "
                f"{metric}={float(row['test_metric']):.4f}"
            )
    if skipped:
        print(f"\nSkipped (missing data): {', '.join(skipped)}")
    print(f"\nSaved benchmark summary to: {summary_path}")
    print("Run `python results/main.py` to build per-dataset curves and the full results table.")


if __name__ == "__main__":
    main()
