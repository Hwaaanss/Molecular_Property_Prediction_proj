"""Verify the four split protocols on all five datasets, without training.

Computes splits only -- no GPU, no featurization -- and reports, per
(split_type, dataset):

  * train / val / test sizes and their fractions
  * how many tasks have an undefined ROC-AUC in val and in test (single-class
    after dropping missing labels; common/metrics.py skips exactly these)
  * per-task positive rate in test vs. train, and the largest deviation

It then asserts the properties each protocol is supposed to have:

  * scaffold protocols never place one scaffold group into two splits
  * deterministic_scaffold produces identical splits under different seeds
  * random_scaffold and random produce different splits under different seeds
  * every protocol partitions the dataset exactly (no overlap, nothing dropped)

Usage:
  python scripts/verify_splits.py
  python scripts/verify_splits.py --datasets bbbp bace --split-types random
  python scripts/verify_splits.py --csv results/artifacts/revision/split_diagnostics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import (  # noqa: E402
    SPLIT_TYPE_CHOICES,
    SPLIT_TYPES,
    resolve_split_type,
    count_auc_defined_tasks,
    count_positives,
    scaffold_groups,
    split_indices,
)
from common.datasets import DATASETS, TASK_TYPES, datasets_for_task, resolve_target_columns  # noqa: E402

DATASETS_ALL = list(DATASETS.keys())
DATASETS_ORIGINAL = ["bace", "bbbp", "clintox", "sider", "tox21"]
SCAFFOLD_SPLIT_TYPES = ("deterministic_scaffold", "random_scaffold", "label_aware_scaffold")
SEED_A, SEED_B = 42, 7
# Wide multitask sets (ToxCast has 617) would otherwise print a page per split.
MAX_TASKS_PRINTED = 8


def load_dataset(name: str) -> tuple[pd.DataFrame, list[str], str]:
    spec = DATASETS[name]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    return pd.read_csv(data_path), target_columns, spec.smiles_column


def target_moments(dataframe: pd.DataFrame, target_columns: list[str], indices: list[int]):
    """Per-task (mean, std) of non-missing regression targets."""
    if not indices:
        return np.array([]), np.array([])
    labels = dataframe.iloc[indices][target_columns].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(labels, axis=0), np.nanstd(labels, axis=0)


def positive_rates(dataframe: pd.DataFrame, target_columns: list[str], indices: list[int]) -> np.ndarray:
    """Per-task positive rate among rows with a non-missing label."""
    if not indices:
        return np.zeros(len(target_columns))
    labels = dataframe.iloc[indices][target_columns].to_numpy(dtype=float)
    rates = np.zeros(labels.shape[1])
    for task in range(labels.shape[1]):
        values = labels[:, task]
        values = values[~np.isnan(values) & (values != -1.0)]
        rates[task] = float((values == 1.0).mean()) if values.size else float("nan")
    return rates


def assert_partition(name: str, n: int, splits: tuple[list[int], list[int], list[int]]) -> None:
    train_idx, val_idx, test_idx = splits
    combined = list(train_idx) + list(val_idx) + list(test_idx)
    assert len(combined) == n, f"{name}: split covers {len(combined)} of {n} rows"
    assert len(set(combined)) == n, f"{name}: splits overlap or repeat indices"


def assert_groups_intact(name: str, groups: dict[str, list[int]],
                         splits: tuple[list[int], list[int], list[int]]) -> None:
    """No scaffold group may have members in more than one split."""
    owner: dict[int, str] = {}
    for split_name, indices in zip(("train", "val", "test"), splits):
        for idx in indices:
            owner[idx] = split_name
    for scaffold, group in groups.items():
        split_names = {owner[idx] for idx in group}
        assert len(split_names) == 1, (
            f"{name}: scaffold group '{scaffold}' (n={len(group)}) straddles splits {sorted(split_names)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS_ALL, metavar="NAME",
                        help="Dataset names, or a group: 'all', 'classification', 'regression'.")
    parser.add_argument("--split-types", nargs="+", default=list(SPLIT_TYPES),
                        choices=SPLIT_TYPE_CHOICES,
                        help="Protocols to check. 'scaffold' and 'deterministic_scaffold' "
                             "are the same protocol.")
    parser.add_argument("--csv", default=None, help="Optional path to write the diagnostics table to.")
    args = parser.parse_args()
    args.split_types = list(dict.fromkeys(resolve_split_type(s) for s in args.split_types))

    resolved: list[str] = []
    for entry in args.datasets:
        key = entry.lower()
        if key == "all":
            resolved.extend(DATASETS_ALL)
        elif key in TASK_TYPES:
            resolved.extend(datasets_for_task(key))
        else:
            resolved.append(key)
    args.datasets = list(dict.fromkeys(resolved))
    unknown = [name for name in args.datasets if name not in DATASETS]
    if unknown:
        parser.error(f"Unknown dataset(s): {', '.join(unknown)}")

    rows = []
    undefined_by_protocol: dict[str, int] = {split_type: 0 for split_type in args.split_types}
    # The pre-expansion five are dense enough that every task must stay scorable;
    # that is a hard guarantee. ToxCast (617 sparse assays) cannot meet it, and
    # common/metrics.py already skips unscorable tasks, so the wide multitask
    # sets are reported rather than asserted on.
    undefined_original: dict[str, int] = {split_type: 0 for split_type in args.split_types}

    for dataset in args.datasets:
        spec = DATASETS[dataset]
        is_regression = spec.is_regression
        dataframe, target_columns, smiles_column = load_dataset(dataset)
        n = len(dataframe)
        # Both grouping conventions: the standard one (acyclic molecules share the
        # empty scaffold) and the legacy one label_aware_scaffold_split uses.
        groups_standard = scaffold_groups(dataframe, smiles_column)
        groups_legacy = scaffold_groups(dataframe, smiles_column, empty_scaffold_singleton=True)

        print(f"\n{'=' * 78}")
        print(f"{dataset.upper()} [{spec.task_type}]  n={n}  tasks={len(target_columns)}  "
              f"scaffold groups={len(groups_standard)} (legacy grouping: {len(groups_legacy)})")
        print("=" * 78)

        for split_type in args.split_types:
            if split_type == "label_aware_scaffold" and is_regression:
                print(f"\n  [{split_type}] skipped — classification-only protocol")
                continue
            splits = split_indices(
                dataframe,
                split_type=split_type,
                target_columns=target_columns,
                smiles_column=smiles_column,
                seed=SEED_A,
                task_type=spec.task_type,
            )
            train_idx, val_idx, test_idx = splits
            label = f"{dataset}/{split_type}"
            assert_partition(label, n, splits)

            # --- scaffold groups must stay inside one split -------------------
            if split_type in SCAFFOLD_SPLIT_TYPES:
                groups = groups_legacy if split_type == "label_aware_scaffold" else groups_standard
                assert_groups_intact(label, groups, splits)

            # --- seed sensitivity --------------------------------------------
            splits_other_seed = split_indices(
                dataframe,
                split_type=split_type,
                target_columns=target_columns,
                smiles_column=smiles_column,
                seed=SEED_B,
                task_type=spec.task_type,
            )
            same_as_other_seed = [sorted(a) == sorted(b) for a, b in zip(splits, splits_other_seed)]
            if split_type in ("deterministic_scaffold", "label_aware_scaffold"):
                assert all(same_as_other_seed), (
                    f"{label}: expected seed-independent splits, but seed {SEED_A} and {SEED_B} differ"
                )
            else:
                assert not all(same_as_other_seed), (
                    f"{label}: expected seed {SEED_A} and {SEED_B} to give different splits, got identical ones"
                )

            print(f"\n  [{split_type}]")
            print(f"    sizes            train={len(train_idx):6d} ({len(train_idx)/n:5.1%})  "
                  f"val={len(val_idx):6d} ({len(val_idx)/n:5.1%})  "
                  f"test={len(test_idx):6d} ({len(test_idx)/n:5.1%})")
            print(f"    seed-sensitive   "
                  f"{'no (identical at seeds %d/%d)' % (SEED_A, SEED_B) if all(same_as_other_seed) else 'yes'}")

            row = {
                "dataset": dataset,
                "task_type": spec.task_type,
                "split_type": split_type,
                "n_total": n,
                "n_tasks": len(target_columns),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "n_test": len(test_idx),
                "seed_dependent": not all(same_as_other_seed),
            }

            if is_regression:
                # ROC-AUC definability is meaningless here; what matters is that
                # test targets stay on the same scale train was standardized to.
                train_mean, train_std = target_moments(dataframe, target_columns, train_idx)
                test_mean, test_std = target_moments(dataframe, target_columns, test_idx)
                shift = np.abs(test_mean - train_mean) / np.where(train_std < 1e-8, 1.0, train_std)
                max_shift = float(np.nanmax(shift)) if shift.size else 0.0
                worst_task = target_columns[int(np.nanargmax(shift))] if shift.size else "-"
                print(f"    AUC undefined    n/a (regression)")
                print(f"    max |test-train| target mean shift = {max_shift:.3f} train-sd  "
                      f"(task: {worst_task})")
                print("    per-task target mean+-sd  train -> test:")
                for task, tr_m, tr_s, te_m, te_s in zip(
                    target_columns[:MAX_TASKS_PRINTED], train_mean, train_std, test_mean, test_std
                ):
                    print(f"      {task[:28]:28s} {tr_m:8.3f}+-{tr_s:6.3f} -> {te_m:8.3f}+-{te_s:6.3f}")
                row.update({
                    "n_tasks_auc_defined_val": None,
                    "n_tasks_auc_defined_test": None,
                    "n_tasks_auc_undefined_val": None,
                    "n_tasks_auc_undefined_test": None,
                    "max_shift_test_vs_train": round(max_shift, 4),
                    "shift_unit": "train_sd",
                    "worst_shift_task": worst_task,
                })
            else:
                n_defined_val = count_auc_defined_tasks(dataframe, target_columns, val_idx)
                n_defined_test = count_auc_defined_tasks(dataframe, target_columns, test_idx)
                n_undefined_val = len(target_columns) - n_defined_val
                n_undefined_test = len(target_columns) - n_defined_test
                undefined_by_protocol[split_type] += n_undefined_val + n_undefined_test
                if dataset in DATASETS_ORIGINAL:
                    undefined_original[split_type] += n_undefined_val + n_undefined_test

                train_rates = positive_rates(dataframe, target_columns, train_idx)
                test_rates = positive_rates(dataframe, target_columns, test_idx)
                with np.errstate(invalid="ignore"):
                    rate_shift = np.abs(test_rates - train_rates)
                max_shift = float(np.nanmax(rate_shift)) if rate_shift.size else 0.0
                worst_task = target_columns[int(np.nanargmax(rate_shift))] if rate_shift.size else "-"

                print(f"    AUC undefined    val={n_undefined_val}/{len(target_columns)}  "
                      f"test={n_undefined_test}/{len(target_columns)}")
                print(f"    max |test-train| positive-rate shift = {max_shift:.3f}  (task: {worst_task})")
                print("    per-task positive rate  train -> test:")
                for task, train_rate, test_rate in zip(
                    target_columns[:MAX_TASKS_PRINTED], train_rates, test_rates
                ):
                    print(f"      {task[:28]:28s} {train_rate:6.3f} -> {test_rate:6.3f}  "
                          f"({test_rate - train_rate:+.3f})")
                if len(target_columns) > MAX_TASKS_PRINTED:
                    print(f"      ... (+{len(target_columns) - MAX_TASKS_PRINTED} more tasks)")
                row.update({
                    "n_tasks_auc_defined_val": n_defined_val,
                    "n_tasks_auc_defined_test": n_defined_test,
                    "n_tasks_auc_undefined_val": n_undefined_val,
                    "n_tasks_auc_undefined_test": n_undefined_test,
                    "max_shift_test_vs_train": round(max_shift, 4),
                    "shift_unit": "positive_rate",
                    "worst_shift_task": worst_task,
                })

            rows.append(row)

    summary = pd.DataFrame(rows)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print("=" * 78)
    for task_type in TASK_TYPES:
        subset = summary[summary["task_type"] == task_type]
        if subset.empty:
            continue
        print(f"\n--- {task_type} ---")
        print(subset[["dataset", "split_type", "n_train", "n_val", "n_test",
                      "n_tasks_auc_undefined_val", "n_tasks_auc_undefined_test",
                      "max_shift_test_vs_train", "shift_unit",
                      "seed_dependent"]].to_string(index=False))

    print("\nClassification tasks with undefined ROC-AUC (summed over datasets, val+test):")
    print(f"  {'protocol':24s} {'original 5':>12s} {'all datasets':>14s}")
    for split_type in undefined_by_protocol:
        print(f"  {split_type:24s} {undefined_original[split_type]:>12d} "
              f"{undefined_by_protocol[split_type]:>14d}")
    print("  (ToxCast contributes the non-zero 'all datasets' counts: its sparsest assays\n"
          "   have too few labelled molecules for both classes to appear in a 10% split.\n"
          "   common/metrics.py skips those tasks, so the reported mean AUC stays well-defined.)")

    checked_original = [d for d in args.datasets if d in DATASETS_ORIGINAL]
    if "deterministic_scaffold" in args.split_types and checked_original:
        assert undefined_original["deterministic_scaffold"] == 0, (
            "deterministic_scaffold produced tasks with an undefined ROC-AUC on the "
            f"original five datasets: {undefined_original['deterministic_scaffold']}"
        )

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
