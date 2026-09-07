from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "dual_kd_gnn" / "runs"


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for index, row in df.iterrows():
        rendered_row = [str(index)]
        for value in row.tolist():
            if pd.isna(value):
                rendered_row.append("")
            elif isinstance(value, float):
                rendered_row.append(format(value, floatfmt))
            else:
                rendered_row.append(str(value))
        rows.append(rendered_row)

    table_headers = [df.index.name or "dataset", *headers]
    separator = ["---"] * len(table_headers)
    lines = [
        "| " + " | ".join(table_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def collect_latest_runs() -> dict[str, Path]:
    """Map each run directory name to its latest training_log.csv path."""
    latest_runs: dict[str, Path] = {}
    for history_path in RUNS_DIR.glob("*/training_log.csv"):
        metrics_path = history_path.with_name("metrics.json")
        if not metrics_path.exists():
            continue
        run_name = history_path.parent.name
        existing = latest_runs.get(run_name)
        if existing is None or history_path.stat().st_mtime > existing.stat().st_mtime:
            latest_runs[run_name] = history_path
    return latest_runs


def load_run_records() -> list[dict[str, object]]:
    records = []
    for history_path in collect_latest_runs().values():
        metrics = json.loads(history_path.with_name("metrics.json").read_text(encoding="utf-8"))
        history_df = pd.read_csv(history_path)
        records.append(
            {
                "dataset_name": metrics.get("dataset_name", history_path.parent.name),
                "history": history_df,
                "metrics": metrics,
            }
        )
    records.sort(key=lambda record: str(record["dataset_name"]))
    return records


def plot_validation_auc_curves(records, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 7))
    plotted = False
    for record in records:
        history = record["history"]
        if "global_epoch" not in history.columns or "val_metric" not in history.columns:
            continue
        history = history.dropna(subset=["val_metric"])
        if history.empty:
            continue
        plt.plot(
            history["global_epoch"],
            history["val_metric"],
            linewidth=2,
            label=str(record["dataset_name"]),
        )
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation ROC-AUC")
    plt.title("dual_kd_gnn validation ROC-AUC across datasets")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "val_auc_curves.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_test_auc_bars(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = metrics_df.dropna(subset=["test_roc_auc"])
    if plot_df.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(plot_df["dataset_name"], plot_df["test_roc_auc"], color="#4C72B0")
    for bar, value in zip(bars, plot_df["test_roc_auc"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    plt.ylim(0.0, 1.05)
    plt.xlabel("Dataset")
    plt.ylabel("Test ROC-AUC")
    plt.title("dual_kd_gnn test ROC-AUC by dataset")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "test_auc_by_dataset.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_performance_tables(records, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = [record["metrics"] for record in records]
    metrics_df = pd.DataFrame(metrics_rows)
    if metrics_df.empty:
        return

    metrics_df = metrics_df.sort_values("dataset_name").reset_index(drop=True)
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)
    plot_test_auc_bars(metrics_df, output_dir)

    table_columns = [
        column
        for column in ["num_targets", "best_val_auc", "best_epoch", "test_roc_auc", "num_parameters", "elapsed_seconds"]
        if column in metrics_df.columns
    ]
    table = metrics_df.set_index("dataset_name")[table_columns]
    table.index.name = "dataset"
    table.to_csv(output_dir / "results_table.csv")
    (output_dir / "results_table.md").write_text(
        dataframe_to_markdown(table, floatfmt=".4f"),
        encoding="utf-8",
    )


def main() -> None:
    records = load_run_records()
    if not records:
        raise SystemExit(
            "No dual_kd_gnn run artifacts were found. Train at least one dataset first "
            "(see commands.md)."
        )

    artifacts_dir = PROJECT_ROOT / "results" / "artifacts"
    plot_validation_auc_curves(records, artifacts_dir)
    save_performance_tables(records, artifacts_dir)
    print(f"Saved aggregated artifacts to: {artifacts_dir}")


if __name__ == "__main__":
    main()
