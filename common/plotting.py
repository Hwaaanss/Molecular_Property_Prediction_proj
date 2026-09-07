from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from common.io_utils import ensure_dir

FIGURE_DPI = 300
# 300 dpi PNG is what ICML/ICLR camera-ready wants and is the only format
# written by default. A 300 dpi TIFF of a full-page figure is tens of MB even
# LZW-compressed, and a full sweep now writes a figure per run, so TIFF is
# opt-in: DIKAT_FIGURE_FORMATS="png,tiff" (or "pdf" for vector art).
DEFAULT_FIGURE_FORMATS = ("png",)


def figure_formats() -> tuple[str, ...]:
    raw = os.environ.get("DIKAT_FIGURE_FORMATS")
    if not raw:
        return DEFAULT_FIGURE_FORMATS
    formats = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    return formats or DEFAULT_FIGURE_FORMATS


# Backwards-compatible module-level name; resolved once at import.
FIGURE_FORMATS = figure_formats()


def save_figure(fig, output_path: Path, dpi: int = FIGURE_DPI,
                formats: tuple[str, ...] | None = None, **savefig_kwargs) -> list[Path]:
    """Write one figure at publication resolution in every requested format.

    ``output_path``'s suffix is ignored; one file per entry in ``formats`` is
    written next to it. TIFF is LZW-compressed -- a 300 dpi uncompressed TIFF of
    a full-page figure runs to tens of MB, and LZW is lossless.

    Returns the paths actually written; a format whose backend is missing is
    skipped rather than failing the whole run.
    """
    ensure_dir(output_path.parent)
    formats = formats if formats is not None else figure_formats()
    written: list[Path] = []
    for fmt in formats:
        path = output_path.with_suffix(f".{fmt}")
        kwargs = dict(savefig_kwargs)
        if fmt == "tiff":
            kwargs.setdefault("pil_kwargs", {"compression": "tiff_lzw"})
        try:
            fig.savefig(path, dpi=dpi, format=fmt, **kwargs)
            written.append(path)
        except Exception as exc:  # pragma: no cover - depends on the Pillow build
            print(f"  [warn] could not write {path.name}: {exc}")
    return written


def _has_values(series) -> bool:
    return any(value is not None and value == value for value in series)  # value == value drops NaN


def plot_training_curves(
    history_rows: Iterable[dict[str, object]],
    output_path: Path,
    title: str,
    metric_name: str = "roc_auc",
    stage1_epochs: int | None = None,
) -> Path | None:
    """Save loss and metric training curves for a single run.

    ``metric_name`` labels the right-hand panel; it used to say "ROC-AUC"
    unconditionally, which mislabels every regression run's RMSE curve.

    ``stage1_epochs`` draws the phase boundary. The two phases optimize
    different parameter sets against different objectives, so a curve that does
    not mark the switch reads as one training run with an unexplained
    discontinuity at the hand-off.

    Returns the written path stem, or ``None`` when matplotlib is unavailable
    or there is nothing to plot.
    """
    rows = list(history_rows)
    if not rows:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting is best-effort.
        return None

    epochs = [row.get("global_epoch") for row in rows]
    train_loss = [row.get("train_loss") for row in rows]
    val_loss = [row.get("val_loss") for row in rows]
    train_metric = [row.get("train_metric") for row in rows]
    val_metric = [row.get("val_metric") for row in rows]

    label = "RMSE" if metric_name == "rmse" else metric_name.replace("_", "-").upper()
    if stage1_epochs is None:
        phases = [row.get("phase") for row in rows]
        stage1_epochs = sum(1 for phase in phases if phase == "stage1_gcn_kd") or None

    ensure_dir(output_path.parent)
    fig, (loss_ax, auc_ax) = plt.subplots(1, 2, figsize=(14, 5))

    if _has_values(train_loss):
        loss_ax.plot(epochs, train_loss, label="Train loss", linewidth=1.8)
    if _has_values(val_loss):
        loss_ax.plot(epochs, val_loss, label="Val loss", linewidth=1.8)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title(f"{title} — loss")

    if _has_values(train_metric):
        auc_ax.plot(epochs, train_metric, label=f"Train {label}", linewidth=1.8)
    if _has_values(val_metric):
        auc_ax.plot(epochs, val_metric, label=f"Val {label}", linewidth=1.8)
    auc_ax.set_xlabel("Epoch")
    auc_ax.set_ylabel(label)
    auc_ax.set_title(f"{title} — {label}")

    for axis in (loss_ax, auc_ax):
        if stage1_epochs:
            axis.axvline(stage1_epochs + 0.5, color="#888888", linestyle="--", linewidth=1.0)
            axis.annotate("stage 1 │ stage 2", xy=(stage1_epochs + 0.5, 1.005),
                          xycoords=("data", "axes fraction"), ha="center",
                          fontsize=8, color="#555555")
        axis.grid(alpha=0.3)
        axis.legend()

    fig.tight_layout()
    save_figure(fig, output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
