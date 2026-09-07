from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_history_rows(run_dir: Path, rows: Iterable[dict[str, object]]) -> Path:
    run_dir = ensure_dir(run_dir)
    rows = list(rows)
    history_path = run_dir / "training_log.csv"
    fieldnames: list[str] = []
    if rows:
        fieldnames = list(rows[0].keys())
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    return history_path


def save_json(path: Path, payload: dict[str, object]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    return path


def save_run_artifacts(
    run_dir: Path,
    history_rows: Iterable[dict[str, object]],
    metrics: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> None:
    ensure_dir(run_dir)
    save_history_rows(run_dir, history_rows)
    save_json(run_dir / "metrics.json", metrics)
    metadata_payload = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    if metadata:
        metadata_payload.update(metadata)
    save_json(run_dir / "run_metadata.json", metadata_payload)
