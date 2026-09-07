"""Download the benchmark datasets into data/.

Classification: BACE, BBBP, SIDER, Tox21, ClinTox, ToxCast, HIV.
Regression:     ESOL, FreeSolv, Lipophilicity, CEP, Malaria.

Examples
--------
    python scripts/download_data.py                    # download every dataset
    python scripts/download_data.py all                # same as above
    python scripts/download_data.py bbbp bace          # download a subset
    python scripts/download_data.py classification     # every classification set
    python scripts/download_data.py regression         # every regression set
    python scripts/download_data.py --force tox21

Gzipped sources (``.csv.gz``) are decompressed to plain ``.csv`` under data/.
Sources that ship without a header row get one written from the spec's
``header_names`` so every dataset is a plain, self-describing CSV on disk.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.datasets import (  # noqa: E402
    DATASETS,
    TASK_TYPES,
    available_datasets,
    datasets_for_task,
    get_dataset_spec,
    resolve_target_columns,
)


def download_one(name: str, data_dir: Path, force: bool) -> None:
    spec = get_dataset_spec(name)
    destination = data_dir / spec.csv_filename
    if destination.exists() and not force:
        print(f"[skip] {name}: already present at {destination} (use --force to re-download).")
        return

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {name}: {spec.url}")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(spec.url, tmp_path)
        if spec.url.endswith(".gz"):
            with gzip.open(tmp_path, "rb") as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(tmp_path, destination)
        if spec.header_names is not None:
            _write_header(destination, spec.header_names)
    finally:
        tmp_path.unlink(missing_ok=True)
    _report(name, spec, destination)


def _write_header(destination: Path, header_names: tuple[str, ...]) -> None:
    """Prepend a header row to a headerless CSV, in place."""
    import csv

    with destination.open("r", encoding="utf-8", newline="") as handle:
        first_line = handle.readline()
    existing = next(csv.reader([first_line])) if first_line else []
    if list(existing[:len(header_names)]) == list(header_names):
        return  # already has the header (e.g. re-run without --force)
    if len(existing) != len(header_names):
        raise ValueError(
            f"{destination.name}: expected {len(header_names)} columns to match "
            f"header_names={header_names}, found {len(existing)}"
        )
    body = destination.read_text(encoding="utf-8")
    destination.write_text(",".join(header_names) + "\n" + body, encoding="utf-8")
    print(f"  [fixup] wrote header {header_names}")


def _report(name: str, spec, destination: Path) -> None:
    import pandas as pd

    try:
        frame = pd.read_csv(destination)
        targets = resolve_target_columns(spec, destination)
        missing = [column for column in targets if column not in frame.columns]
        if missing:
            raise ValueError(f"target column(s) not in CSV: {missing}")
        if spec.smiles_column not in frame.columns:
            raise ValueError(f"smiles column '{spec.smiles_column}' not in CSV")
        print(f"[done] {name}: {destination}  "
              f"n={len(frame)}  tasks={len(targets)}  type={spec.task_type}")
    except Exception as exc:  # noqa: BLE001 - report and keep going on the other datasets
        print(f"[warn] {name}: saved to {destination} but failed validation: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets into data/.")
    parser.add_argument(
        "datasets",
        nargs="*",
        default=["all"],
        help=f"Datasets to download (default: all). Choices: {', '.join(available_datasets())}, "
             f"{', '.join(TASK_TYPES)}, all.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Destination directory (default: ./data).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists.")
    args = parser.parse_args()

    requested = args.datasets or ["all"]
    names: list[str] = []
    for entry in requested:
        key = entry.lower()
        if key == "all":
            names.extend(DATASETS.keys())
        elif key in TASK_TYPES:
            names.extend(datasets_for_task(key))
        else:
            names.append(key)
    names = list(dict.fromkeys(names))  # de-duplicate, keep order
    unknown = [name for name in names if name not in DATASETS]
    if unknown:
        parser.error(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Choices: {', '.join(available_datasets())}, {', '.join(TASK_TYPES)}, all."
        )

    for name in names:
        download_one(name, args.data_dir, args.force)


if __name__ == "__main__":
    main()
