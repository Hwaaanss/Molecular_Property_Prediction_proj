from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from common.config import DEFAULT_TARGET_COLUMNS, get_project_root

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a MoleculeNet-style molecular property dataset.

    ``target_columns`` may be ``None`` to mean "every column except the SMILES
    column and anything in ``exclude_columns``"; this is convenient for wide
    multitask sets such as SIDER and ToxCast.

    ``task_type`` decides the loss, the reported metric and whether a higher or
    lower score is better: classification sets are trained with masked BCE and
    scored by ROC-AUC (higher better), regression sets with masked MSE on
    standardized targets and scored by RMSE in the original units (lower better).

    ``exclude_columns`` lists non-target columns that are neither SMILES nor a
    label -- identifiers (``mol_id``), precomputed descriptors (Delaney ships
    nine of them) or a second encoding of the same label (HIV's ``activity`` is
    the string form of ``HIV_active``). They would otherwise be picked up as
    targets by the ``target_columns=None`` rule.
    """

    name: str
    csv_filename: str
    smiles_column: str
    url: str
    target_columns: tuple[str, ...] | None = None
    description: str = ""
    task_type: TaskType = "classification"
    exclude_columns: tuple[str, ...] = ()
    # Written by scripts/download_data.py when the upstream file needs fixing up
    # before it is usable (see malaria below).
    header_names: tuple[str, ...] | None = None

    @property
    def is_regression(self) -> bool:
        return self.task_type == "regression"

    @property
    def metric_name(self) -> str:
        return "rmse" if self.is_regression else "roc_auc"

    @property
    def greater_is_better(self) -> bool:
        return not self.is_regression

    def data_path(self, data_dir: Path | None = None) -> Path:
        data_dir = data_dir if data_dir is not None else (get_project_root() / "data")
        return data_dir / self.csv_filename


# DeepChem hosts the canonical MoleculeNet CSVs. ``.csv.gz`` files are gzipped;
# the downloader (scripts/download_data.py) decompresses them to plain CSV.
_S3 = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
# CEP and Malaria were dropped from the DeepChem bucket; the AttentiveFP repo
# carries the same MoleculeNet-derived CSVs and is the source the AttentiveFP
# paper (a standard baseline on both sets) reports against.
_ATTENTIVEFP = "https://raw.githubusercontent.com/OpenDrugAI/AttentiveFP/master/data"

DATASETS: dict[str, DatasetSpec] = {
    "tox21": DatasetSpec(
        name="tox21",
        csv_filename="tox21.csv",
        smiles_column="smiles",
        url=f"{_S3}/tox21.csv.gz",
        target_columns=tuple(DEFAULT_TARGET_COLUMNS),
        description="Tox21 - 12 nuclear-receptor / stress-response toxicity assays (multitask binary).",
    ),
    "bbbp": DatasetSpec(
        name="bbbp",
        csv_filename="bbbp.csv",
        smiles_column="smiles",
        url=f"{_S3}/BBBP.csv",
        target_columns=("p_np",),
        description="BBBP - blood-brain barrier penetration (single binary task).",
    ),
    "bace": DatasetSpec(
        name="bace",
        csv_filename="bace.csv",
        smiles_column="mol",
        url=f"{_S3}/bace.csv",
        target_columns=("Class",),
        description="BACE - beta-secretase 1 (BACE-1) inhibition (single binary task).",
    ),
    "sider": DatasetSpec(
        name="sider",
        csv_filename="sider.csv",
        smiles_column="smiles",
        url=f"{_S3}/sider.csv.gz",
        target_columns=None,  # 27 side-effect system-organ-class tasks (all non-SMILES columns).
        description="SIDER - 27 marketed-drug adverse-reaction system-organ-class tasks (multitask binary).",
    ),
    "clintox": DatasetSpec(
        name="clintox",
        csv_filename="clintox.csv",
        smiles_column="smiles",
        url=f"{_S3}/clintox.csv.gz",
        target_columns=("FDA_APPROVED", "CT_TOX"),
        description="ClinTox - FDA approval status vs. clinical-trial toxicity (2 binary tasks).",
    ),
    "toxcast": DatasetSpec(
        name="toxcast",
        csv_filename="toxcast.csv",
        smiles_column="smiles",
        url=f"{_S3}/toxcast_data.csv.gz",
        target_columns=None,  # 617 in-vitro assay endpoints.
        description="ToxCast - 617 EPA in-vitro high-throughput toxicology endpoints (multitask binary).",
    ),
    "hiv": DatasetSpec(
        name="hiv",
        csv_filename="hiv.csv",
        smiles_column="smiles",
        url=f"{_S3}/HIV.csv",
        target_columns=("HIV_active",),
        # 'activity' is the same label as a 3-level string (CI/CM/CA), not a target.
        exclude_columns=("activity",),
        description="HIV - inhibition of HIV replication (single binary task).",
    ),
    "esol": DatasetSpec(
        name="esol",
        csv_filename="esol.csv",
        smiles_column="smiles",
        url=f"{_S3}/delaney-processed.csv",
        target_columns=("measured log solubility in mols per litre",),
        task_type="regression",
        description="ESOL (Delaney) - aqueous solubility, log mol/L (single regression task).",
    ),
    "freesolv": DatasetSpec(
        name="freesolv",
        csv_filename="freesolv.csv",
        smiles_column="smiles",
        url=f"{_S3}/SAMPL.csv",
        target_columns=("expt",),
        task_type="regression",
        description="FreeSolv (SAMPL) - experimental hydration free energy, kcal/mol "
                    "(single regression task).",
    ),
    "lipo": DatasetSpec(
        name="lipo",
        csv_filename="lipo.csv",
        smiles_column="smiles",
        url=f"{_S3}/Lipophilicity.csv",
        target_columns=("exp",),
        task_type="regression",
        description="Lipophilicity - octanol/water distribution coefficient logD at pH 7.4 "
                    "(single regression task).",
    ),
    "cep": DatasetSpec(
        name="cep",
        csv_filename="cep.csv",
        smiles_column="smiles",
        url=f"{_ATTENTIVEFP}/cep-processed.csv",
        target_columns=("PCE",),
        task_type="regression",
        description="CEP (Harvard Clean Energy Project) - photovoltaic power conversion "
                    "efficiency (single regression task).",
    ),
    "malaria": DatasetSpec(
        name="malaria",
        csv_filename="malaria.csv",
        smiles_column="smiles",
        url=f"{_ATTENTIVEFP}/malaria-processed.csv",
        target_columns=("activity",),
        task_type="regression",
        # Ships without a header row, columns in (label, smiles) order.
        header_names=("activity", "smiles"),
        description="Malaria - EC50 against the 3D7 P. falciparum strain, log scale "
                    "(single regression task).",
    ),
}

# Order used by the batch benchmark runner.
DEFAULT_BENCHMARK_DATASETS = ["bace", "bbbp", "sider", "tox21", "clintox"]
DEFAULT_DATASET = "tox21"

CLASSIFICATION_DATASETS = [name for name, spec in DATASETS.items() if not spec.is_regression]
REGRESSION_DATASETS = [name for name, spec in DATASETS.items() if spec.is_regression]
TASK_TYPES: tuple[str, ...] = ("classification", "regression")


def available_datasets() -> list[str]:
    return list(DATASETS.keys())


def datasets_for_task(task_type: str) -> list[str]:
    """Dataset names belonging to one task type ('classification' or 'regression')."""
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unknown task_type '{task_type}'. Expected one of {TASK_TYPES}.")
    return [name for name, spec in DATASETS.items() if spec.task_type == task_type]


def group_by_task_type(names: list[str]) -> dict[str, list[str]]:
    """Partition dataset names into {'classification': [...], 'regression': [...]}."""
    grouped: dict[str, list[str]] = {task_type: [] for task_type in TASK_TYPES}
    for name in names:
        grouped[get_dataset_spec(name).task_type].append(name)
    return grouped


def get_dataset_spec(name: str) -> DatasetSpec:
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Available datasets: {', '.join(available_datasets())}."
        )
    return DATASETS[key]


def resolve_target_columns(spec: DatasetSpec, data_path: str | Path) -> list[str]:
    """Return the explicit target columns, or infer them from the CSV header.

    Inference keeps every column that is neither the SMILES column nor listed in
    ``exclude_columns``.
    """
    if spec.target_columns is not None:
        return list(spec.target_columns)
    header = pd.read_csv(data_path, nrows=0)
    skip = {spec.smiles_column, *spec.exclude_columns}
    return [column for column in header.columns if column not in skip]
