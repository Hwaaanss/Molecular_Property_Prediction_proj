"""Prototype interpretability analysis for the Dual-KD-GNN classifier head.

Produces four artifact families per saved single-seed model:

  results/artifacts/prototypes/<dataset>_assignment_heatmap.png
      Side-by-side visualization of the K x M soft-assignment alpha matrix
      from InteractionTensorHead.get_assignment_probabilities():
        (a) Raw alpha with vmin/vmax auto-fit to the actual data range so
            subtle differences are visible (previously used fixed [0,1] scale
            which rendered near-uniform alphas as a single flat color).
        (b) Deviation from the uniform baseline 1/M using a symmetric
            divergent colormap; this reveals whether the codebook developed
            task-specific prototype preferences or remained near-uniform.

  results/artifacts/prototypes/<dataset>_alpha_matrix.csv
      Raw alpha values as a K x M table with class names as row headers and
      prototype indices as column headers. Also reports summary stats:
      min, max, mean, and max deviation from uniform (1/M) per row.

  results/artifacts/prototypes/<dataset>_prototype_norms.csv
      Per-prototype activation statistics over the test split:
      mean L2 norm of the projection z -> C_m^T z across molecules.
      This is INDEPENDENT of alpha and reflects intrinsic codebook capacity.

  results/artifacts/prototypes/<dataset>_top_molecules.csv
      Top-K (default 10) molecules per prototype ranked by activation norm,
      with their SMILES, Murcko scaffold SMILES, MACCS-key fingerprints,
      and ground-truth labels. Substructure analysis downstream consumes this.

Single-seed run weights are read from dual_kd_gnn/runs/<dataset>/. These were
trained from a tuned best_config; their hyperparameters are consistent with the
5-seed ablation full_model runs.

Usage on server:
  conda activate dualgnn
  python scripts/prototype_analysis.py
  python scripts/prototype_analysis.py --datasets tox21    # single dataset
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.data import MoleculeDualDataset, label_aware_scaffold_split  # noqa: E402
from common.datasets import DATASETS, resolve_target_columns  # noqa: E402
from common.plotting import save_figure  # noqa: E402
from dual_kd_gnn.model import DualDistillationModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DEFAULT_DATASETS = ["bace", "bbbp", "clintox", "sider", "tox21"]
RUNS_DIR = PROJECT_ROOT / "dual_kd_gnn" / "runs"
OUT_DIR = PROJECT_ROOT / "results" / "artifacts" / "prototypes"


def load_model_for_dataset(dataset: str, device: torch.device) -> tuple[DualDistillationModel, dict, list[str]]:
    run_dir = RUNS_DIR / dataset
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    target_columns = list(metrics["target_columns"])

    model = DualDistillationModel(
        num_classes=len(target_columns),
        **metadata["model_kwargs"],
    ).to(device)

    state = torch.load(run_dir / "model_weights.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  [warn] state_dict mismatch (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    return model, metadata, target_columns


def get_assignment_matrix(model: DualDistillationModel) -> np.ndarray:
    if not model.classifier.use_codebook:
        return np.zeros((model.classifier.num_classes, 0))
    return model.classifier.get_assignment_probabilities().cpu().numpy()


def save_heatmap(alpha: np.ndarray, target_columns: list[str], out_path: Path, title: str) -> None:
    """Side-by-side heatmap: (a) raw alpha with auto-fit range and (b) deviation from uniform.

    The dual view exists because a naive [0, 1] color scale on a near-uniform matrix
    renders every cell as a single flat color and hides the actual signal magnitude.
    Panel (b) with a symmetric divergent map makes it obvious whether alpha is
    uniform (all near-white) or task-specific (colorful).
    """
    if alpha.size == 0:
        print(f"  [skip] no codebook for {title}")
        return
    import matplotlib.pyplot as plt

    K, M = alpha.shape
    uniform = 1.0 / M
    dev = alpha - uniform
    dev_absmax = max(float(np.abs(dev).max()), 1e-6)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(12, M * 1.4 + 5), max(3.5, K * 0.35 + 1))
    )
    short_labels = [tc[:20] + ("..." if len(tc) > 20 else "") for tc in target_columns]
    proto_labels = [f"P{m}" for m in range(M)]

    # (a) Raw alpha, auto-fit range so subtle differences are visible.
    im1 = ax1.imshow(alpha, aspect="auto", cmap="viridis",
                     vmin=float(alpha.min()), vmax=float(alpha.max()))
    ax1.set_xticks(range(M)); ax1.set_xticklabels(proto_labels, fontsize=9)
    ax1.set_yticks(range(K)); ax1.set_yticklabels(short_labels, fontsize=7)
    ax1.set_xlabel("Prototype index m")
    ax1.set_ylabel("Class index k")
    ax1.set_title(f"(a) raw $\\alpha$ — range [{alpha.min():.3f}, {alpha.max():.3f}]  "
                  f"uniform = {uniform:.3f}", fontsize=10)
    for i in range(K):
        for j in range(M):
            v = alpha[i, j]
            color = "white" if v < (alpha.min() + alpha.max()) / 2 else "black"
            ax1.text(j, i, f"{v:.3f}", ha="center", va="center", color=color, fontsize=7)
    plt.colorbar(im1, ax=ax1, label=r"$\alpha_{k,m}$")

    # (b) Deviation from uniform, symmetric divergent map.
    im2 = ax2.imshow(dev, aspect="auto", cmap="RdBu_r",
                     vmin=-dev_absmax, vmax=dev_absmax)
    ax2.set_xticks(range(M)); ax2.set_xticklabels(proto_labels, fontsize=9)
    ax2.set_yticks(range(K)); ax2.set_yticklabels(short_labels, fontsize=7)
    ax2.set_xlabel("Prototype index m")
    signal_note = "very small: assignment ≈ uniform" if dev_absmax < 0.02 else "task-specific"
    ax2.set_title(f"(b) $\\alpha - 1/M$ — max |dev| = {dev_absmax:.4f}  ({signal_note})",
                  fontsize=10)
    for i in range(K):
        for j in range(M):
            v = dev[i, j]
            color = "white" if abs(v) > dev_absmax * 0.6 else "black"
            ax2.text(j, i, f"{v:+.3f}", ha="center", va="center", color=color, fontsize=7)
    plt.colorbar(im2, ax=ax2, label=r"$\alpha_{k,m} - 1/M$")

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    save_figure(fig, out_path, formats=("png", "tiff", "pdf"), bbox_inches="tight")
    plt.close(fig)


def save_alpha_csv(alpha: np.ndarray, target_columns: list[str], out_path: Path) -> None:
    """Save the raw alpha matrix + per-row deviation statistics."""
    if alpha.size == 0:
        return
    K, M = alpha.shape
    uniform = 1.0 / M
    df = pd.DataFrame(alpha,
                      index=[f"class_{k:02d}_{tc[:30]}" for k, tc in enumerate(target_columns)],
                      columns=[f"P{m}" for m in range(M)])
    df["_row_max"] = alpha.max(axis=1)
    df["_row_min"] = alpha.min(axis=1)
    df["_max_dev_from_uniform"] = np.abs(alpha - uniform).max(axis=1)
    df.to_csv(out_path)


def build_test_split(dataset: str) -> tuple[MoleculeDualDataset, list[str]]:
    spec = DATASETS[dataset]
    data_path = str(spec.data_path())
    target_columns = (
        list(spec.target_columns) if spec.target_columns else resolve_target_columns(spec, data_path)
    )
    dataframe = pd.read_csv(data_path)
    # Analyses the checkpoints trained under the legacy label-aware split, so it
    # must keep using that split to look at the same test molecules.
    _, _, test_idx = label_aware_scaffold_split(
        dataframe,
        target_columns=target_columns,
        smiles_column=spec.smiles_column,
        seed=42,
    )
    ds = MoleculeDualDataset(
        data_path=data_path,
        target_columns=target_columns,
        indices=test_idx,
        smiles_column=spec.smiles_column,
    )
    return ds, target_columns


def compute_prototype_activations(
    model: DualDistillationModel,
    dataset: MoleculeDualDataset,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Return per-molecule prototype activation norms; shape [N, M].

    Activation[n, m] = || C_m^T z_n ||_2 in the classifier's effective space.
    Higher = molecule n drives prototype m more strongly.
    """
    from torch_geometric.loader import DataLoader

    head = model.classifier
    if not head.use_codebook:
        return np.zeros((len(dataset), 0))

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    activations: list[np.ndarray] = []

    codebook = head.codebook_u.detach()  # [M, d, r]

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # The model owns the fusion path (per-block LayerNorm, the optional
            # fingerprint block, pooling), so read it from there rather than
            # re-deriving it here.
            z = model.fused_graph_representation(data)  # [B, concat_dim]
            # Whole-vector, block-diagonal, or no projection, as configured.
            z = head.project(z)

            # y[b, m, r] = sum_d z[b, d] * codebook[m, d, r]
            y = torch.einsum("bd,mdr->bmr", z, codebook)
            norms = y.norm(dim=-1)  # [B, M]
            activations.append(norms.cpu().numpy())

    return np.concatenate(activations, axis=0)


def topk_molecules_per_prototype(
    activations: np.ndarray,
    smiles_list: list[str],
    labels: np.ndarray,
    target_columns: list[str],
    k: int = 10,
) -> pd.DataFrame:
    rows = []
    for m in range(activations.shape[1]):
        order = np.argsort(-activations[:, m])[:k]
        for rank, idx in enumerate(order, start=1):
            smi = smiles_list[idx]
            mol = Chem.MolFromSmiles(smi)
            scaffold = ""
            maccs_bits = ""
            if mol is not None:
                try:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                except Exception:
                    scaffold = ""
                try:
                    bits = MACCSkeys.GenMACCSKeys(mol)
                    on_bits = sorted(bit for bit in range(bits.GetNumBits()) if bits.GetBit(bit))
                    maccs_bits = ";".join(map(str, on_bits))
                except Exception:
                    maccs_bits = ""
            label_str = ";".join(
                f"{tc}={int(v) if not np.isnan(v) and v >= 0 else '?'}"
                for tc, v in zip(target_columns, labels[idx])
            )
            rows.append(
                {
                    "prototype": m,
                    "rank": rank,
                    "activation_norm": float(activations[idx, m]),
                    "molecule_index": int(idx),
                    "smiles": smi,
                    "murcko_scaffold": scaffold,
                    "maccs_on_bits": maccs_bits,
                    "labels": label_str,
                }
            )
    return pd.DataFrame(rows)


def run_dataset(dataset: str, device: torch.device) -> None:
    print(f"\n[{dataset}]")
    try:
        model, _, target_columns = load_model_for_dataset(dataset, device)
    except FileNotFoundError as e:
        print(f"  [skip] missing artifact: {e}")
        return

    alpha = get_assignment_matrix(model)
    print(f"  alpha shape: {alpha.shape}")
    if alpha.size > 0:
        uniform = 1.0 / alpha.shape[1]
        dev = float(np.abs(alpha - uniform).max())
        print(f"  alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]  "
              f"uniform = {uniform:.4f}  max |dev from uniform| = {dev:.4f}")
        if dev < 0.02:
            print(f"  [NOTE] alpha is effectively uniform on {dataset}; "
                  "codebook did not develop task-specific prototype preferences.")

    heatmap_path = OUT_DIR / f"{dataset}_assignment_heatmap.png"
    save_heatmap(alpha, target_columns, heatmap_path,
                 title=f"{dataset} codebook prototype assignment ({alpha.shape[0]} tasks × {alpha.shape[1]} prototypes)")
    print(f"  saved heatmap: {heatmap_path.relative_to(PROJECT_ROOT)}")

    if alpha.size == 0:
        return  # no codebook; nothing more to do

    alpha_csv_path = OUT_DIR / f"{dataset}_alpha_matrix.csv"
    save_alpha_csv(alpha, target_columns, alpha_csv_path)
    print(f"  saved alpha CSV: {alpha_csv_path.relative_to(PROJECT_ROOT)}")

    test_ds, _ = build_test_split(dataset)
    activations = compute_prototype_activations(model, test_ds, device)
    print(f"  activations shape: {activations.shape}")

    norm_stats = pd.DataFrame(
        {
            "prototype": np.arange(activations.shape[1]),
            "mean_norm": activations.mean(axis=0),
            "std_norm": activations.std(axis=0, ddof=1) if activations.shape[0] > 1 else 0.0,
            "max_norm": activations.max(axis=0),
        }
    )
    norm_stats.to_csv(OUT_DIR / f"{dataset}_prototype_norms.csv", index=False)
    print(f"  saved norms: {dataset}_prototype_norms.csv")

    smiles_list = test_ds.smiles
    labels = test_ds.labels.cpu().numpy()
    topk = topk_molecules_per_prototype(activations, smiles_list, labels, target_columns, k=10)
    topk.to_csv(OUT_DIR / f"{dataset}_top_molecules.csv", index=False)
    print(f"  saved top molecules: {dataset}_top_molecules.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=DEFAULT_DATASETS)
    parser.add_argument("--device", default=None, help="torch device override (cpu / cuda / mps)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Output dir: {OUT_DIR.relative_to(PROJECT_ROOT)}")

    for dataset in args.datasets:
        run_dataset(dataset, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
