from __future__ import annotations

import os
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import (
    AllChem,
    ChemicalFeatures,
    Descriptors,
    MACCSkeys,
    rdFingerprintGenerator,
    rdReducedGraphs,
)
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data, Dataset


RDLogger.DisableLog("rdApp.warning")


class MolData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "subgraph_edge_index":
            return int(self.subgraph_x.size(0))
        if key == "assign_index":
            return value.new_tensor([[self.x.size(0)], [self.subgraph_x.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


ATOM_SYMBOLS_100 = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
]
HYBRIDIZATIONS_127 = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]
BOND_STEREOS_12 = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]
MLFGNN_ATOMS_16 = ["C", "N", "O", "F", "Si", "Cl", "As", "Se", "Br", "Te", "I", "At"]
MLFGNN_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES_4 = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "P": 1.80,
}

FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
)
ACIDIC_SMARTS = Chem.MolFromSmarts("[$([C,S,P](=O)[O;H,-1])]")
BASIC_SMARTS = Chem.MolFromSmarts(
    "[#7;+,$([N;H2&+0][C,c]),$([N;H1&+0]([C,c])[C,c]),$([N;H0&+0]([C,c])([C,c])[C,c])]"
)
DESCRIPTOR_FUNCS = [(name, fn) for name, fn in Descriptors._descList[:200]]
MORGAN_GENERATOR_1024 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def one_hot_with_unknown(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices] + [0.0 if value in choices else 1.0]


def one_hot_no_unknown(value, choices):
    return [1.0 if value == choice else 0.0 for choice in choices]


def bitvect_to_array(fp, n_bits: int) -> np.ndarray:
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def get_auxiliary_atom_sets(mol):
    donor_atoms, acceptor_atoms = set(), set()
    for feature in FEATURE_FACTORY.GetFeaturesForMol(mol):
        if feature.GetFamily() == "Donor":
            donor_atoms.update(feature.GetAtomIds())
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(feature.GetAtomIds())

    acidic_atoms = set(
        atom_idx
        for match in mol.GetSubstructMatches(ACIDIC_SMARTS)
        for atom_idx in match
    ) if ACIDIC_SMARTS is not None else set()
    basic_atoms = set(
        atom_idx
        for match in mol.GetSubstructMatches(BASIC_SMARTS)
        for atom_idx in match
    ) if BASIC_SMARTS is not None else set()
    return donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms


def get_atom_features_127(atom) -> np.ndarray:
    features = []
    features.extend(one_hot_no_unknown(atom.GetSymbol(), ATOM_SYMBOLS_100))
    features.extend(one_hot_no_unknown(min(atom.GetDegree(), 5), list(range(6))))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.extend(one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATIONS_127))
    features.extend(one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS))
    features.extend(one_hot_no_unknown(min(atom.GetTotalNumHs(), 4), list(range(5))))
    features.append(float(atom.IsInRing()))
    features.append(float(atom.GetIsAromatic()))
    return np.asarray(features, dtype=np.float32)


def get_bond_features_12(bond) -> np.ndarray:
    features = []
    features.extend(one_hot_no_unknown(bond.GetBondType(), BOND_TYPES_4))
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    features.extend(one_hot_no_unknown(bond.GetStereo(), BOND_STEREOS_12))
    return np.asarray(features, dtype=np.float32)


def get_atom_features_54(atom, donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms) -> np.ndarray:
    features = []
    features.extend(one_hot_with_unknown(atom.GetSymbol(), MLFGNN_ATOMS_16))
    features.extend(one_hot_no_unknown(min(atom.GetDegree(), 5), list(range(6))))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetNumRadicalElectrons()))
    features.extend(one_hot_with_unknown(atom.GetHybridization(), MLFGNN_HYBRIDIZATIONS))
    features.append(float(atom.GetIsAromatic()))
    features.extend(one_hot_no_unknown(min(atom.GetTotalNumHs(), 4), list(range(5))))
    features.extend(one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAGS))
    features.append(float(atom.IsInRing()))
    features.extend([float(atom.IsInRingSize(size)) for size in [3, 4, 5, 6]])
    features.append(float(atom.GetMass() / 200.0))
    implicit_valence = min(max(int(atom.GetValence(Chem.ValenceType.IMPLICIT)), 0), 6)
    features.extend(one_hot_no_unknown(implicit_valence, list(range(7))))
    atom_idx = atom.GetIdx()
    features.append(float(atom_idx in acceptor_atoms))
    features.append(float(atom_idx in donor_atoms))
    features.append(float(atom_idx in acidic_atoms))
    features.append(float(atom_idx in basic_atoms))
    return np.asarray(features, dtype=np.float32)


def get_bond_features_13(bond) -> np.ndarray:
    features = [1.0]
    features.extend(one_hot_no_unknown(bond.GetBondType(), BOND_TYPES_4))
    features.append(float(bond.GetIsConjugated()))
    features.append(float(bond.IsInRing()))
    stereo_map = list(range(6))
    stereo_value = int(bond.GetStereo())
    stereo_value = stereo_value if stereo_value in stereo_map else 0
    features.extend(one_hot_no_unknown(stereo_value, stereo_map))
    return np.asarray(features, dtype=np.float32)


def get_rdkit_descriptor_200(mol) -> np.ndarray:
    values: list[float] = []
    for _, descriptor_fn in DESCRIPTOR_FUNCS:
        try:
            value = float(descriptor_fn(mol))
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            value = float(np.clip(value, -1e6, 1e6))
        except Exception:
            value = 0.0
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def fit_rdkit_descriptor_scaler(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    descriptors = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            descriptors.append(np.zeros((200,), dtype=np.float32))
        else:
            descriptors.append(get_rdkit_descriptor_200(mol))

    descriptor_array = np.asarray(descriptors, dtype=np.float32)
    mean = descriptor_array.mean(axis=0)
    scale = descriptor_array.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def get_pubchem_like_fp(mol, n_bits: int = 881) -> np.ndarray:
    return bitvect_to_array(Chem.PatternFingerprint(mol, fpSize=n_bits), n_bits)


def get_erg_fp_441(mol) -> np.ndarray:
    fp = np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32)
    if fp.shape[0] >= 441:
        return fp[:441]
    return np.pad(fp, (0, 441 - fp.shape[0]), mode="constant").astype(np.float32)


def get_morgan_fp_1024(mol) -> np.ndarray:
    """Morgan ECFP4 (radius 2), 1024 bits."""
    return bitvect_to_array(MORGAN_GENERATOR_1024.GetFingerprint(mol), 1024)


# ---------------------------------------------------------------------------
# Molecule-level fingerprints for the fingerprint fusion branch.
#
# IMPORTANT: every fingerprint here is computed from the 2D molecular graph
# alone. None of them reads a conformer, so a molecule whose MMFF94 embedding
# failed -- and whose physical (geometric) branch therefore degenerates to zero
# coordinates -- still receives a full, informative fingerprint. Fingerprint
# availability and conformer availability are independent by construction; only
# an unparseable SMILES (mol is None) yields a zero fingerprint.
#
# Dimensions are exported as constants so the model never hardcodes them.
# ---------------------------------------------------------------------------
FINGERPRINT_BUILDERS = {
    "morgan": (1024, get_morgan_fp_1024),
    "pubchem": (881, get_pubchem_like_fp),
    "erg": (441, get_erg_fp_441),
}
DEFAULT_FP_SET: tuple[str, ...] = ("morgan", "pubchem", "erg")


def fingerprint_dim(fp_set: tuple[str, ...] = DEFAULT_FP_SET) -> int:
    """Total width of the concatenated fingerprint for ``fp_set``."""
    unknown = [name for name in fp_set if name not in FINGERPRINT_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown fingerprint(s): {', '.join(unknown)}. "
            f"Known: {', '.join(FINGERPRINT_BUILDERS)}."
        )
    return sum(FINGERPRINT_BUILDERS[name][0] for name in fp_set)


DEFAULT_FINGERPRINT_DIM = fingerprint_dim()


def molecule_fingerprint_with_status(
    mol,
    fp_set: tuple[str, ...] = DEFAULT_FP_SET,
) -> tuple[np.ndarray, bool]:
    """``(fingerprint, ok)`` for one molecule.

    ``ok`` is False when ``mol`` is None or any generator raised, in which case
    the fingerprint is all zeros. Callers that featurize a whole dataset use the
    flag to report a failure count rather than silently training on zeros.
    """
    total = fingerprint_dim(fp_set)
    if mol is None:
        return np.zeros((total,), dtype=np.float32), False
    blocks: list[np.ndarray] = []
    ok = True
    for name in fp_set:
        width, builder = FINGERPRINT_BUILDERS[name]
        try:
            block = np.asarray(builder(mol), dtype=np.float32).reshape(-1)
            if block.shape[0] != width:
                raise ValueError(f"{name} returned {block.shape[0]} bits, expected {width}")
        except Exception:
            block = np.zeros((width,), dtype=np.float32)
            ok = False
        blocks.append(block)
    return np.concatenate(blocks, axis=0).astype(np.float32), ok


def get_molecule_fingerprint(mol, fp_set: tuple[str, ...] = DEFAULT_FP_SET) -> np.ndarray:
    """Concatenated molecule-level fingerprint; all zeros if it cannot be built."""
    return molecule_fingerprint_with_status(mol, fp_set)[0]


def build_subgraph_data(mol):
    cliques: list[list[int]] = []
    for ring in mol.GetRingInfo().AtomRings():
        cliques.append(sorted(set(ring)))
    for bond in mol.GetBonds():
        if not bond.IsInRing():
            cliques.append(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))

    covered = {atom_idx for clique in cliques for atom_idx in clique}
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in covered:
            cliques.append([atom.GetIdx()])

    if not cliques:
        cliques = [[0]]

    sub_features = []
    assign_edges = []
    for subgraph_idx, clique in enumerate(cliques):
        is_ring = 1.0 if len(clique) > 2 else 0.0
        is_bond = 1.0 if len(clique) == 2 else 0.0
        is_singleton = 1.0 if len(clique) == 1 else 0.0
        sub_features.append([is_ring, is_bond, is_singleton, len(clique) / 8.0])
        for atom_idx in clique:
            assign_edges.append([atom_idx, subgraph_idx])

    sub_edges: list[list[int]] = []
    for left_idx in range(len(cliques)):
        clique_left = set(cliques[left_idx])
        for right_idx in range(left_idx + 1, len(cliques)):
            if clique_left.intersection(cliques[right_idx]):
                sub_edges.extend([[left_idx, right_idx], [right_idx, left_idx]])

    if sub_edges:
        sub_edge_index = torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
    else:
        sub_edge_index = torch.empty((2, 0), dtype=torch.long)

    assign_index = torch.tensor(assign_edges, dtype=torch.long).t().contiguous()
    sub_x = torch.tensor(np.asarray(sub_features, dtype=np.float32), dtype=torch.float)
    sub_batch = torch.zeros(sub_x.size(0), dtype=torch.long)
    return sub_x, sub_edge_index, assign_index, sub_batch


def smiles_to_graph_standard(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms = get_auxiliary_atom_sets(mol)
    x = torch.tensor(
        np.asarray([get_atom_features_127(atom) for atom in mol.GetAtoms()]),
        dtype=torch.float,
    )
    x_mlfgnn = torch.tensor(
        np.asarray(
            [
                get_atom_features_54(atom, donor_atoms, acceptor_atoms, acidic_atoms, basic_atoms)
                for atom in mol.GetAtoms()
            ]
        ),
        dtype=torch.float,
    )

    edge_pairs, edge_attr, edge_attr_mlfgnn = [], [], []
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feature_12 = get_bond_features_12(bond)
        bond_feature_13 = get_bond_features_13(bond)
        edge_pairs.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_attr.extend([bond_feature_12, bond_feature_12])
        edge_attr_mlfgnn.extend([bond_feature_13, bond_feature_13])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr_tensor = torch.tensor(np.asarray(edge_attr, dtype=np.float32), dtype=torch.float)
        edge_attr_mlfgnn_tensor = torch.tensor(
            np.asarray(edge_attr_mlfgnn, dtype=np.float32),
            dtype=torch.float,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr_tensor = torch.empty((0, 12), dtype=torch.float)
        edge_attr_mlfgnn_tensor = torch.empty((0, 13), dtype=torch.float)

    pubchem_like = get_pubchem_like_fp(mol)
    fp_fpgnn = np.concatenate(
        [
            pubchem_like,
            bitvect_to_array(MACCSkeys.GenMACCSKeys(mol), 167),
            get_erg_fp_441(mol),
        ],
        axis=0,
    ).astype(np.float32)
    fp_mlfgnn = np.concatenate(
        [
            bitvect_to_array(MORGAN_GENERATOR_1024.GetFingerprint(mol), 1024),
            pubchem_like,
            get_erg_fp_441(mol),
        ],
        axis=0,
    ).astype(np.float32)
    sub_x, subgraph_edge_index, assign_index, subgraph_batch = build_subgraph_data(mol)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr_tensor,
        "x_mlfgnn": x_mlfgnn,
        "edge_attr_mlfgnn": edge_attr_mlfgnn_tensor,
        "fp_fpgnn": torch.tensor(fp_fpgnn, dtype=torch.float).unsqueeze(0),
        "fp_mlfgnn": torch.tensor(fp_mlfgnn, dtype=torch.float).unsqueeze(0),
        "rdkit_desc": torch.tensor(get_rdkit_descriptor_200(mol), dtype=torch.float).unsqueeze(0),
        "subgraph_x": sub_x,
        "subgraph_edge_index": subgraph_edge_index,
        "assign_index": assign_index,
        "subgraph_batch": subgraph_batch,
    }


def get_chemical_features(atom) -> np.ndarray:
    features = [atom.GetAtomicNum() / 100.0, atom.GetFormalCharge(), 1.0 if atom.GetIsAromatic() else 0.0]
    hybridizations = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    atom_hybridization = atom.GetHybridization()
    features.extend([1.0 if atom_hybridization == hybridization else 0.0 for hybridization in hybridizations])
    features.append(1.0 if atom_hybridization not in hybridizations else 0.0)
    atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]
    atom_symbol = atom.GetSymbol()
    features.extend([1.0 if atom_symbol == atom_type else 0.0 for atom_type in atom_types])
    features.append(1.0 if atom_symbol not in atom_types else 0.0)
    features.append(float(atom.GetDegree()))
    return np.asarray(features, dtype=np.float32)


def get_physical_features(atom, conformer, center_of_mass) -> np.ndarray:
    features = []
    if conformer is not None and center_of_mass is not None:
        position = conformer.GetAtomPosition(atom.GetIdx())
        features.extend(
            [
                position.x - center_of_mass[0],
                position.y - center_of_mass[1],
                position.z - center_of_mass[2],
            ]
        )
    else:
        features.extend([0.0, 0.0, 0.0])
    features.append(atom.GetMass() / 100.0)
    features.append(VDW_RADII.get(atom.GetSymbol(), 1.70))
    return np.asarray(features, dtype=np.float32)


def get_bond_features_dual(bond, conformer) -> np.ndarray:
    features = []
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    bond_type = bond.GetBondType()
    features.extend([1.0 if bond_type == candidate else 0.0 for candidate in bond_types])
    features.append(1.0 if bond.GetIsConjugated() else 0.0)
    features.append(1.0 if bond.IsInRing() else 0.0)
    if conformer is not None:
        try:
            begin_pos = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
            end_pos = conformer.GetAtomPosition(bond.GetEndAtomIdx())
            features.append(float(begin_pos.Distance(end_pos)))
        except Exception:
            features.append(1.5)
    else:
        features.append(1.5)
    return np.asarray(features, dtype=np.float32)


def smiles_to_graph_dual(smiles: str):
    """``(x_chem, x_phys, edge_index, edge_attr, fp)`` for one molecule, or None.

    ``fp`` is the molecule-level fingerprint (see get_molecule_fingerprint). It
    is derived from the 2D graph only, so it is filled in normally even when the
    conformer embedding below fails and ``x_phys`` falls back to zero positions.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Computed before AddHs/embedding so the fingerprint never depends on the
    # 3D pipeline succeeding.
    fp = torch.tensor(get_molecule_fingerprint(mol), dtype=torch.float)

    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        conformer = mol.GetConformer()
    except Exception:
        conformer = None

    mol = Chem.RemoveHs(mol)
    center_of_mass = None
    if conformer is not None:
        positions, masses = [], []
        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            masses.append(atom.GetMass())
        center_of_mass = np.average(np.asarray(positions), axis=0, weights=np.asarray(masses))

    x_chem = torch.tensor(
        np.asarray([get_chemical_features(atom) for atom in mol.GetAtoms()]),
        dtype=torch.float,
    )
    x_phys = torch.tensor(
        np.asarray([get_physical_features(atom, conformer, center_of_mass) for atom in mol.GetAtoms()]),
        dtype=torch.float,
    )

    edge_index_rows, edge_features = [], []
    for bond in mol.GetBonds():
        begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_feature = get_bond_features_dual(bond, conformer)
        edge_index_rows.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_features.extend([bond_feature, bond_feature])

    if not edge_index_rows:
        return (
            x_chem,
            x_phys,
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 7), dtype=torch.float),
            fp,
        )

    return (
        x_chem,
        x_phys,
        torch.tensor(edge_index_rows, dtype=torch.long).t().contiguous(),
        torch.tensor(np.asarray(edge_features), dtype=torch.float),
        fp,
    )


def generate_scaffold(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return None


def _scaffold_stratum(label_block: np.ndarray) -> int:
    """Coarse label stratum for a scaffold group (kept intact during splitting).

    Returns +1 for positive-leaning groups, -1 for negative-leaning groups, and
    0 when no labels are observed. Missing labels (NaN / -1) are ignored.
    """
    positives = float(np.sum(label_block == 1.0))
    negatives = float(np.sum(label_block == 0.0))
    if positives == 0.0 and negatives == 0.0:
        return 0
    return 1 if positives >= negatives else -1


SPLIT_TYPES = (
    "deterministic_scaffold",
    "random_scaffold",
    "random",
    "label_aware_scaffold",
)
SplitType = Literal["deterministic_scaffold", "random_scaffold", "random", "label_aware_scaffold"]

# Name used in generated artifacts -- result filenames and the ``split_protocol``
# column. "deterministic_scaffold" is an implementation detail (it distinguishes
# the sorted fill from random_scaffold's shuffled one); to a reader of the
# results it is simply *the* scaffold split, which is what published baselines
# call it. The internal key stays unchanged so run directories, existing
# metrics.json files and in-flight sweeps keep resolving.
SPLIT_LABELS = {
    "deterministic_scaffold": "scaffold",
    "random_scaffold": "random_scaffold",
    "random": "random",
    "label_aware_scaffold": "label_aware_scaffold",
}
# Accepted on the command line: the published label, the internal key, or (for
# the scaffold split) either spelling.
SPLIT_TYPE_ALIASES = {label: key for key, label in SPLIT_LABELS.items()}
SPLIT_TYPE_ALIASES.update({key: key for key in SPLIT_TYPES})
SPLIT_TYPE_CHOICES = sorted(SPLIT_TYPE_ALIASES)


def resolve_split_type(name: str) -> str:
    """Map a user-facing split name onto the internal key."""
    try:
        return SPLIT_TYPE_ALIASES[name]
    except KeyError:
        raise ValueError(
            f"Unknown split type '{name}'. Expected one of {SPLIT_TYPE_CHOICES}."
        ) from None


def split_label(name: str) -> str:
    """Name to print, and to use in generated filenames, for a split type."""
    return SPLIT_LABELS.get(resolve_split_type(name), name)


def scaffold_groups(
    dataframe: pd.DataFrame,
    smiles_column: str = "smiles",
    empty_scaffold_singleton: bool = False,
) -> dict[str, list[int]]:
    """Group row indices by Bemis-Murcko scaffold (``includeChirality=False``).

    Molecules RDKit cannot parse get a singleton group each, since they have no
    scaffold to share. Acyclic molecules yield an empty scaffold string; by
    default they form one shared group, matching DeepChem's ScaffoldSplitter.
    ``empty_scaffold_singleton=True`` reproduces the legacy grouping used by
    :func:`label_aware_scaffold_split`, where they were singletons instead.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(dataframe[smiles_column]):
        scaffold = generate_scaffold(smiles)
        if scaffold is None or (empty_scaffold_singleton and not scaffold):
            groups[f"__unparsed_{idx}__"].append(idx)
        else:
            groups[scaffold].append(idx)
    return dict(groups)


def _fill_splits(group_list: list[list[int]], n_total: int, train_size: float, val_size: float):
    """Greedily pour pre-ordered scaffold groups into train, then val, then test.

    A group is added to train only if it does not push train past its quota,
    which is what keeps whole scaffold groups inside a single split.
    """
    train_cutoff = train_size * n_total
    val_cutoff = (train_size + val_size) * n_total
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for group in group_list:
        if len(train_indices) + len(group) > train_cutoff:
            if len(train_indices) + len(val_indices) + len(group) > val_cutoff:
                test_indices.extend(group)
            else:
                val_indices.extend(group)
        else:
            train_indices.extend(group)
    return train_indices, val_indices, test_indices


def deterministic_scaffold_split(
    dataframe: pd.DataFrame,
    smiles_column: str = "smiles",
    train_size: float = 0.8,
    val_size: float = 0.1,
):
    """Standard deterministic scaffold split (Hu et al. 2020 / DeepChem).

    Scaffold groups are sorted by (size, first index) descending, so the largest
    and most common scaffolds land in train and the rarest ones in test. The
    ordering is fully determined by the data, hence no ``seed`` argument: the
    absence of randomness *is* the protocol, and this is the split public
    MoleculeNet baselines report against.
    """
    groups = scaffold_groups(dataframe, smiles_column)
    ordered = [
        group
        for _, group in sorted(groups.items(), key=lambda item: (len(item[1]), item[1][0]), reverse=True)
    ]
    return _fill_splits(ordered, len(dataframe), train_size, val_size)


def random_scaffold_split(
    dataframe: pd.DataFrame,
    smiles_column: str = "smiles",
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
):
    """Scaffold split with a randomly permuted group order (Uni-Mol family).

    Identical grouping to :func:`deterministic_scaffold_split`; only the order in
    which groups are poured into the splits differs, which makes the protocol
    seed-dependent and therefore reportable as mean +/- std over seeds.

    Fills val and test to their quotas first and gives train the remainder, as in
    the reference implementation (Hu et al.'s pretrain-gnns splitter). Reusing the
    train-first fill of the deterministic variant would not work here: the sorted
    order guarantees the largest scaffold groups reach train while it is still
    empty, but a shuffled order does not. Tox21's 1775-molecule empty-scaffold
    group, for instance, then lands in val and leaves test empty.
    """
    groups = scaffold_groups(dataframe, smiles_column)
    # Sort keys first so the permutation depends only on the seed, not on dict
    # insertion order, then permute positions (groups are ragged, so permuting
    # an object array of lists would be fragile).
    ordered_keys = sorted(groups.keys())
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(ordered_keys))

    n_total = len(dataframe)
    test_size = max(0.0, 1.0 - train_size - val_size)
    n_val = int(np.floor(val_size * n_total))
    n_test = int(np.floor(test_size * n_total))
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for position in permutation:
        group = groups[ordered_keys[position]]
        if len(val_indices) + len(group) <= n_val:
            val_indices.extend(group)
        elif len(test_indices) + len(group) <= n_test:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
    return train_indices, val_indices, test_indices


def random_split(n: int, train_size: float = 0.8, val_size: float = 0.1, seed: int = 42):
    """Uniform random split over row indices (no scaffold structure preserved)."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)
    n_train = int(n * train_size)
    n_val = int(n * val_size)
    return (
        indices[:n_train].tolist(),
        indices[n_train:n_train + n_val].tolist(),
        indices[n_train + n_val:].tolist(),
    )


def label_aware_scaffold_split(
    dataframe: pd.DataFrame,
    target_columns: list[str] | None = None,
    smiles_column: str = "smiles",
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
):
    """Label-aware scaffold split.

    Non-standard: kept only to reproduce results produced before the switch to
    :func:`deterministic_scaffold_split`. Numbers from this split are not
    comparable with published MoleculeNet baselines. The ``seed`` argument has
    no effect -- no random number is drawn after the call to ``np.random.seed``.

    Molecules sharing a Bemis-Murcko scaffold stay in the same split, but each
    scaffold group is first assigned to a label stratum and every stratum is
    distributed across train/val/test in the requested proportions. This keeps
    the scaffold-generalization property while guaranteeing each split sees both
    classes. It was introduced to avoid single-class val/test splits with an
    undefined ROC-AUC; scripts/verify_splits.py shows the standard split does not
    actually produce any on these five datasets, so that motivation does not hold.
    """
    np.random.seed(seed)
    scaffolds = scaffold_groups(dataframe, smiles_column, empty_scaffold_singleton=True)

    if target_columns:
        labels = dataframe[target_columns].to_numpy(dtype=float)
    else:
        labels = None

    strata: dict[int, list[list[int]]] = defaultdict(list)
    for scaffold_set in scaffolds.values():
        key = _scaffold_stratum(labels[scaffold_set]) if labels is not None else 0
        strata[key].append(scaffold_set)

    test_size = max(0.0, 1.0 - train_size - val_size)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    buckets = {"train": train_indices, "val": val_indices, "test": test_indices}

    # Distribute each stratum independently so the minority class is spread
    # across all three splits, not absorbed entirely into the largest split.
    for key in sorted(strata.keys()):
        group_list = strata[key]
        stratum_total = sum(len(group) for group in group_list)
        targets = {
            "train": train_size * stratum_total,
            "val": val_size * stratum_total,
            "test": test_size * stratum_total,
        }
        counts = {"train": 0, "val": 0, "test": 0}
        for group in sorted(group_list, key=len, reverse=True):
            split = max(("train", "val", "test"), key=lambda s: targets[s] - counts[s])
            buckets[split].extend(group)
            counts[split] += len(group)

    if not val_indices and len(test_indices) > 1:
        midpoint = len(test_indices) // 2
        val_indices = test_indices[:midpoint]
        test_indices = test_indices[midpoint:]

    return train_indices, val_indices, test_indices


def split_indices(
    dataframe: pd.DataFrame,
    split_type: SplitType = "deterministic_scaffold",
    target_columns: list[str] | None = None,
    smiles_column: str = "smiles",
    train_size: float = 0.8,
    val_size: float = 0.1,
    seed: int = 42,
    task_type: str = "classification",
):
    """Dispatch to one of the four split protocols. Returns (train, val, test).

    ``seed`` is accepted for every protocol but ignored by the two deterministic
    ones, so callers can pass it uniformly.
    """
    if split_type == "deterministic_scaffold":
        return deterministic_scaffold_split(dataframe, smiles_column, train_size, val_size)
    if split_type == "random_scaffold":
        return random_scaffold_split(dataframe, smiles_column, train_size, val_size, seed)
    if split_type == "random":
        return random_split(len(dataframe), train_size, val_size, seed)
    if split_type == "label_aware_scaffold":
        if task_type == "regression":
            raise ValueError(
                "label_aware_scaffold is a classification-only protocol: it buckets scaffold "
                "groups by whether positives outnumber negatives, which is undefined for "
                "continuous targets. Use deterministic_scaffold, random_scaffold or random."
            )
        return label_aware_scaffold_split(
            dataframe,
            target_columns=target_columns,
            smiles_column=smiles_column,
            train_size=train_size,
            val_size=val_size,
            seed=seed,
        )
    raise ValueError(f"Unknown split_type '{split_type}'. Expected one of {SPLIT_TYPES}.")


def count_positives(dataframe: pd.DataFrame, target_columns: list[str], indices: list[int]) -> list[int]:
    """Per-task count of label==1 rows within ``indices`` (missing labels ignored)."""
    if not indices or not target_columns:
        return [0] * len(target_columns)
    labels = dataframe.iloc[indices][target_columns].to_numpy(dtype=float)
    return [int(np.sum(labels[:, task] == 1.0)) for task in range(labels.shape[1])]


def describe_splits(
    dataframe: pd.DataFrame,
    target_columns: list[str],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    task_type: str = "classification",
    max_tasks_shown: int = 20,
) -> None:
    """Print per-split label diagnostics: positives per task, or target stats."""
    for name, indices in (("train", train_indices), ("val", val_indices), ("test", test_indices)):
        if task_type == "regression":
            labels = dataframe.iloc[indices][target_columns].to_numpy(dtype=float)
            with np.errstate(invalid="ignore"):
                means = np.nanmean(labels, axis=0) if labels.size else np.array([])
                stds = np.nanstd(labels, axis=0) if labels.size else np.array([])
            print(f"  targets ({name}, n={len(indices)}): "
                  f"mean={np.round(means[:max_tasks_shown], 3).tolist()} "
                  f"std={np.round(stds[:max_tasks_shown], 3).tolist()}")
        else:
            positives = count_positives(dataframe, target_columns, indices)
            shown = positives[:max_tasks_shown]
            suffix = f" ... (+{len(positives) - max_tasks_shown} more tasks)" if len(positives) > max_tasks_shown else ""
            print(f"  positives per task ({name}, n={len(indices)}): {shown}{suffix}")


def count_auc_defined_tasks(
    dataframe: pd.DataFrame,
    target_columns: list[str],
    indices: list[int],
) -> int:
    """Number of tasks with both classes present, i.e. with a defined ROC-AUC.

    Mirrors the skip condition in common/metrics.py: missing labels (NaN, later
    encoded as -1) are dropped, then a task counts only if what remains contains
    at least two distinct label values.
    """
    if not indices or not target_columns:
        return 0
    labels = dataframe.iloc[indices][target_columns].to_numpy(dtype=float)
    defined = 0
    for task in range(labels.shape[1]):
        values = labels[:, task]
        values = values[~np.isnan(values) & (values != -1.0)]
        if values.size and np.unique(values).size >= 2:
            defined += 1
    return defined


class MoleculeDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        target_columns: list[str],
        indices=None,
        rdkit_desc_mean: np.ndarray | None = None,
        rdkit_desc_scale: np.ndarray | None = None,
        smiles_column: str = "smiles",
    ):
        super().__init__()
        dataframe = pd.read_csv(data_path)
        if indices is not None:
            dataframe = dataframe.iloc[indices].reset_index(drop=True)
        self.smiles = dataframe[smiles_column].tolist()
        labels = torch.tensor(dataframe[target_columns].values, dtype=torch.float)
        self.labels = torch.where(torch.isnan(labels), torch.tensor(-1.0), labels)
        self.rdkit_desc_mean = rdkit_desc_mean
        self.rdkit_desc_scale = rdkit_desc_scale
        # Featurize once up front; get() is called every epoch by the DataLoader,
        # so recomputing graphs/fingerprints/descriptors per access wastes CPU.
        self._graph_cache = [smiles_to_graph_standard(smiles) for smiles in self.smiles]

    def len(self):
        return len(self.smiles)

    def get(self, idx: int):
        graph_data = self._graph_cache[idx]
        rdkit_desc = torch.zeros((1, 200), dtype=torch.float)
        if graph_data is None:
            pass
        else:
            rdkit_desc = graph_data["rdkit_desc"].clone()
            if self.rdkit_desc_mean is not None and self.rdkit_desc_scale is not None:
                desc_array = rdkit_desc.squeeze(0).numpy()
                desc_array = (desc_array - self.rdkit_desc_mean) / self.rdkit_desc_scale
                rdkit_desc = torch.from_numpy(desc_array.astype(np.float32)).unsqueeze(0)

        if graph_data is None:
            return MolData(
                x=torch.zeros((1, 127), dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 12), dtype=torch.float),
                x_mlfgnn=torch.zeros((1, 54), dtype=torch.float),
                edge_attr_mlfgnn=torch.empty((0, 13), dtype=torch.float),
                fp_fpgnn=torch.zeros((1, 1489), dtype=torch.float),
                fp_mlfgnn=torch.zeros((1, 2346), dtype=torch.float),
                rdkit_desc=rdkit_desc,
                subgraph_x=torch.tensor([[0.0, 0.0, 1.0, 0.125]], dtype=torch.float),
                subgraph_edge_index=torch.empty((2, 0), dtype=torch.long),
                assign_index=torch.tensor([[0], [0]], dtype=torch.long),
                subgraph_batch=torch.zeros(1, dtype=torch.long),
                y=self.labels[idx].clone(),
            )

        return MolData(
            x=graph_data["x"].clone(),
            edge_index=graph_data["edge_index"].clone(),
            edge_attr=graph_data["edge_attr"].clone(),
            x_mlfgnn=graph_data["x_mlfgnn"].clone(),
            edge_attr_mlfgnn=graph_data["edge_attr_mlfgnn"].clone(),
            fp_fpgnn=graph_data["fp_fpgnn"].clone(),
            fp_mlfgnn=graph_data["fp_mlfgnn"].clone(),
            rdkit_desc=rdkit_desc,
            subgraph_x=graph_data["subgraph_x"].clone(),
            subgraph_edge_index=graph_data["subgraph_edge_index"].clone(),
            assign_index=graph_data["assign_index"].clone(),
            subgraph_batch=graph_data["subgraph_batch"].clone(),
            y=self.labels[idx].clone(),
        )


class MoleculeDualDataset(Dataset):
    """Dual-branch (chemical + physical) graphs for one split of a CSV.

    Pass ``feature_cache`` (see common/featurize_cache) to reuse one featurization
    across the train/val/test splits and across runs; without it the molecules in
    ``indices`` are featurized inline, which is fine for small datasets but costs
    hours on HIV/CEP.

    Label encoding depends on ``task_type``. Classification keeps the historical
    -1 sentinel for missing labels. Regression cannot: -1 is a legitimate target
    value (most of ESOL is negative), so missing entries are zero-filled and
    flagged in ``y_mask`` instead.
    """

    def __init__(
        self,
        data_path: str,
        target_columns: list[str],
        indices=None,
        smiles_column: str = "smiles",
        task_type: str = "classification",
        feature_cache=None,
    ):
        super().__init__()
        dataframe = pd.read_csv(data_path)
        row_indices = list(range(len(dataframe))) if indices is None else list(indices)
        dataframe = dataframe.iloc[row_indices].reset_index(drop=True)
        self.smiles = dataframe[smiles_column].tolist()
        self.task_type = task_type
        self.is_regression = task_type == "regression"

        labels = torch.tensor(dataframe[target_columns].to_numpy(dtype=float), dtype=torch.float)
        missing = torch.isnan(labels)
        if self.is_regression:
            self.labels = torch.where(missing, torch.zeros_like(labels), labels)
            self.label_mask = (~missing).float()
        else:
            self.labels = torch.where(missing, torch.full_like(labels, -1.0), labels)
            self.label_mask = None

        self._feature_cache = feature_cache
        self._cache_rows = row_indices if feature_cache is not None else None
        if feature_cache is None:
            # Featurize once up front; the 3D conformer embedding + MMFF optimization
            # in smiles_to_graph_dual is expensive and must not run every epoch.
            self._graph_cache = [smiles_to_graph_dual(smiles) for smiles in self.smiles]
        else:
            self._graph_cache = None

    def len(self):
        return len(self.smiles)

    def _graph_tensors(self, idx: int):
        if self._feature_cache is not None:
            arrays = self._feature_cache.get(self._cache_rows[idx])
            if arrays is None:
                return None
            x_chem, x_phys, edge_index, edge_attr, fp = arrays
            return (
                torch.from_numpy(np.ascontiguousarray(x_chem)),
                torch.from_numpy(np.ascontiguousarray(x_phys)),
                torch.from_numpy(np.ascontiguousarray(edge_index)),
                torch.from_numpy(np.ascontiguousarray(edge_attr)),
                torch.from_numpy(np.ascontiguousarray(fp)),
            )
        graph_data = self._graph_cache[idx]
        if graph_data is None:
            return None
        return tuple(tensor.clone() for tensor in graph_data)

    def _fingerprint(self, idx: int) -> torch.Tensor:
        """Molecule fingerprint for row ``idx``; zeros only for unparseable SMILES.

        Read from the cache independently of the graph so that a molecule whose
        conformer embedding failed still carries its real (2D-derived) bits.
        """
        if self._feature_cache is not None:
            fp = self._feature_cache.get_fingerprint(self._cache_rows[idx])
            return torch.from_numpy(np.ascontiguousarray(fp))
        graph_data = self._graph_cache[idx]
        if graph_data is None:
            return torch.zeros((DEFAULT_FINGERPRINT_DIM,), dtype=torch.float)
        return graph_data[4].clone()

    def get(self, idx: int):
        graph_data = self._graph_tensors(idx)
        if graph_data is None:
            x_chem = torch.zeros((1, 19), dtype=torch.float)
            x_phys = torch.zeros((1, 5), dtype=torch.float)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 7), dtype=torch.float)
            fp = self._fingerprint(idx)
        else:
            x_chem, x_phys, edge_index, edge_attr, fp = graph_data

        data = Data(
            x=x_chem,
            x_chem=x_chem,
            x_phys=x_phys,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=self.labels[idx].clone(),
        )
        # [1, F], never [F]: PyG concatenates attributes whose name carries no
        # "index" along dim 0, so a 2D row batches to [B, F] while a 1D vector
        # would be flattened into a single [B*F] tensor.
        data.fp = fp.view(1, -1)
        if self.label_mask is not None:
            data.y_mask = self.label_mask[idx].clone()
        return data


def target_statistics(
    dataframe: pd.DataFrame,
    target_columns: list[str],
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-task mean and std of the training targets, for regression standardization.

    Computed on the train split only -- using val/test statistics would leak the
    held-out label distribution into training. Degenerate (near-zero) stds are
    replaced with 1.0 so a constant task cannot produce infinities.
    """
    labels = dataframe.iloc[indices][target_columns].to_numpy(dtype=float)
    mean = np.nanmean(labels, axis=0)
    scale = np.nanstd(labels, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean.astype(np.float32), scale.astype(np.float32)


def create_dual_datasets(
    data_path: str,
    target_columns: list[str],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
    smiles_column: str = "smiles",
    task_type: str = "classification",
    feature_cache=None,
):
    """Build the three dual-branch splits, sharing one featurization cache."""
    def build(indices):
        return MoleculeDualDataset(
            data_path,
            target_columns,
            indices,
            smiles_column=smiles_column,
            task_type=task_type,
            feature_cache=feature_cache,
        )

    return build(train_indices), build(val_indices), build(test_indices)


def create_datasets(
    data_path: str,
    target_columns: list[str],
    seed: int = 42,
    dual: bool = False,
    smiles_column: str = "smiles",
    split_type: SplitType = "deterministic_scaffold",
    train_size: float = 0.8,
    val_size: float = 0.1,
    task_type: str = "classification",
    feature_cache=None,
):
    dataframe = pd.read_csv(data_path)
    train_indices, val_indices, test_indices = split_indices(
        dataframe,
        split_type=split_type,
        target_columns=target_columns,
        smiles_column=smiles_column,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        task_type=task_type,
    )
    rdkit_desc_mean = None
    rdkit_desc_scale = None
    if not dual:
        train_smiles = dataframe.iloc[train_indices][smiles_column].tolist()
        rdkit_desc_mean, rdkit_desc_scale = fit_rdkit_descriptor_scaler(train_smiles)
    print(
        f"Split '{split_type}' for {'dual' if dual else 'standard'} features: "
        f"train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )
    describe_splits(dataframe, target_columns, train_indices, val_indices, test_indices, task_type)
    if dual:
        return create_dual_datasets(
            data_path,
            target_columns,
            train_indices,
            val_indices,
            test_indices,
            smiles_column=smiles_column,
            task_type=task_type,
            feature_cache=feature_cache,
        )
    return (
        MoleculeDataset(data_path, target_columns, train_indices, rdkit_desc_mean, rdkit_desc_scale, smiles_column),
        MoleculeDataset(data_path, target_columns, val_indices, rdkit_desc_mean, rdkit_desc_scale, smiles_column),
        MoleculeDataset(data_path, target_columns, test_indices, rdkit_desc_mean, rdkit_desc_scale, smiles_column),
    )
