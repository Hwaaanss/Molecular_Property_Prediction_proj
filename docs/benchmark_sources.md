# Benchmark Source Notes

This project evaluates a single model, `dual_kd_gnn`, across several MoleculeNet
classification datasets. (Earlier revisions compared multiple GNN baselines on
Tox21; those baseline packages have been removed.)

## Model

- `dual_kd_gnn`: dual-branch GCN (chemical + physical atom features) with an
  EMA-teacher knowledge-distillation stage, a transformer fusion stage, and a
  codebook-shared interaction-tensor classifier head. The classifier head and
  loss adapt to the number of target tasks of the active dataset.

## Datasets (MoleculeNet, via DeepChem S3)

All CSVs are downloaded from
`https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/` (see
[../scripts/download_data.py](../scripts/download_data.py) and
[../commands.md](../commands.md)).

- BACE: `bace.csv` — beta-secretase 1 (BACE-1) inhibition, 1 task.
- BBBP: `BBBP.csv` — blood-brain barrier penetration, 1 task.
- SIDER: `sider.csv.gz` — 27 adverse-reaction system-organ-class tasks.
- Tox21: `tox21.csv.gz` — 12 toxicity assay tasks.
- ClinTox: `clintox.csv.gz` — FDA approval vs. clinical-trial toxicity, 2 tasks.

## Primary references

- MoleculeNet benchmark: https://moleculenet.org/
- MoleculeNet paper: https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a
- DeepChem (dataset hosting): https://github.com/deepchem/deepchem
