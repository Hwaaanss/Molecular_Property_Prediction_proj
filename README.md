# DiKAT: Dichotomous Knowledge Alignment via Tensors for Molecular Property Prediction

Reference implementation for the paper *"DiKAT: Dichotomous Knowledge Alignment via Tensors for Molecular Property Prediction"* (Lee and Kim, submitted to JCIM).

DiKAT is a dual-branch graph neural network that decomposes molecular representation into a topology-aware chemical branch and a geometry-aware physical branch (MMFF94-derived 3D coordinates + van der Waals radii). Training runs in two phases: phase 1 pretrains both view encoders under EMA self-distillation, a cross-modal InfoNCE term and the task loss; phase 2 freezes them and fine-tunes the head on the pooled fusion vector. Predictions come from a codebook-shared interaction-tensor head whose logit is a pure quadratic form `z^T W_k z + b_k` with `W_k` built from shared prototypes and per-task routing weights (Gumbel-softmax). The message-passing operator is selectable: `--gnn-conv {gcn,gine,gat}`.

## Environment

- Python 3.10 – 3.12
- PyTorch (CUDA recommended)
- PyTorch Geometric
- RDKit (install via `conda` if pip fails)
- Optuna, NumPy, pandas, scikit-learn, SciPy, matplotlib, Pillow

```bash
conda activate dualgnn
pip install -r requirements.txt
```

### Resource budget

The training host reports 128 cores but the job is entitled to far fewer. Left
alone, PyTorch sizes its intra-op thread pool from `os.cpu_count()` (64 threads)
and every DataLoader worker inherits an OpenMP pool of the same size, which
overruns a small core allocation immediately. `common/resources.py` pins every
thread pool to a declared budget and is imported before torch in each entry
point:

```bash
export CUDA_VISIBLE_DEVICES=1     # single GPU
export DIKAT_CPU_BUDGET=8         # total cores; loader/featurize workers derive from this
```

Measured steady state for one training run under this budget: ~1.4 cores, ~1.9 GB RSS.

## Data

12 MoleculeNet datasets — 7 classification (ROC-AUC) and 5 regression (RMSE):

| Type | Datasets |
| --- | --- |
| Classification | BACE, BBBP, ClinTox, SIDER, Tox21, ToxCast, HIV |
| Regression | FreeSolv, ESOL, Lipophilicity, Malaria, CEP |

```bash
python scripts/download_data.py                 # all 12
python scripts/download_data.py classification  # or: regression
```

Most sources come from the DeepChem distribution. CEP and Malaria are no longer
hosted there and are fetched from the AttentiveFP repository, which carries the
same MoleculeNet-derived CSVs; Malaria ships headerless, so the downloader writes
its header and validates the columns.

### Featurization cache (run this before any sweep)

The dual-branch featurizer embeds a 3D conformer and runs MMFF94 per molecule —
tens of hours across a full sweep if repeated per run. Featurize once:

```bash
python scripts/precompute_features.py --datasets all --workers 7
```

Caches are keyed by a hash of the SMILES column, so editing a CSV invalidates
them automatically.

## Splits

The default protocol is the **standard Bemis-Murcko scaffold split** (Hu et al.
2020 / DeepChem `ScaffoldSplitter`): scaffold groups sorted by (size, first
index) descending, poured into train → val → test at 80/10/10. It is
deterministic by construction, so it takes no seed. Three alternatives are
available: `random_scaffold` (seed-shuffled group order, Uni-Mol family),
`random`, and `label_aware_scaffold` (non-standard, retained only to reproduce
pre-refactor results).

```bash
python scripts/verify_splits.py    # sizes, AUC-definability, group-integrity asserts
```

> Generated artifacts name the default protocol `scaffold`. The internal key
> `deterministic_scaffold` — which distinguishes it from `random_scaffold` — is
> what run directories use.

## Reproducing the reported numbers

Per-dataset best hyperparameters live in
`dual_kd_gnn/optuna/<dataset>_xkd/best_config.json`. Results reported in the
paper use 5 seeds `{0, 1, 2, 3, 42}` on the scaffold split.

The whole pipeline — Optuna where a config is missing, ablation sweep, CSV
validation, figures, significance tests — runs end to end, smallest datasets
first, skipping anything already complete:

```bash
nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
```

### Individual stages

```bash
# Everything, in three sets, sequentially on one GPU
bash scripts/run_v2_experiments.sh
STAGES="fast" bash scripts/run_v2_experiments.sh    # one set only
DRY_RUN=1     bash scripts/run_v2_experiments.sh    # print the plan

# Or by hand — screen: 3 datasets × 6 cells × 5 seeds
python scripts/seed_expansion.py --datasets bbbp bace sider \
    --conditions fast --seeds 0 1 2 3 42 --skip-random

# One encoder's full ablation grid, every dataset
python scripts/seed_expansion.py --datasets all \
    --conditions gcn_all --seeds 0 1 2 3 42 --skip-random

# Aggregate, test, plot
python ablation/main.py --runs-dir ablation/runs_v2
python scripts/compute_ci.py --runs-root ablation/runs_v2 --seeds 0 1 2 3 42
python scripts/revision_experiments.py --task c1 --runs-root ablation/runs_v2
python scripts/make_figures.py --task-type all --seeds 0 1 2 3 42 --results-tree runs_v2
```

Per-run outputs land in
`ablation/runs_v2/<split_type>/<condition>/<dataset>_seed<N>/metrics.json`.
Classification and regression are aggregated into separate CSVs throughout,
since ROC-AUC (higher better) and RMSE (lower better) cannot share a ranking.

### Ablation conditions

Four removable factors — the 3D (physical) branch, the cross-modal InfoNCE, the
codebook head, and the fingerprint branch — give the intact model plus all 15
non-empty removal subsets. Each is instantiated once per message-passing
encoder, so condition names read `<conv>_full_model` and `<conv>_no_<factors>`
with `conv ∈ {gcn, gine, gat}` and factors listed in the order
`3d, infonce, codebook, fp`:

| condition | meaning |
|---|---|
| `gcn_full_model` | intact: fingerprint on, block-diagonal head projection on |
| `gcn_no_3d` | `x_phys = 0`; the branch stays wired up, so the parameter count is unchanged |
| `gcn_no_infonce` | `cross_distill_weight = 0` |
| `gcn_no_codebook` | quadratic codebook head → plain linear head |
| `gcn_no_fp` | fingerprint branch removed |
| `gcn_no_3d_fp`, … | the 2-, 3- and 4-factor combinations |

Group aliases save spelling all 48 out: `--conditions fast` is the six
`<conv>_full_model` / `<conv>_no_fp` cells, `--conditions gcn_all` is one
encoder's sixteen, `--conditions full_models` is the three intact models.

Each ablation is paired against **its own** encoder's full model. A GINE cell
compared against a GCN reference would fold the encoder swap — which changes
every parameter shape — into the ablation delta.

Results of this architecture live in `ablation/runs_v2/`. The pre-refactor
results in `ablation/runs/` came from a version with a transformer fusion stage
and must not be pooled with them; the aggregation script takes `--runs-dir` and
writes a separate `*_v2.csv` for each tree.

### Fingerprint branch

A molecule-level fingerprint (Morgan ECFP4 1024 + PubChem-like 881 + ErG 441 =
2346 bits) enters as a third fusion block. It has no node dimension, so it is
never fed to message passing or broadcast over atoms: a small encoder
(2346 → 64) produces one vector per molecule that is concatenated onto the
pooled `[geometry, topology]` vector, giving the head a third block and three
new cross-blocks in its quadratic form — geometry×fingerprint being the one
that ties substructure identity to steric environment.

Two constraints make it interpretable rather than merely additive. The head's
projection is block-diagonal (`--ih-block-proj`), so no weight mixes the blocks
and `InteractionTensorHead.quadratic_block_norms()` can report all six block
pairs. And the branch trains in the head phase only (`--fp-stage head`, the
default; `stage2` is accepted as the old spelling): a fingerprint is a strong
predictor on its own, so opening it during encoder pretraining would let the
head read labels off it and weaken the gradient reaching the GNN encoders,
which are frozen from phase 2 onwards. Phase 1 substitutes an exact zero vector
in the fingerprint slot instead, keeping the single warm-started classifier at
one shape across both phases.

```bash
python dual_kd_gnn/main.py --dataset bace --use-fingerprint --ih-block-proj
```

Defaults leave the branch off, so runs without `--use-fingerprint` are
structurally and numerically identical to those from before it existed.

### Interpretability artifacts

```bash
python scripts/prototype_analysis.py --datasets bace bbbp clintox sider tox21
python scripts/cross_modal_alignment_analysis.py --dataset tox21 --seed 42
```

## Reading the results

Every figure is written as 300 dpi PNG and LZW-compressed TIFF under
`results/artifacts/figures/`. Tables land in `results/artifacts/revision/`.

Two caveats are worth knowing before interpreting the numbers:

- **Seed counts.** `compute_ci.py` uses every seed directory it finds unless
  `--seeds` is given, so a dataset swept twice can contribute n=15 alongside a
  newer one at n=5. Always pass `--seeds`; doing so appends `_<n>seed` to the
  output filenames.
- **Malaria's negative R².** The Malaria CSV is sorted by activity, and roughly
  half its molecules form singleton scaffold groups, so the scaffold split's
  (size, first index) tie-break sends the file's highest-activity rows to test —
  a 1.59 train-sd label shift. That compresses the test variance R² is
  normalised against. The model still beats every baseline available to it
  (RMSE 1.836 vs 2.073 for predicting the train mean), so report RMSE there. The
  regression tables carry `train_test_shift_sd` and `r2_reliable` columns
  recording this.

At n=5 the two-sided Wilcoxon minimum p-value is 0.0625, so no single cell can
reach p<0.05; significance claims need more seeds.

## Repository layout

```
common/
  data.py                 Featurisation, the four split protocols, dataset classes
  datasets.py             Dataset registry (paths, targets, task type, metric)
  featurize_cache.py      Parallel disk-backed MMFF featurization cache
  resources.py            CPU budget / thread pinning
  trainer.py              Two-stage trainer (classification + regression)
  metrics.py              ROC-AUC and RMSE/MAE/R²
  plotting.py             300 dpi PNG + TIFF figure writer
dual_kd_gnn/
  model.py                DualDistillationModel definition
  main.py                 Standalone training entry point
  tune_optuna.py          Optuna TPE + Hyperband search
  configs.py              Tuned-config loader with size-scaled fallbacks
  optuna/<dataset>_xkd/   Per-dataset best_config.json
ablation/
  main.py                 Per-cell summary CSVs (split by task type)
  runs/                   Per-run outputs
scripts/
  download_data.py        Dataset downloader
  precompute_features.py  Build the featurization cache
  verify_splits.py        Split diagnostics + invariant asserts
  seed_expansion.py       Ablation sweep runner
  compute_ci.py           Student-t 95% CI aggregator
  revision_experiments.py Wilcoxon tests, MMFF failure rates, random-split runs
  make_figures.py         CSV validation + ablation figures
  prototype_analysis.py   Codebook interpretability
  cross_modal_alignment_analysis.py  Alignment trajectory
  run_all.sh              Six-phase end-to-end pipeline
results/artifacts/        Aggregated CSVs and figures
```

See [commands.md](commands.md) for the full CLI reference.

## Citation

To be added on acceptance.

## License

MIT. See `LICENSE`.
