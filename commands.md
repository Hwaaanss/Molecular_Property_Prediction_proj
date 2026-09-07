# Commands

CLI reference for **DiKAT** (`dual_kd_gnn`) across 12 MoleculeNet datasets —
7 classification (ROC-AUC, higher better) and 5 regression (RMSE, lower better).

Task type, target columns, and the metric are resolved from the dataset registry
([common/datasets.py](common/datasets.py)), so the loss, the model head, and the
early-stopping direction all adapt to whichever dataset you select.

| Dataset | File | SMILES col | Molecules | Tasks | Type | Metric | Tuned config |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bace     | `data/bace.csv`     | `mol`    | 1,513  | 1   | classification | ROC-AUC | ✔ |
| bbbp     | `data/bbbp.csv`     | `smiles` | 2,050  | 1   | classification | ROC-AUC | ✔ |
| clintox  | `data/clintox.csv`  | `smiles` | 1,484  | 2   | classification | ROC-AUC | ✔ |
| sider    | `data/sider.csv`    | `smiles` | 1,427  | 27  | classification | ROC-AUC | ✔ |
| tox21    | `data/tox21.csv`    | `smiles` | 7,831  | 12  | classification | ROC-AUC | ✔ |
| toxcast  | `data/toxcast.csv`  | `smiles` | 8,597  | 617 | classification | ROC-AUC | ✔ |
| hiv      | `data/hiv.csv`      | `smiles` | 41,127 | 1   | classification | ROC-AUC | ✔ |
| freesolv | `data/freesolv.csv` | `smiles` | 642    | 1   | regression | RMSE | ✔ |
| esol     | `data/esol.csv`     | `smiles` | 1,128  | 1   | regression | RMSE | ✔ |
| lipo     | `data/lipo.csv`     | `smiles` | 4,200  | 1   | regression | RMSE | ✔ |
| malaria  | `data/malaria.csv`  | `smiles` | 9,999  | 1   | regression | RMSE | ✔ |
| cep      | `data/cep.csv`      | `smiles` | 29,978 | 1   | regression | RMSE | ✔ |

Most scripts accept the group aliases **`all`**, **`classification`**,
**`regression`** wherever a dataset list is expected.

---

## 0. Environment and resource budget

### Create the environment (conda only, inside the project)

`--prefix` puts the environment in the project directory instead of in the
conda installation's shared `envs/`, so it travels with the checkout and cannot
be confused with another project's. `envs/` is gitignored.

```bash
cd main_v2
conda env create --prefix ./envs/dikat -f environment.yml
conda activate ./envs/dikat
```

Two conveniences worth setting once, globally:

```bash
conda config --set env_prompt '({name})'   # show "(dikat)", not the whole path
conda config --set solver libmamba         # much faster solve for this env
```

Everything installs from `conda-forge` alone. Mixing the `pytorch` channel with
`conda-forge` is the usual way this kind of environment becomes unsolvable — the
two build against different CUDA and MKL packages — so `environment.yml` pins
`nodefaults` and takes PyTorch from conda-forge as well. `torch-scatter` and
`torch-sparse` are deliberately absent: this code uses only `GCNConv`,
`GINEConv`, `GATConv` and `global_mean_pool`, which PyG 2.x implements with
native torch scatter ops. Those two packages are the usual reason a PyG install
has to fall back to pip wheels, and not needing them is what keeps this
installable with conda alone.

Verify:

```bash
python -c "import torch, torch_geometric, rdkit, optuna, scipy, sklearn; \
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

`requirements.txt` is kept for reference; the supported path is the conda file.

To update after editing `environment.yml`:

```bash
conda env update --prefix ./envs/dikat -f environment.yml --prune
```

### Resource budget

The host reports 128 cores, but the job is entitled to far fewer; PyTorch would
otherwise size its thread pool from `os.cpu_count()` and get the process killed.
[common/resources.py](common/resources.py) pins every thread pool to a budget:

| Variable | Default | Meaning |
| --- | --- | --- |
| `DIKAT_CPU_BUDGET` | `8` | Total cores the job may use |
| `DIKAT_LOADER_WORKERS` | `budget-1`, max 4 | DataLoader worker processes |
| `DIKAT_FEATURIZE_WORKERS` | `budget-1` | Parallel featurization processes |
| `DIKAT_TF32` | `1` | TF32 matmuls on Ampere; `0` forces strict FP32 |
| `CUDA_VISIBLE_DEVICES` | — | Set this to pick the GPU |

```bash
export CUDA_VISIBLE_DEVICES=1
export DIKAT_CPU_BUDGET=8
```

---

## 1. Download datasets

```bash
python scripts/download_data.py                 # all 12
python scripts/download_data.py classification  # the 7 classification sets
python scripts/download_data.py regression      # the 5 regression sets
python scripts/download_data.py bbbp bace       # a subset
python scripts/download_data.py --force tox21   # re-download
```

Gzipped sources are decompressed automatically. Malaria ships without a header
row, so the downloader writes one (`activity,smiles`) and then validates that
every declared target column really exists.

CEP and Malaria are no longer on the DeepChem bucket; they are fetched from the
AttentiveFP repository, which carries the same MoleculeNet-derived CSVs.

---

## 2. Precompute the featurization cache

The dual-branch featurizer embeds a 3D conformer and runs MMFF94 per molecule.
Without a cache every run repeats it — hours per sweep on HIV and CEP.

```bash
python scripts/precompute_features.py --datasets all --workers 7
python scripts/precompute_features.py --datasets regression
python scripts/precompute_features.py --datasets hiv --force   # rebuild
```

Caches land in `data/cache/<dataset>_dualv1_<hash>.npz`. The hash is taken over
the SMILES column, so editing a CSV invalidates the cache automatically. Run
this **once before any sweep**; training runs then hit the cache.

---

## 3. Verify the splits (no GPU)

```bash
python scripts/verify_splits.py                                   # all datasets × 4 protocols
python scripts/verify_splits.py --datasets bbbp bace --split-types scaffold
python scripts/verify_splits.py --csv results/artifacts/split_diagnostics.csv
```

Reports split sizes, tasks with an undefined ROC-AUC, and the train→test label
shift; asserts that scaffold groups never straddle two splits, that `scaffold`
is seed-independent, and that `random_scaffold`/`random` are not.

### Split protocols

| `--split-type` | Description |
| --- | --- |
| `scaffold` *(default)* | Standard Bemis-Murcko scaffold split (Hu et al. 2020 / DeepChem). Deterministic — no seed. `deterministic_scaffold` is accepted as the same thing. |
| `random_scaffold` | Same grouping, seed-shuffled group order (Uni-Mol family) |
| `random` | Uniform random split over row indices |
| `label_aware_scaffold` | Non-standard; kept only to reproduce pre-refactor results. Classification only. |

---

## 4. Train a single dataset

[dual_kd_gnn/main.py](dual_kd_gnn/main.py) writes to `dual_kd_gnn/runs/<dataset>/`.

```bash
python dual_kd_gnn/main.py --dataset bbbp
python dual_kd_gnn/main.py --dataset esol                     # regression works the same way
python dual_kd_gnn/main.py --dataset sider --gcn-pretrain-epochs 150 --head-epochs 150
python dual_kd_gnn/main.py --dataset bbbp --seeds 0 1 2 3 42  # multi-seed, prints mean ± std
```

Common overrides:

- Schedule: `--gcn-pretrain-epochs` (phase 1), `--head-epochs` (phase 2), `--patience`, `--num-epochs`
- Optimization: `--batch-size`, `--lr`, `--pretrain-lr`, `--head-lr`, `--weight-decay`
- Distillation: `--distill-weight`, `--cross-distill-weight`, `--ema-decay`, `--ema-decay-init`
- Architecture: `--gnn-conv {gcn,gine,gat}`, `--gnn-hidden`, `--gnn-layers`, `--fusion-dropout`,
  `--ih-rank`, `--ih-num-prototypes`, `--ih-proj-dim`, `--ih-block-proj`
- Fingerprint: `--use-fingerprint`, `--fp-dim`, `--fp-dropout`, `--fp-stage {head,both}`
- Data: `--data-path`, `--smiles-column`, `--target-columns`, `--split-type {scaffold,random_scaffold,random,label_aware_scaffold}`
- Device: `--device cuda` / `cuda:1` / `cpu`

The task type is taken from the dataset registry, never from a flag: it sets the
loss, the reported metric and the early-stopping direction together, so a
regression set trains on standardized targets with masked MSE and is scored by
RMSE in the original units. `metrics.json` carries `task_type`, `metric_name`,
`split_protocol` and `test_metric`, the same field names the sweep writes, so
single runs and sweep runs aggregate into one table.

### Train from a tuned config

```bash
python dual_kd_gnn/main.py --dataset bbbp \
    --best-config dual_kd_gnn/optuna/bbbp_xkd/best_config.json
```

A loaded config is passed through the same sanitizers the sweep uses: the
removed transformer knobs are dropped, `tf_dropout` → `fusion_dropout` and
`transformer_{epochs,lr}` → `head_{epochs,lr}` are renamed, and `ih_symmetric`
is pinned to `False` (9 of the 12 pre-refactor `_xkd` configs set it `True`, and
a symmetric head is PSD, which collapses to bias-only on imbalanced or
regression targets and cannot recover).

The four branch switches — `--use-fingerprint`, `--ih-block-proj`,
`--no-phys-branch`, `--no-fp-branch` — are unset unless you pass them, so they
no longer overwrite a loaded config. Each has an explicit opposite
(`--no-fingerprint`, `--no-ih-block-proj`, `--phys-branch`, `--fp-branch`) for
overriding a config the other way.

---

## 5. Batch benchmark across datasets

[main.py](main.py) trains sequentially and writes a comparison table to
`results/artifacts/benchmark_summary.csv`.

```bash
python main.py                                     # the 5 default benchmark sets
python main.py --datasets bace bbbp tox21          # a subset
python main.py --seeds 0 1 2 3 42                  # 5 seeds each
python main.py --best-config dual_kd_gnn/optuna/tox21_xkd/best_config.json
```

---

## 6. Hyperparameter tuning (Optuna)

Use `--study-name <dataset>_xkd` so the result lands where the sweep looks for it.

```bash
python dual_kd_gnn/tune_optuna.py --dataset toxcast --study-name toxcast_xkd --n-trials 10
python dual_kd_gnn/tune_optuna.py --dataset esol    --study-name esol_xkd    --n-trials 10

# One study per (dataset, encoder, fingerprint) cell. --gnn-conv and the
# fingerprint switch are fixed for the study, not searched: they change every
# parameter shape, so mixing them inside one study makes its trials
# incomparable with each other.
python dual_kd_gnn/tune_optuna.py --dataset bace --study-name bace_gine_full_model_v2 \
    --n-trials 5 --gnn-conv gine --use-fingerprint
python dual_kd_gnn/tune_optuna.py --dataset bace --study-name bace_gine_no_fp_v2 \
    --n-trials 5 --gnn-conv gine --no-fingerprint
python dual_kd_gnn/tune_optuna.py --dataset sider   --study-name sider_xkd   --n-trials 40 \
    --gcn-pretrain-epochs 150 --head-epochs 150 --patience 10
```

The study direction follows the task type automatically — **maximize** ROC-AUC
for classification, **minimize** RMSE for regression.

Artifacts land in `dual_kd_gnn/optuna/<study-name>/`: `study.db`,
`best_config.json`, `trials.csv`. Studies resume if re-run with the same name.

Other knobs: `--sampler {tpe,random}`, `--pruner {hyperband,median,none}`,
`--timeout <seconds>`, `--seed`, `--storage <url>`.

### Replay a tuned config (train + evaluate on test, saves weights)

```bash
python dual_kd_gnn/tune_optuna.py --replay-best dual_kd_gnn/optuna/bbbp_xkd/best_config.json
```

---

## 7. Ablation sweep

[scripts/seed_expansion.py](scripts/seed_expansion.py) is the sweep runner. It
loads each dataset's tuned `best_config.json` (falling back to size-scaled
defaults from [dual_kd_gnn/configs.py](dual_kd_gnn/configs.py) when none exists),
applies the condition's overrides, and writes one directory per cell.

```
ablation/runs_v2/<split_type>/<condition>/<dataset>_seed<N>/metrics.json
```

Existing `metrics.json` files are skipped, so the sweep is restart-safe.

### Conditions

Four removable factors give the intact model plus all 15 non-empty removal
subsets, instantiated once per encoder — 48 conditions named
`<conv>_full_model` / `<conv>_no_<factors>` with `conv ∈ {gcn, gine, gat}` and
factors always ordered `3d, infonce, codebook, fp`.

| factor in the name | what it removes |
| --- | --- |
| `3d` | physical branch (`zero_phys_branch=True`); the branch stays wired up, so the parameter count is unchanged |
| `infonce` | cross-modal InfoNCE (`cross_distill_weight=0`) |
| `codebook` | the quadratic codebook head → plain linear head (`ih_rank=0, ih_num_prototypes=0`) |
| `fp` | fingerprint branch (`use_fingerprint=False`) |

The intact model has the fingerprint branch **on** and the head projection
block-diagonal (`ih_block_proj=True`), so geometry / topology / fingerprint
survive as separate blocks into the quadratic form.

Group aliases:

| alias | cells |
| --- | --- |
| `<conv>_paper` | 12: intact + 4 single removals + 6 pairs + the all-four cell |
| `<conv>_all` | 16: `<conv>_paper` plus the four three-factor cells |
| `<conv>_r1` / `_r2` / `_r3` / `_r4` | one removal-count slice (4 / 6 / 4 / 1 cells) |
| `paper` | `<conv>_paper` for all three encoders (36) |
| `fast` | the six `<conv>_full_model` / `<conv>_no_fp` cells |
| `full_models` | the three intact models |
| `all` | all 48 |

### Hyperparameters an ablation runs with

A cell uses, in order: its own per-cell study (`--config-template`), else **the
same encoder's `<conv>_full_model` study**, else the dataset-wide
`<dataset>_xkd` config. The middle step matters: an ablation must differ from
the model it is subtracted from in exactly the removed factor, and falling
straight through to `_xkd` would give it a different width, learning rate and
dropout as well, so the paired delta would carry the hyperparameter change too.
`config_source` in `metrics.json` records which of the three was used.

### Memory and restart behaviour

`--max-batch-size N` caps the batch regardless of the tuned config (default
`DIKAT_MAX_BATCH`, else 256; further capped to 128 at `gnn_hidden >= 512` and to
64 above 256 tasks). A cell that still runs out of memory halves its batch and
retries down to 16 rather than failing the sweep.

`metrics.json` is written **last**, after every other artifact for that run, so
a cell on disk is either absent or complete and a restart redoes at most the run
that was in flight.

### Per-run artifacts

Each cell directory holds `metrics.json`, `run_metadata.json`,
`training_log.csv`, `training_curves.png`, and — when the codebook is on —
`assignment_probs.npy`. `metrics.json` additionally carries `head_diagnostics`:
the Frobenius norm of every block pair's contribution to the learned quadratic
form (`geo-geo`, `topo-topo`, `fp-fp`, `geo-topo`, `geo-fp`, `topo-fp`) plus
each pair's share of the total. Model weights are **not** saved unless
`--save-weights` is passed (~30 MB per run).

Each ablation pairs against **its own** encoder's full model. Pairing a GINE
cell against a GCN reference would fold the encoder swap — which changes every
parameter shape — into the ablation delta.

```bash
# Screen: 3 datasets x 6 cells x 5 seeds
python scripts/seed_expansion.py --datasets bbbp bace sider \
    --conditions fast --seeds 0 1 2 3 42 --skip-random

# One encoder's full grid, every dataset
python scripts/seed_expansion.py --datasets all \
    --conditions gcn_all --seeds 0 1 2 3 42 --skip-random

# Intact model only, every encoder, every dataset
python scripts/seed_expansion.py --datasets all \
    --conditions full_models --seeds 0 1 2 3 42 --skip-random

# Replay per-cell Optuna winners instead of the dataset's shared config
python scripts/seed_expansion.py --datasets bbbp bace --conditions fast \
    --seeds 0 1 2 3 42 --skip-random --runs-root ablation/runs_v2_optuna \
    --config-template 'dual_kd_gnn/optuna/{dataset}_{condition}_v2/best_config.json'
```

### Results tree

`--runs-root` defaults to `ablation/runs_v2`, which does not exist on this
checkout — always pass the tree you mean. Trees present: `ablation/runs`
(pre-refactor, an architecture with a transformer fusion stage, never written by
the current code), `runs_v3`, `runs_v3_optuna`, `runs_v4_optuna`, and whatever
`scripts/run_pipeline.sh` creates (`ablation/runs_<TAG>`). Results from
different architectures must not be pooled into one table, which is why every
summary, CI and Wilcoxon filename carries the tree tag.

`--skip-random` skips the random-split comparison runs; `--skip-scaffold` skips
the scaffold-family sweep. `--shard I/N` runs a deterministic slice of the job
list (round-robin, so every shard gets a similar mix of dataset sizes).

### Single-run ablations via the training entry points

For a one-off run outside the sweep, `--ablation-name` routes output to
`ablation/runs/<name>/` instead of `dual_kd_gnn/runs/`:

```bash
python dual_kd_gnn/main.py --dataset bbbp --no-phys-branch \
    --ablation-name a1_no_phys --seeds 0 1 2 3 42
python main.py --ih-num-prototypes 0 --ablation-name a4_no_codebook --seeds 0 1 2 3 42
```

Available flags: `--no-phys-branch`, `--cross-distill-weight 0`,
`--distill-weight 0`, `--ih-num-prototypes 0` / `--ih-rank 0` (linear head),
`--use-fingerprint` / `--no-fp-branch`, `--gnn-conv {gcn,gine,gat}`.

---

## 8. Aggregate results

Classification and regression are **never mixed into one file** — ROC-AUC
(higher better) and RMSE (lower better) cannot share a ranking.

### 8.1 Per-cell summary

```bash
python ablation/main.py --runs-dir ablation/runs_v2     # current architecture
python ablation/main.py                                 # pre-refactor results
```

Writes `ablation/ablation_summary_{classification,regression}[_<tree>].csv`, one
row per (split protocol × condition × dataset). The tree tag comes from the
`--runs-dir` name (`runs_v2` → `_v2`), so the two architectures never overwrite
each other's table. Reads both the `<split_type>/<condition>/...` layout and the
flat pre-refactor one.

### 8.2 Confidence intervals

```bash
python scripts/compute_ci.py --seeds 0 1 2 3 42                  # both task types
python scripts/compute_ci.py --task-type regression --seeds 0 1 2 3 42
python scripts/compute_ci.py --split-type random_scaffold
python scripts/compute_ci.py --runs-root ablation/runs           # pre-refactor tree
python scripts/compute_ci.py --legacy                            # pre-refactor label-aware runs
```

> **Always pass `--seeds`.** Without it every seed directory found is used, so a
> dataset swept twice contributes n=15 while a newer one contributes n=5, and the
> table silently mixes them. Restricting seeds appends `_<n>seed` to the
> filenames, leaving the unrestricted tables in place.

Outputs under `results/artifacts/revision/`:

```
ablation_summary_with_ci_classification_scaffold_5seed.csv
ablation_summary_with_ci_regression_scaffold_5seed.csv
random_split_summary_with_ci_classification_random_5seed.csv
```

Regression tables carry two extra diagnostic columns:

- `train_test_shift_sd` — |test mean − train mean| in train standard deviations
- `r2_reliable` — `False` when that shift is ≥ 1 sd, which compresses the test
  variance R² is normalised against. Report RMSE for those datasets.

### 8.3 Significance tests

```bash
python scripts/revision_experiments.py --task c1 --split-type scaffold
python scripts/revision_experiments.py --task c1 --task-type regression
```

Paired Wilcoxon signed-rank, each ablation vs `full_model`, written to
`results/artifacts/revision/wilcoxon_tests_<task_type>_<split>_<n>seed_<tree>.csv`.
The filename carries the seed count and the results tree for the same reason
`compute_ci.py`'s does — otherwise a run over `runs_v4` silently overwrites the
table computed from `runs_v3`. `--seeds` also restricts which seeds enter the
test, so one table never mixes cells with n=5 and cells with n=15. The
`full_model_better` column is direction-aware, since a positive delta means an
improvement for ROC-AUC but a regression for RMSE.

> At n=5 the two-sided Wilcoxon minimum p-value is 0.0625, so **no cell can reach
> p<0.05**. Significance claims need more seeds.

### 8.4 Figures

```bash
python scripts/make_figures.py --task-type all --seeds 0 1 2 3 42
python scripts/make_figures.py --task-type regression --seeds 0 1 2 3 42
python scripts/make_figures.py --task-type classification --allow-incomplete
```

Validates the CSVs first — missing files, NaN standard deviations, or any cell
with fewer than two seeds cause a non-zero exit, so a half-finished sweep fails
loudly instead of producing meaningless error bars. `--allow-incomplete`
downgrades those to warnings.

Every figure is written at **300 dpi**, PNG only by default. Set
`DIKAT_FIGURE_FORMATS="png,tiff"` (or `"png,pdf"` for vector art) to add
formats. Output goes to `results/artifacts/figures/`:

```
ablation_<task_type>_<split>_<n>seed_by_dataset.{png,tiff}   grouped bars, mean ± 1 sd
ablation_<task_type>_<split>_<n>seed_delta.{png,tiff}        degradation vs full model
ablation_<task_type>_<split>_<n>seed_ci.{png,tiff}           full model, 95% CI
```

`--seeds` must match what `compute_ci.py` used, since it reads that table.

### 8.5 Figures read from the run tree

The CSV-driven figures above cannot show anything the summary tables do not
carry. These five come from the run directories themselves:

```bash
python scripts/paper_figures.py --runs-root ablation/runs_v5 \
    --split-type scaffold --task-type all --seeds 0 1 2 3 42
```

| figure | what it shows |
| --- | --- |
| `learning_curve_loss_<cond>_<task>_<suffix>.png` | train/val loss vs epoch, mean ± 1 sd over seeds, one panel per dataset, stage 1 → stage 2 marked |
| `learning_curve_metric_<cond>_<task>_<suffix>.png` | the same for ROC-AUC / RMSE |
| `block_norms_<cond>_<task>_<suffix>.png` | share of the learned quadratic form in each block pair — the figure the geometry claim rests on |
| `prototype_assignment_<dataset>_<cond>_<suffix>.png` | task × prototype assignment α, rows grouped by dominant prototype (multitask sets only) |
| `condition_rank_<conv>_<task>_<suffix>.png` | mean rank of each condition across datasets |
| `paired_delta_<conv>_<task>_<suffix>.png` | per-seed paired deltas, the quantity the Wilcoxon test consumes |

`--conditions` selects which conditions get the per-condition figures (default:
every `*_full_model` present).

### 8.6 Single-run benchmark tables

```bash
python results/main.py
```

Scans `dual_kd_gnn/runs/*` and writes `all_metrics.csv`, `results_table.csv`,
`results_table.md`, plus validation-curve and per-dataset bar figures.

---

## 9. Interpretability artifacts

```bash
python scripts/prototype_analysis.py --datasets bace bbbp clintox sider tox21
python scripts/cross_modal_alignment_analysis.py --dataset tox21 --seed 42 --epochs 50
```

Codebook assignment matrices and top-K molecule tables land in
`results/artifacts/prototypes/`; the alignment trajectory in
`results/artifacts/alignment/`.

> `prototype_analysis.py` deliberately uses `label_aware_scaffold_split`: it
> analyses checkpoints trained under that protocol, so it must look at the same
> test molecules.

---

## 10. Run everything (3 GPUs)

[scripts/run_pipeline.sh](scripts/run_pipeline.sh) drives the whole thing on
three RTX 3090s: feature cache, one Optuna study per (dataset, encoder) at seed
42, the ablation sweep sharded across the cards, then every table and figure.

```bash
nohup bash scripts/run_pipeline.sh > logs/pipeline.log 2>&1 &
echo $! > pipeline.pid
tail -f logs/pipeline.log
```

| Phase | Step | Parallelism |
| --- | --- | --- |
| 0 | Feature cache for all 12 datasets | 1 process, N featurize workers |
| 1 | Optuna: 12 datasets × 3 encoders, seed 42 | 3 workers, one per GPU |
| 2 | Ablation sweep | 3 shards, one per GPU |
| 3 | Summary → CI → Wilcoxon → figures | 1 process |

One training process per card. Two on one 24 GB card would each get ~12 GB and
the large multitask sets do not fit in that, so the parallelism is across cards
only. The CPU budget is the host core count divided by the number of GPUs,
capped at 8 per process.

Overrides:

```bash
PHASES="2 3" bash scripts/run_pipeline.sh              # sweep + results only
GPUS="0 1"   bash scripts/run_pipeline.sh              # two cards
CONDITIONS="gcn_all gine_all gat_all" bash scripts/run_pipeline.sh   # full 16 per encoder
TAG=v6 TRIALS=20 SEEDS="0 1 2 3 4 5 42" bash scripts/run_pipeline.sh
MAX_BATCH=128 bash scripts/run_pipeline.sh             # tighter VRAM cap
DRY_RUN=1 bash scripts/run_pipeline.sh                 # print the plan, run nothing
```

Restarting after a crash is the same command. Optuna studies resume from
`study.db`; a study whose `best_config.json` exists is skipped; sweep cells with
a `metrics.json` are skipped. Phase 3 is pure aggregation and can be run at any
time, including while phase 2 is still going, to see partial results.

Monitor:

```bash
tail -f logs/v5_sweep_gpu0.log
find ablation/runs_v5 -name metrics.json | wc -l
nvidia-smi
kill $(cat pipeline.pid)
```

## Output layout

```
data/
├── <dataset>.csv                      # downloaded datasets
└── cache/<dataset>_dualv1_<hash>.npz  # featurization cache

dual_kd_gnn/
├── runs/<dataset>/                    # single-run artifacts
│   ├── metrics.json                   # best val metric, test metric, #params, runtime
│   ├── run_metadata.json              # hparams, model_kwargs, data path, device
│   ├── training_log.csv               # per-epoch train/val loss and metric
│   ├── model_weights.pt
│   └── training_curves.{png,tiff}
└── optuna/<dataset>_xkd/              # study.db, best_config.json, trials.csv

ablation/
├── runs_<TAG>/<split_type>/<condition>/<dataset>_seed<N>/
│   ├── metrics.json          # written last; its presence = this cell is done
│   ├── run_metadata.json     # model_kwargs, hparams, config_source
│   ├── training_log.csv      # per-epoch, phase-tagged
│   ├── training_curves.png
│   └── assignment_probs.npy  # task x prototype, when the codebook is on
├── runs_<TAG>/full_model_random/<dataset>_seed<N>/   # matched-protocol random split
└── ablation_summary_{classification,regression}_<TAG>.csv

results/artifacts/
├── revision/    # CI tables, Wilcoxon tests
├── figures/     # ablation figures (300 dpi PNG + TIFF)
├── prototypes/  # codebook interpretability
└── alignment/   # cross-modal alignment trajectory
```

### `metrics.json` fields

Common: `dataset`, `task_type`, `metric_name`, `greater_is_better`,
`split_protocol`, `n_train`, `n_val`, `n_test`, `best_val_metric`, `best_epoch`,
`test_metric`, `config_source`, `num_parameters`, `elapsed_seconds`, `seed`.

Classification adds `test_roc_auc` and `n_tasks_auc_defined_{val,test}`;
regression adds `test_rmse`, `test_mae`, `test_r2`.

Sweep runs also carry `batch_size`, `oom_retry_batch_sizes`, and
`head_diagnostics` — `{uses_codebook, blocks_preserved, block_names,
effective_block_dims, block_norms, block_norms_fraction}`. `block_norms` is
`null` with an explanation when a full mixing projection destroyed the block
partition (`ih_block_proj=False`), and all-zero when the codebook head was
ablated away to a plain linear head.

> The run directory is `ablation/runs/deterministic_scaffold/` while
> `split_protocol` reads `scaffold`. `deterministic_scaffold` is the internal key
> that distinguishes this protocol from `random_scaffold`; `scaffold` is the name
> used in every generated artifact. The directory keeps the old name so completed
> runs stay discoverable.
