#!/usr/bin/env bash
# DiKAT full pipeline on three RTX 3090s (24 GB each), GPUs 0/1/2.
#
#   conda activate ./envs/dikat                     # see commands.md section 0
#   bash scripts/run_pipeline.sh                    # everything, phases 0-3
#   PHASES="2 3" bash scripts/run_pipeline.sh       # ablation + results only
#   DRY_RUN=1  bash scripts/run_pipeline.sh         # print the plan, run nothing
#
# Resumability is the design constraint, because a 3-day sweep will be
# interrupted. Nothing here depends on a previous phase still being in memory:
#   phase 1  Optuna studies live in dual_kd_gnn/optuna/<study>/study.db and are
#            opened with load_if_exists, so a killed study resumes at the trial
#            it reached. A study whose best_config.json already exists is
#            skipped outright.
#   phase 2  seed_expansion.py writes metrics.json last, after every other
#            artifact for that run, and skips any cell that already has one. So
#            a cell is either absent or complete -- never half-recorded -- and a
#            restart redoes at most the one run that was in flight per GPU.
#   phase 3  pure aggregation over what is on disk; safe to run at any time,
#            including while phase 2 is still going, to see partial results.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs

# ── GPUs ─────────────────────────────────────────────────────────────────────
# One training process per card. Two processes on one 24 GB card would each get
# ~12 GB and the large multitask sets do not fit in that, so the parallelism is
# across cards only.
GPUS="${GPUS:-0 1 2}"
read -r -a GPU_ARRAY <<< "$GPUS"
NGPU=${#GPU_ARRAY[@]}

# ── CPU budget, per process ──────────────────────────────────────────────────
# Shared across the concurrent workers: the host core count divided by the
# number of GPUs, capped at 8. Left unpinned, every worker's OpenMP pool sizes
# itself from the host's full core count and the box thrashes.
TOTAL_CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"
PER_PROC=$(( TOTAL_CORES / NGPU ))
[ "$PER_PROC" -lt 4 ] && PER_PROC=4
[ "$PER_PROC" -gt 8 ] && PER_PROC=8
export DIKAT_CPU_BUDGET="${DIKAT_CPU_BUDGET:-$PER_PROC}"
export DIKAT_LOADER_WORKERS="${DIKAT_LOADER_WORKERS:-$(( DIKAT_CPU_BUDGET > 4 ? 4 : DIKAT_CPU_BUDGET - 1 ))}"
export DIKAT_FEATURIZE_WORKERS="${DIKAT_FEATURIZE_WORKERS:-$(( TOTAL_CORES > 8 ? 7 : TOTAL_CORES - 1 ))}"
# Ampere TF32: free speedup, ~1e-3 relative shift, well inside seed noise.
export DIKAT_TF32="${DIKAT_TF32:-1}"
# 300 dpi PNG only. Set to "png,pdf" if the venue wants vector figures too.
export DIKAT_FIGURE_FORMATS="${DIKAT_FIGURE_FORMATS:-png}"
# Long sweeps fragment the allocator; expandable segments keep a 24 GB card
# from OOMing on a batch it was fitting an hour earlier.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Experiment grid ──────────────────────────────────────────────────────────
PY="${PY:-$(command -v python)}"
TAG="${TAG:-v5}"
DATASETS="${DATASETS:-freesolv esol sider clintox bace bbbp lipo tox21 toxcast malaria cep hiv}"
CONVS="${CONVS:-gcn gine gat}"
HPO_SEED="${HPO_SEED:-42}"
TRIALS="${TRIALS:-10}"
SEEDS="${SEEDS:-0 1 2 3 42}"
SPLIT="${SPLIT:-scaffold}"
SPLIT_DIR="${SPLIT_DIR:-deterministic_scaffold}"
# One factor out (4), two out (6), all four out (1), plus the intact model = 12
# conditions per encoder. Set CONDITIONS="gcn_all gine_all gat_all" for the full
# 16 (adds the four three-factor cells).
CONDITIONS="${CONDITIONS:-gcn_paper gine_paper gat_paper}"
RUNS_ROOT="${RUNS_ROOT:-ablation/runs_${TAG}}"
RESULTS_TREE="$(basename "$RUNS_ROOT")"
CONFIG_TEMPLATE="dual_kd_gnn/optuna/{dataset}_{condition}_${TAG}/best_config.json"
MAX_BATCH="${MAX_BATCH:-256}"
PHASES="${PHASES:-0 1 2 3}"
DRY_RUN="${DRY_RUN:-0}"

has_phase () { [[ " $PHASES " == *" $1 "* ]]; }
stamp ()     { date '+%F %T'; }

run () {   # run <label> <logfile> <command...>
  local name="$1" log="$2"; shift 2
  echo "──── $name  |  $(stamp)"
  echo "     \$ $*"
  if [ "$DRY_RUN" = "1" ]; then echo "     (dry-run)"; return 0; fi
  if "$@" 2>&1 | tee "logs/$log"; then echo "     ok  $name"; else echo "     FAILED  $name (continuing)"; fi
}

# ── Environment sanity check ─────────────────────────────────────────────────
# Fail here, in the first second, rather than three phases and several hours in.
# The usual cause is a forgotten `conda activate ./envs/dikat`, which leaves $PY
# pointing at the base interpreter where torch is absent.
if [ "$DRY_RUN" != "1" ]; then
  if [ -z "$PY" ] || [ ! -x "$PY" ]; then
    echo "No python on PATH. Activate the environment first:"
    echo "    conda activate ./envs/dikat"
    exit 1
  fi
  if ! "$PY" - <<'PYCHECK'
import sys
missing = []
for module in ("torch", "torch_geometric", "rdkit", "optuna", "scipy",
               "sklearn", "pandas", "numpy", "matplotlib"):
    try:
        __import__(module)
    except ImportError:
        missing.append(module)
if missing:
    print("missing packages: " + ", ".join(missing))
    sys.exit(1)
import torch
print(f"    torch {torch.__version__} | cuda available={torch.cuda.is_available()} "
      f"| visible devices={torch.cuda.device_count()}")
PYCHECK
  then
    echo "Environment is incomplete. Create/update it with:"
    echo "    conda env create --prefix ./envs/dikat -f environment.yml"
    echo "    conda activate ./envs/dikat"
    exit 1
  fi
fi

n_ds=$(echo "$DATASETS" | wc -w); n_conv=$(echo "$CONVS" | wc -w); n_seed=$(echo "$SEEDS" | wc -w)
echo "══════════════════════════════════════════════════════════════"
echo " DiKAT pipeline   $(stamp)"
echo "   python      : $PY"
echo "   GPUs        : $GPUS  ($NGPU cards, one process each)"
echo "   CPU/process : $DIKAT_CPU_BUDGET cores (host has $TOTAL_CORES)"
echo "   datasets    : $n_ds"
echo "   encoders    : $CONVS"
echo "   HPO         : $(( n_ds * n_conv )) studies x $TRIALS trials, seed $HPO_SEED"
echo "   conditions  : $CONDITIONS"
echo "   seeds       : $SEEDS  ($n_seed)"
echo "   split       : $SPLIT"
echo "   runs root   : $RUNS_ROOT"
echo "   max batch   : $MAX_BATCH (auto-halves on OOM)"
echo "   phases      : $PHASES"
echo "══════════════════════════════════════════════════════════════"

# ── (0) feature cache ────────────────────────────────────────────────────────
# Conformer embedding + MMFF94 once per dataset, shared by every later run.
# Single process: it is CPU-bound and already parallel internally.
if has_phase 0; then
  echo; echo "════ (0/3) feature cache"
  run "features" p0_features.log \
    "$PY" -u scripts/precompute_features.py --datasets all --workers "$DIKAT_FEATURIZE_WORKERS"
fi

# ── (1) Optuna HPO: one study per (dataset, encoder), seed 42 ────────────────
# The study is the intact model for that encoder. Every ablation of that encoder
# inherits this config (seed_expansion falls back to the full-model sibling), so
# an ablation differs from its reference in exactly the removed factor and
# nothing else -- which is the whole point of a paired ablation.
if has_phase 1; then
  echo; echo "════ (1/3) Optuna  $(( n_ds * n_conv )) studies x $TRIALS trials"
  JOBS=()
  for ds in $DATASETS; do for conv in $CONVS; do JOBS+=("$ds:$conv"); done; done

  hpo_worker () {          # hpo_worker <slot>
    local slot="$1" gpu="${GPU_ARRAY[$1]}" i=0
    for job in "${JOBS[@]}"; do
      if [ $(( i % NGPU )) -eq "$slot" ]; then
        local ds="${job%%:*}" conv="${job##*:}"
        local study="${ds}_${conv}_full_model_${TAG}"
        if [ -f "dual_kd_gnn/optuna/${study}/best_config.json" ]; then
          echo "[gpu$gpu] skip $study (best_config exists)"
        else
          echo "[gpu$gpu] optuna $study  $(stamp)"
          if [ "$DRY_RUN" = "1" ]; then
            echo "         (dry-run) CUDA_VISIBLE_DEVICES=$gpu $PY -u dual_kd_gnn/tune_optuna.py --dataset $ds --study-name $study --gnn-conv $conv --use-fingerprint --n-trials $TRIALS --seed $HPO_SEED"
          else
            CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u dual_kd_gnn/tune_optuna.py \
              --dataset "$ds" --study-name "$study" \
              --gnn-conv "$conv" --use-fingerprint \
              --n-trials "$TRIALS" --seed "$HPO_SEED" --device cuda \
              > "logs/${TAG}_optuna_${ds}_${conv}.log" 2>&1 \
              || echo "[gpu$gpu] FAILED $study (continuing)"
          fi
        fi
      fi
      i=$(( i + 1 ))
    done
    echo "[gpu$gpu] HPO worker done  $(stamp)"
  }

  for slot in $(seq 0 $(( NGPU - 1 ))); do hpo_worker "$slot" & done
  wait
  echo "════ HPO finished: $(ls dual_kd_gnn/optuna/*_${TAG}/best_config.json 2>/dev/null | wc -l) / $(( n_ds * n_conv )) best_config.json"
fi

# ── (2) Ablation sweep, sharded across the cards ─────────────────────────────
# --shard I/N deals the job list round-robin, so each card gets a comparable mix
# of small and large datasets rather than one card drawing every 40k-molecule
# set. Shards write disjoint directories, so no coordination is needed.
if has_phase 2; then
  echo; echo "════ (2/3) ablation sweep  ->  $RUNS_ROOT"
  for slot in $(seq 0 $(( NGPU - 1 ))); do
    gpu="${GPU_ARRAY[$slot]}"
    echo "  [gpu$gpu] shard $slot/$NGPU  ->  logs/${TAG}_sweep_gpu${gpu}.log"
    if [ "$DRY_RUN" = "1" ]; then continue; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u scripts/seed_expansion.py \
      --split-type "$SPLIT" \
      --datasets $DATASETS \
      --conditions $CONDITIONS \
      --seeds $SEEDS \
      --skip-random \
      --device cuda \
      --runs-root "$RUNS_ROOT" \
      --config-template "$CONFIG_TEMPLATE" \
      --max-batch-size "$MAX_BATCH" \
      --shard "$slot/$NGPU" \
      > "logs/${TAG}_sweep_gpu${gpu}.log" 2>&1 &
  done
  wait
  echo "════ sweep finished: $(find "$RUNS_ROOT" -name metrics.json 2>/dev/null | wc -l) runs on disk"
fi

# ── (3) Aggregate, test, plot ────────────────────────────────────────────────
if has_phase 3; then
  echo; echo "════ (3/3) results and figures"
  run "summary"   "${TAG}_summary.log"  "$PY" -u ablation/main.py --runs-dir "$RUNS_ROOT"

  for tt in classification regression; do
    run "CI $tt"       "${TAG}_ci_${tt}.log"       "$PY" -u scripts/compute_ci.py \
        --runs-root "$RUNS_ROOT" --split-type "$SPLIT" --task-type "$tt" --seeds $SEEDS
    run "wilcoxon $tt" "${TAG}_wilcoxon_${tt}.log" "$PY" -u scripts/revision_experiments.py \
        --task c1 --split-type "$SPLIT" --task-type "$tt" \
        --runs-root "$RUNS_ROOT" --seeds $SEEDS
    run "figures $tt"  "${TAG}_figures_${tt}.log"  "$PY" -u scripts/make_figures.py \
        --split-type "$SPLIT" --task-type "$tt" \
        --results-tree "$RESULTS_TREE" --seeds $SEEDS --allow-incomplete
  done

  # Curves, block norms, prototype routing, rank and paired-delta plots, read
  # from the run tree rather than the summary CSVs.
  run "paper figures" "${TAG}_paper_figures.log" "$PY" -u scripts/paper_figures.py \
      --runs-root "$RUNS_ROOT" --split-type "$SPLIT" --task-type all \
      --seeds $SEEDS --allow-incomplete

  echo
  echo "════ artifacts"
  echo "  summary CSV : ablation/ablation_summary_{classification,regression}_${RESULTS_TREE#runs_}.csv"
  echo "  CI / tests  : results/artifacts/revision/"
  echo "  figures     : results/artifacts/figures/  ($(ls results/artifacts/figures/*.png 2>/dev/null | wc -l) PNG)"
fi

echo; echo "══════ done  $(stamp)"
