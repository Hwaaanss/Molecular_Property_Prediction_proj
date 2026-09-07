#!/usr/bin/env bash
# GCN 단일 encoder, fingerprint on/off 2셀만 비교하는 optuna 세트.
#
# 전부 환경변수로 조절한다. TAG 는 study 이름 접미사이자 결과 트리/로그 구분자다.
#
#   (1) 데이터셋 x 2셀 마다 optuna 5 trial  -> best_config.json
#   (2) 그 best_config 로 5 seed 재현        -> ablation/runs_v3_optuna
#   (3) 결과 집계                            -> ablation/ablation_summary_*_v3_optuna.csv
#
# study 이름은 <dataset>_<cell>_v3. 접미사를 v2 에서 바꾼 이유는 optuna 가
# load_if_exists=True 라서, 같은 이름을 쓰면 symmetric head 시절 trial 60개를
# 그대로 이어받아 낡은 이력으로 TPE 가 샘플링하기 때문이다.
#
#   TAG=v4 TRIALS=10 DATASETS="freesolv esol bace bbbp sider clintox" \
#       bash scripts/run_optuna_set.sh
#   DRY_RUN=1 bash scripts/run_optuna_set.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DIKAT_CPU_BUDGET="${DIKAT_CPU_BUDGET:-8}"
export DIKAT_LOADER_WORKERS="${DIKAT_LOADER_WORKERS:-4}"

# Whatever python is on PATH -- i.e. the activated environment. Overridable
# with PY=... for a specific interpreter.
PY="${PY:-$(command -v python)}"
DATASETS="${DATASETS:-freesolv bace bbbp esol sider}"
CELLS="${CELLS:-gcn_full_model gcn_no_fp}"
SEEDS="${SEEDS:-0 1 2 3 42}"
TRIALS="${TRIALS:-5}"
SPLIT="${SPLIT:-scaffold}"
TAG="${TAG:-v3}"
RUNS_ROOT="${RUNS_ROOT:-ablation/runs_${TAG}_optuna}"
LOGPFX="${LOGPFX:-${TAG}opt}"
ETA="${ETA:-?}"
CONFIG_TEMPLATE="dual_kd_gnn/optuna/{dataset}_{condition}_${TAG}/best_config.json"
DRY_RUN="${DRY_RUN:-0}"

FAILED=""
stamp () { date '+%F %T'; }
conv_of ()   { echo "${1%%_*}"; }
fpflag_of () { case "$1" in *_no_fp) echo "--no-fingerprint";; *) echo "--use-fingerprint";; esac; }

run () {                        # run <이름> <로그파일> <커맨드...>
  local name="$1" log="$2"; shift 2
  echo "──────── $name  |  $(stamp)"
  echo "    \$ $*"
  if [ "$DRY_RUN" = "1" ]; then echo "    (dry-run)"; return 0; fi
  if "$@" 2>&1 | tee "logs/$log"; then
    echo "    ✓ $name  ($(stamp))"
  else
    echo "    ✗ $name — 계속 진행"; FAILED="$FAILED
      $name"
  fi
}

n_ds=$(echo $DATASETS | wc -w); n_cell=$(echo $CELLS | wc -w)
echo "════════════════════════════════════════════════════════════"
echo " v3 optuna 세트 시작 $(stamp)"
echo "   GPU       : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   datasets  : $DATASETS  ($n_ds 종)"
echo "   cells     : $CELLS  ($n_cell 셀)"
echo "   study     : $((n_ds*n_cell)) 개 x $TRIALS trial"
echo "   재현      : $((n_ds*n_cell*$(echo $SEEDS | wc -w))) run  (seeds $SEEDS)"
echo "   결과 트리 : $RUNS_ROOT"
echo "   로그 접두 : logs/${LOGPFX}_*"
echo "   예상      : 약 $ETA 시간"
echo "════════════════════════════════════════════════════════════"

# ── (1) optuna ───────────────────────────────────────────────────────────────
echo; echo "════════ (1/3) optuna  $((n_ds*n_cell)) study x $TRIALS trial"
for ds in $DATASETS; do
  for cell in $CELLS; do
    run "1 optuna $ds/$cell" "${LOGPFX}_optuna_${ds}_${cell}.log" \
      $PY -u dual_kd_gnn/tune_optuna.py \
        --dataset "$ds" --study-name "${ds}_${cell}_${TAG}" --n-trials "$TRIALS" \
        --gnn-conv "$(conv_of $cell)" "$(fpflag_of $cell)" --device cuda
  done
done

# ── (2) best_config 재현 ─────────────────────────────────────────────────────
echo; echo "════════ (2/3) best_config 로 $(echo $SEEDS | wc -w) seed 재현"
run "2 replay sweep" v3opt_sweep.log \
  $PY -u scripts/seed_expansion.py --split-type "$SPLIT" \
    --datasets $DATASETS --conditions $CELLS --seeds $SEEDS \
    --skip-random --device cuda --runs-root "$RUNS_ROOT" \
    --config-template "$CONFIG_TEMPLATE"

# ── (3) 집계 ─────────────────────────────────────────────────────────────────
echo; echo "════════ (3/3) 결과 집계"
run "3 집계" v3opt_summary.log $PY -u ablation/main.py --runs-dir "$RUNS_ROOT"

echo
echo "════════ 종료 $(stamp)"
echo "  완료 run : $(find "$RUNS_ROOT" -name metrics.json 2>/dev/null | wc -l)"
echo "  best_config : $(ls -d dual_kd_gnn/optuna/*_${TAG} 2>/dev/null | wc -l) study 중 $(ls dual_kd_gnn/optuna/*_${TAG}/best_config.json 2>/dev/null | wc -l) 개 생성"
if [ -n "$FAILED" ]; then echo "  실패한 단계:$FAILED"; else echo "  전부 성공"; fi
echo "  결과 CSV : ablation/ablation_summary_{classification,regression}_${TAG}_optuna.csv"
