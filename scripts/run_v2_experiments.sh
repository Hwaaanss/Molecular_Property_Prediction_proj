#!/usr/bin/env bash
# 재구성된 아키텍처(transformer 제거, GNN 사전학습 -> codebook head 미세조정)의
# 실험 3세트를 GPU 1번 한 장에서 순차 실행하고, 세트가 끝날 때마다 결과를 집계한다.
#
#   bash scripts/run_v2_experiments.sh                 # fast -> optuna -> standard
#   STAGES="fast" bash scripts/run_v2_experiments.sh   # 한 세트만
#   DRY_RUN=1 bash scripts/run_v2_experiments.sh       # 실행 계획만 출력
#
# 세트
#   fast     : bbbp bace sider x {gcn,gine,gat} x {fp on, fp off} x 5 seed = 90 run
#              현재 세팅(데이터셋별 기존 best_config) 그대로.
#   optuna   : bbbp bace 의 같은 6조건마다 5 trial 씩 탐색 -> 각 셀의 best_config 로
#              5 seed 재현. 튜닝 결과는 fast 와 섞이지 않도록 별도 트리에 쓴다.
#   standard : 전 데이터셋 12종. ablation 16조건은 GCN 으로만, encoder 비교는
#              full model 로만(gine/gat). 가장 오래 걸리므로 마지막.
#
# 결과 트리는 pre-refactor 결과(ablation/runs)와 분리되어 있다. 아키텍처가
# 달라서 같은 표에 넣으면 안 된다.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs

# ── 하드웨어 예산 ────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DIKAT_CPU_BUDGET="${DIKAT_CPU_BUDGET:-8}"
export DIKAT_LOADER_WORKERS="${DIKAT_LOADER_WORKERS:-4}"

# Whatever python is on PATH -- i.e. the activated environment. Overridable
# with PY=... for a specific interpreter.
PY="${PY:-$(command -v python)}"

SEEDS="${SEEDS:-0 1 2 3 42}"
SPLIT="${SPLIT:-scaffold}"
TRIALS="${TRIALS:-5}"
STAGES="${STAGES:-fast optuna standard}"
DRY_RUN="${DRY_RUN:-0}"

FAST_DATASETS="${FAST_DATASETS:-bbbp bace sider}"
OPTUNA_DATASETS="${OPTUNA_DATASETS:-bbbp bace}"
# 작은 것부터. 중간에 멈춰도 값싼 셀은 이미 끝나 있게 한다.
STD_DATASETS="${STD_DATASETS:-freesolv esol bace bbbp sider clintox lipo tox21 malaria toxcast hiv cep}"
# 6 셀 = encoder 3종 x fingerprint on/off. seed_expansion 의 'fast' 그룹과 같다.
CELLS="${CELLS:-gcn_full_model gcn_no_fp gine_full_model gine_no_fp gat_full_model gat_no_fp}"

RUNS_FAST="${RUNS_FAST:-ablation/runs_v2}"
RUNS_OPTUNA="${RUNS_OPTUNA:-ablation/runs_v2_optuna}"
RUNS_STD="${RUNS_STD:-ablation/runs_v2}"
# {dataset}, {condition} 은 seed_expansion 이 셀마다 채운다.
CONFIG_TEMPLATE='dual_kd_gnn/optuna/{dataset}_{condition}_v2/best_config.json'

FAILED=""
has_stage () { [[ " $STAGES " == *" $1 "* ]]; }
stamp () { date '+%F %T'; }

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

# 셀 이름 -> tune_optuna.py 플래그
conv_of ()  { echo "${1%%_*}"; }
fpflag_of () { case "$1" in *_no_fp) echo "--no-fingerprint";; *) echo "--use-fingerprint";; esac; }

echo "════════════════════════════════════════════════════════════"
echo " v2 실험 시작 $(stamp)"
echo "   GPU     : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "   seeds   : $SEEDS"
echo "   split   : $SPLIT"
echo "   stages  : $STAGES"
echo "════════════════════════════════════════════════════════════"

# ── 0) 피처 캐시 (fingerprint 포함) ──────────────────────────────────────────
# 이미 만들어져 있으면 즉시 통과한다. 없으면 모든 run 이 매번 다시 featurize 해서
# HIV/CEP 에서 수십 시간이 날아간다.
run "0 feature cache" v2_features.log \
  $PY -u scripts/precompute_features.py --datasets all --workers 7

# ── (1) 빠른 실험 ────────────────────────────────────────────────────────────
if has_stage fast; then
  echo; echo "════════ (1) 빠른 실험: $FAST_DATASETS x 6 cell x $(echo $SEEDS | wc -w) seed"
  run "1 fast sweep" v2_fast.log \
    $PY -u scripts/seed_expansion.py --split-type "$SPLIT" \
      --datasets $FAST_DATASETS --conditions $CELLS --seeds $SEEDS \
      --skip-random --device cuda --runs-root "$RUNS_FAST"
  run "1 fast 집계" v2_fast_summary.log $PY -u ablation/main.py --runs-dir "$RUNS_FAST"
fi

# ── (2) Optuna 실험 ──────────────────────────────────────────────────────────
if has_stage optuna; then
  echo; echo "════════ (2) Optuna: $OPTUNA_DATASETS x 6 cell x $TRIALS trials"
  for ds in $OPTUNA_DATASETS; do
    for cell in $CELLS; do
      run "2 optuna $ds/$cell" "v2_optuna_${ds}_${cell}.log" \
        $PY -u dual_kd_gnn/tune_optuna.py \
          --dataset "$ds" --study-name "${ds}_${cell}_v2" --n-trials "$TRIALS" \
          --gnn-conv "$(conv_of $cell)" "$(fpflag_of $cell)" --device cuda
    done
  done
  echo; echo "════════ (2b) Optuna best config 로 $(echo $SEEDS | wc -w) seed 재현"
  run "2 optuna replay sweep" v2_optuna_sweep.log \
    $PY -u scripts/seed_expansion.py --split-type "$SPLIT" \
      --datasets $OPTUNA_DATASETS --conditions $CELLS --seeds $SEEDS \
      --skip-random --device cuda --runs-root "$RUNS_OPTUNA" \
      --config-template "$CONFIG_TEMPLATE"
  run "2 optuna 집계" v2_optuna_summary.log $PY -u ablation/main.py --runs-dir "$RUNS_OPTUNA"
fi

# ── (3) 정석 실험 ────────────────────────────────────────────────────────────
if has_stage standard; then
  echo; echo "════════ (3) 정석: 전 데이터셋 x (GCN ablation 16 + gine/gat full model)"
  # ablation 은 GCN 만, encoder 비교는 full model 로만.
  run "3 standard gcn ablation" v2_std_gcn.log \
    $PY -u scripts/seed_expansion.py --split-type "$SPLIT" \
      --datasets $STD_DATASETS --conditions gcn_all --seeds $SEEDS \
      --skip-random --device cuda --runs-root "$RUNS_STD"
  run "3 standard gine/gat full" v2_std_encoders.log \
    $PY -u scripts/seed_expansion.py --split-type "$SPLIT" \
      --datasets $STD_DATASETS --conditions gine_full_model gat_full_model --seeds $SEEDS \
      --skip-random --device cuda --runs-root "$RUNS_STD"
  echo; echo "════════ (3b) 정석 결과 집계 + 통계 + figure"
  run "3 집계"     v2_std_summary.log  $PY -u ablation/main.py --runs-dir "$RUNS_STD"
  run "3 CI"       v2_std_ci.log       $PY -u scripts/compute_ci.py \
                                          --split-type "$SPLIT" --runs-root "$RUNS_STD" \
                                          --seeds $SEEDS
  run "3 Wilcoxon" v2_std_wilcoxon.log $PY -u scripts/revision_experiments.py \
                                          --task c1 --split-type "$SPLIT" \
                                          --runs-root "$RUNS_STD"
  run "3 figures"  v2_std_figures.log  $PY -u scripts/make_figures.py \
                                          --split-type "$SPLIT" --seeds $SEEDS \
                                          --results-tree "$(basename $RUNS_STD)" \
                                          --allow-incomplete
fi

echo
echo "════════ 종료 $(stamp)"
echo "  완료 run 수: $(find $RUNS_FAST $RUNS_OPTUNA -name metrics.json 2>/dev/null | wc -l)"
if [ -n "$FAILED" ]; then echo "  실패한 단계:$FAILED"; else echo "  전부 성공"; fi
echo "  결과 CSV : ablation/ablation_summary_{classification,regression}_v2.csv"
echo "             ablation/ablation_summary_{classification,regression}_v2_optuna.csv"
