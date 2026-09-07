#!/usr/bin/env bash
# DiKAT 전체 파이프라인 — GPU 0번 1장, CPU 8코어, RAM 100GB 예산 안에서 동작.
#
#   bash scripts/run_all.sh                      # 전체 (1)~(6)
#   PHASES="1 2 3" bash scripts/run_all.sh       # classification 만
#   SEEDS="0 1 2" bash scripts/run_all.sh        # seed 줄여서 빠르게
#   DRY_RUN=1 bash scripts/run_all.sh            # 실행 계획만 출력
#
# 흐름
#   (1) classification best_config 없는 데이터셋 → Optuna 10 trials
#   (2) classification 전 데이터셋 ablation (a1/a2/a4 각 조합 + fingerprint 계열 a8)
#   (3) classification 결과 CSV 검증(표준편차 포함) + figure 생성
#   (4) regression best_config 없는 데이터셋 → Optuna 10 trials
#   (5) regression 전 데이터셋 ablation
#   (6) regression 결과 CSV 검증 + figure 생성
#
# 각 단계는 산출물이 이미 있으면 건너뛴다. 중단 후 재실행해도 안전하다.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p logs

# ── 하드웨어 예산 (관리자 자동 kill 방지) ────────────────────────────────────
# GPU 1번만. torch 는 기본적으로 호스트의 128코어를 보고 스레드풀을 잡으므로
# common/resources.py 가 DIKAT_CPU_BUDGET 에 맞춰 전부 고정한다.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export DIKAT_CPU_BUDGET="${DIKAT_CPU_BUDGET:-8}"
export DIKAT_LOADER_WORKERS="${DIKAT_LOADER_WORKERS:-4}"
# 부모 프로세스가 결과를 모으므로 워커는 예산-1개. 실측 피크 6.5 cores.
export DIKAT_FEATURIZE_WORKERS="${DIKAT_FEATURIZE_WORKERS:-7}"

# ── 실험 조건 ────────────────────────────────────────────────────────────────
SEEDS="${SEEDS:-0 1 2 3 42}"                                      # 5 seeds
# 조건 이름은 <conv>_full_model / <conv>_no_<factors> 형식이다. 기본값은 GCN 의
# 16조건 전부 + gine/gat 의 full model. ablation 해석은 encoder 하나로 충분하고,
# encoder 비교는 full model 로만 해도 confound 가 없다.
CONDITIONS="${CONDITIONS:-gcn_all gine_full_model gat_full_model}"
SPLIT="${SPLIT:-scaffold}"     # 표준 Bemis-Murcko scaffold split
# run 디렉터리는 내부 키를 그대로 쓴다. 이미 쌓인 결과와 진행 중인 sweep 이
# 여기에 있어서, 이름을 바꾸면 결과가 두 경로로 쪼개진다.
SPLIT_DIR="deterministic_scaffold"
# 현재 아키텍처 결과 트리. ablation/runs 는 transformer 단계가 있던 이전
# 아키텍처 결과라 같은 표에 넣으면 안 된다.
RUNS_ROOT="${RUNS_ROOT:-ablation/runs_v2}"
RESULTS_TREE="$(basename "$RUNS_ROOT")"
TRIALS="${TRIALS:-10}"
PHASES="${PHASES:-1 2 3 4 5 6}"
DRY_RUN="${DRY_RUN:-0}"

# 작은 데이터셋 → 큰 데이터셋 순서 (분자 수 기준)
CLS_DATASETS="sider clintox bace bbbp tox21 toxcast hiv"
REG_DATASETS="freesolv esol lipo malaria cep"

FAILED=""
has_phase () { [[ " $PHASES " == *" $1 "* ]]; }

run () {                        # run <이름> <로그파일> <커맨드...>
  local name="$1" log="$2"; shift 2
  echo "──────── $name  |  $(date '+%F %T')"
  echo "    \$ $*"
  if [ "$DRY_RUN" = "1" ]; then echo "    (dry-run)"; return 0; fi
  if "$@" 2>&1 | tee "logs/$log"; then
    echo "    ✓ $name"
  else
    echo "    ✗ $name — 계속 진행"; FAILED="$FAILED
      $name"
  fi
}

# best_config 가 없는 데이터셋만 Optuna 를 돌린다
tune_missing () {
  local phase="$1"; shift
  for ds in "$@"; do
    local cfg="dual_kd_gnn/optuna/${ds}_xkd/best_config.json"
    if [ -f "$cfg" ]; then
      echo "──────── [$phase] $ds : best_config 있음 → Optuna 건너뜀 ($cfg)"
      continue
    fi
    run "[$phase] optuna $ds ($TRIALS trials)" "optuna_${ds}.log" \
      python -u dual_kd_gnn/tune_optuna.py \
        --dataset "$ds" --study-name "${ds}_xkd" --n-trials "$TRIALS" --device cuda
  done
}

echo "════════════════════════════════════════════════════════════"
echo " 시작 $(date)"
echo "   GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (1장)"
echo "   CPU 예산   : $DIKAT_CPU_BUDGET cores (loader $DIKAT_LOADER_WORKERS)"
echo "   split      : $SPLIT"
echo "   seeds      : ($(echo $SEEDS | wc -w)) $SEEDS"
echo "   conditions : ($(echo $CONDITIONS | wc -w))"
echo "   optuna     : $TRIALS trials (best_config 없는 데이터셋만)"
echo "   phases     : $PHASES"
echo "════════════════════════════════════════════════════════════"

# ── 0) 피처 캐시 ─────────────────────────────────────────────────────────────
# MMFF 3D 임베딩을 데이터셋당 한 번만 계산해 재사용한다. 건너뛰면 모든 run 이
# 매번 다시 featurize 해서 HIV/CEP 같은 큰 데이터셋에서 수십 시간이 날아간다.
run "0/6 feature cache" p0_features.log \
  python -u scripts/precompute_features.py --datasets all --workers "$DIKAT_FEATURIZE_WORKERS"

# ── (1) classification Optuna ────────────────────────────────────────────────
if has_phase 1; then
  echo; echo "════════ (1) classification Optuna"
  tune_missing "1/6" $CLS_DATASETS
fi

# ── (2) classification ablation ──────────────────────────────────────────────
if has_phase 2; then
  echo; echo "════════ (2) classification ablation"
  for ds in $CLS_DATASETS; do
    run "2/6 ablation $ds" "ablation_${ds}.log" \
      python -u scripts/seed_expansion.py --split-type "$SPLIT" \
        --datasets "$ds" --conditions $CONDITIONS --seeds $SEEDS \
        --skip-random --device cuda --runs-root "$RUNS_ROOT"
  done
fi

# ── (3) classification 검증 + figure ─────────────────────────────────────────
if has_phase 3; then
  echo; echo "════════ (3) classification 결과 검증 + figure"
  run "3/6 summary"  cls_summary.log  python -u ablation/main.py --runs-dir "$RUNS_ROOT"
  # --seeds 를 줘야 셀마다 seed 수가 섞이지 않는다. 이전에 더 많은 seed 로 돌려둔
  # 데이터셋이 있으면 필터 없이는 n=5 와 n=15 가 한 표에 들어간다.
  run "3/6 CI"       cls_ci.log       python -u scripts/compute_ci.py \
                                        --split-type "$SPLIT" --task-type classification \
                                        --runs-root "$RUNS_ROOT" --seeds $SEEDS
  run "3/6 figures"  cls_figures.log  python -u scripts/make_figures.py \
                                        --split-type "$SPLIT" --task-type classification \
                                        --results-tree "$RESULTS_TREE" --seeds $SEEDS
  run "3/6 Wilcoxon" cls_wilcoxon.log python -u scripts/revision_experiments.py \
                                        --task c1 --split-type "$SPLIT" --task-type classification \
                                        --runs-root "$RUNS_ROOT"
fi

# ── (4) regression Optuna ────────────────────────────────────────────────────
if has_phase 4; then
  echo; echo "════════ (4) regression Optuna"
  tune_missing "4/6" $REG_DATASETS
fi

# ── (5) regression ablation ──────────────────────────────────────────────────
if has_phase 5; then
  echo; echo "════════ (5) regression ablation"
  for ds in $REG_DATASETS; do
    run "5/6 ablation $ds" "ablation_${ds}.log" \
      python -u scripts/seed_expansion.py --split-type "$SPLIT" \
        --datasets "$ds" --conditions $CONDITIONS --seeds $SEEDS \
        --skip-random --device cuda --runs-root "$RUNS_ROOT"
  done
fi

# ── (6) regression 검증 + figure ─────────────────────────────────────────────
if has_phase 6; then
  echo; echo "════════ (6) regression 결과 검증 + figure"
  run "6/6 summary"  reg_summary.log  python -u ablation/main.py --runs-dir "$RUNS_ROOT"
  run "6/6 CI"       reg_ci.log       python -u scripts/compute_ci.py \
                                        --split-type "$SPLIT" --task-type regression \
                                        --runs-root "$RUNS_ROOT" --seeds $SEEDS
  run "6/6 figures"  reg_figures.log  python -u scripts/make_figures.py \
                                        --split-type "$SPLIT" --task-type regression \
                                        --results-tree "$RESULTS_TREE" --seeds $SEEDS
  run "6/6 Wilcoxon" reg_wilcoxon.log python -u scripts/revision_experiments.py \
                                        --task c1 --split-type "$SPLIT" --task-type regression \
                                        --runs-root "$RUNS_ROOT"
fi

echo
echo "════════ 종료 $(date)"
echo "  완료 run 수: $(find "$RUNS_ROOT/$SPLIT_DIR" -name metrics.json 2>/dev/null | wc -l)"
if [ -n "$FAILED" ]; then echo "  실패한 단계:$FAILED"; else echo "  전부 성공"; fi
echo
echo "  결과 CSV : ablation/ablation_summary_{classification,regression}_${RESULTS_TREE#runs_}.csv"
echo "             results/artifacts/revision/*_{classification,regression}_*.csv"
echo "  Figure   : results/artifacts/figures/  (300 dpi PNG + LZW TIFF)"
echo "  로그     : logs/"
