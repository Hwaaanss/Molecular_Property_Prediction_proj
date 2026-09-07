#!/usr/bin/env bash
# 진행 중인 sweep 의 진행률 + 현재까지의 결과표를 한 번에 출력한다.
#   bash scripts/v3_status.sh                       # 현재 세트 (TAG 기본값)
#   TAG=v3 N_STUDY=10 N_RUN=50 bash scripts/v3_status.sh   # 이전 세트
#   ROOT=ablation/runs_v3 bash scripts/v3_status.sh # 기본 config sweep 결과
# 조건/데이터셋 목록은 하드코딩하지 않고 디스크에서 읽어 만든다.
cd "$(dirname "$0")/.."
TAG="${TAG:-v4}"
ROOT="${ROOT:-ablation/runs_${TAG}_optuna}"
LOGPFX="${LOGPFX:-${TAG}opt}"
N_STUDY="${N_STUDY:-12}"
N_RUN="${N_RUN:-60}"
# Whatever python is on PATH -- i.e. the activated environment. Overridable
# with PY=... for a specific interpreter.
PY="${PY:-$(command -v python)}"

echo "═══ $(date '+%F %T')  |  $ROOT ═══"
# driver 이름으로 찾으면 이 스크립트 자신(문자열 포함)까지 잡힌다. 실제 학습
# 프로세스인 python 워커를 본다.
CUR=$(pgrep -af "python .*(tune_optuna|seed_expansion)\.py" | grep -v "bin/bash -c" | head -1)
if [ -n "$CUR" ]; then
  echo "상태: 실행 중"
  echo "  지금: $(echo "$CUR" | sed -E 's/.*--(dataset|datasets) /'"'"'/; s/ --device.*//; s/--study-name /study=/')"
else
  echo "상태: 실행 중인 학습 프로세스 없음"
fi

echo
echo "── (1) optuna: study 진행 ──"
tot=$(ls -d dual_kd_gnn/optuna/*_"$TAG" 2>/dev/null | wc -l)
done_=$(ls dual_kd_gnn/optuna/*_"$TAG"/best_config.json 2>/dev/null | wc -l)
echo "  best_config 생성: $done_ / $N_STUDY (study 디렉터리 $tot 개)"
for f in dual_kd_gnn/optuna/*_"$TAG"/trials.csv; do
  [ -e "$f" ] || continue
  n=$(( $(wc -l < "$f") - 1 ))
  printf "    %-34s trial %d\n" "$(basename "$(dirname "$f")")" "$n"
done

echo
echo "── (2) 재현 sweep ──"
echo "  완료 run: $(find "$ROOT" -name metrics.json 2>/dev/null | wc -l) / $N_RUN"
grep -hE "^\[deterministic" "logs/${LOGPFX}_sweep.log" 2>/dev/null | tail -3 | sed 's/^/    /'

echo
echo "── 현재까지 결과 ──"
$PY - "$ROOT" <<'PY'
import json, glob, sys, collections, numpy as np
root=sys.argv[1]
cells=collections.defaultdict(list); met={}; order=[]
for f in sorted(glob.glob(f"{root}/**/metrics.json", recursive=True)):
    d=json.load(open(f))
    cells[(d["ablation_name"], d["dataset"])].append(d["test_metric"])
    met[d["dataset"]]=d["metric_name"]
    if d["dataset"] not in order: order.append(d["dataset"])
if not cells:
    print("  (아직 결과 없음)"); raise SystemExit
conds=sorted({k[0] for k in cells}, key=lambda c: ("no_fp" in c, c))
w=24
print("  " + " "*18 + "".join(f"{d+' ('+met[d]+')':>{w}s}" for d in order))
for c in conds:
    row=f"  {c:18s}"
    for d in order:
        v=cells.get((c,d))
        s = f"{np.mean(v):.4f}±{np.std(v,ddof=1):.4f}(n{len(v)})" if v and len(v)>1 else (f"{v[0]:.4f}(n1)" if v else "—")
        row+=f"{s:>{w}s}"
    print(row)
# fp 효과
pairs=[(c, c.replace("_full_model","_no_fp")) for c in conds if c.endswith("_full_model")]
if pairs:
    print("\n  fingerprint 효과 (full_model - no_fp):")
    for a,b in pairs:
        row=f"    {a.split('_')[0]:6s}"
        for d in order:
            va, vb = cells.get((a,d)), cells.get((b,d))
            if not va or not vb or len(va)!=len(vb): row+=f"{'—':>{w}s}"; continue
            row+=f"{f'{np.mean(va)-np.mean(vb):+.4f}':>{w}s}"
        print(row)
PY
