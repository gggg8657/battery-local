#!/usr/bin/env bash
# battery local — 원샷 설치·실행 (Linux / macOS)
#   bash setup.sh          # Python 확인 + PyBaMM 환경 탐색 + LLM 서버 탐색(선택) + selftest + 웹 서버 + 브라우저
#   bash setup.sh stop
# 환경변수: BATTERY_PY (PyBaMM 이 설치된 python 경로; 없으면 ~/miniforge3·miniconda3·anaconda3 의 envs/pybamm-inv|pybamm 자동 탐색)
#           BATTERY_JOBS (동시 계산, 2) BATTERY_THREADS (계산당 스레드, 2) BATTERY_TIMEOUT (초, 7200) BATTERY_MAX_CYCLES (5000)
#           LLM_BASE_URL / LLM_API / LLM_MODEL (AI 질문·해설용, 없어도 계산은 동작), PORT (8785), WORKSPACE (작업·결과 보관)
# 웹 서버는 표준 라이브러리만 쓴다. PyBaMM 환경은 복사·수정하지 않고 그 python 을 서브프로세스로 호출한다(pip 설치 없음).
# selftest 는 WORKSPACE 를 임시 폴더로 바꿔 실제 PyBaMM 짧은 계산(수십 초)을 돌리고 지운다. 코드가 바뀌었을 때만 다시 돈다.
set -euo pipefail
PORT="${PORT:-8785}"
if [ -t 1 ]; then B=$'\033[1m'; D=$'\033[2m'; C=$'\033[36m'; G=$'\033[32m'; R=$'\033[31m'; Y=$'\033[33m'; N=$'\033[0m'; else B= D= C= G= R= Y= N=; fi
STEP=0; step() { STEP=$((STEP+1)); printf '  %s[%d/7]%s %s%-14s%s ' "$D" "$STEP" "$N" "$B" "$1" "$N"; }
ok() { printf '%s✔%s %s\n' "$G" "$N" "${1:-}"; }; warn() { printf '%s!%s %s\n' "$Y" "$N" "${1:-}"; }; skip() { printf '%s–%s %s\n' "$D" "$N" "${1:-}"; }
die() { printf '%s✘ %s%s\n\n' "$R" "$*" "$N" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }; probe() { curl -fsS -m 2 "$1" >/dev/null 2>&1; }
wait_for() { for _ in $(seq 1 "${2:-30}"); do probe "$1" && return 0; sleep 1; done; return 1; }
printf '\n%s  battery local%s  PyBaMM 배터리 성능·수명 예측 — 로컬 계산, 외부 전송 없음\n\n' "$B" "$N"

step "OS 감지"; case "$(uname -s)" in Darwin*) OS=mac ;; Linux*) OS=linux ;; *) die "지원하지 않는 OS: $(uname -s)" ;; esac; ok "$OS ($(uname -m))"
step "패키지 확보"; cd "$(dirname "${BASH_SOURCE[0]}")"; [ -f app.py ] && [ -f worker.py ] || die "app.py/worker.py 가 없습니다"; ok "$(pwd)"
if [ "${1:-}" = "stop" ]; then
  if [ -f .server.pid ]; then pkill -TERM -f "battery-local/worker.py run" 2>/dev/null || true; kill "$(cat .server.pid)" 2>/dev/null && ok "웹 서버 종료" || skip "이미 종료됨"; rm -f .server.pid
  else skip "실행 중인 서버 없음"; fi; exit 0; fi

step "Python"
PY=""; for c in python3 python; do has "$c" && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null && { PY=$c; break; }; done
[ -n "$PY" ] || die "Python 3.9+ 가 없습니다"
ok "$($PY --version 2>&1) (웹 서버, 표준 라이브러리만)"

step "PyBaMM 환경"
BPY="${BATTERY_PY:-}"
if [ -z "$BPY" ]; then for base in miniforge3 miniconda3 anaconda3 mambaforge; do for env in pybamm-inv pybamm battery; do
  c="$HOME/$base/envs/$env/bin/python"; [ -x "$c" ] && "$c" -I -c 'import pybamm' 2>/dev/null && { BPY=$c; break 2; }; done; done; fi
[ -z "$BPY" ] && "$PY" -I -c 'import pybamm' 2>/dev/null && BPY=$(command -v "$PY")
[ -n "$BPY" ] || die "PyBaMM 이 설치된 Python 을 못 찾았습니다 → BATTERY_PY=/경로/envs/<환경>/bin/python bash setup.sh (설치: pip install pybamm)"
VER=$("$BPY" -I -c 'import pybamm; print(pybamm.__version__)' 2>/dev/null)
"$BPY" -I -c 'import pybamm; pybamm.IDAKLUSolver()' >/dev/null 2>&1 || die "PyBaMM IDAKLU 솔버를 불러오지 못했습니다 ($BPY)"
"$BPY" -I -c 'import matplotlib' >/dev/null 2>&1 && MPL="matplotlib 있음(PNG)" || MPL="matplotlib 없음(PNG 생략, 화면 SVG 는 됨)"
export BATTERY_PY="$BPY" BATTERY_SKIP_PROBE=1 BATTERY_PYBAMM_VER="$VER"
ok "PyBaMM $VER · $BPY · $MPL"

step "LLM 서버"
export LLM_API="${LLM_API:-}" LLM_BASE_URL="${LLM_BASE_URL:-}" LLM_MODEL="${LLM_MODEL:-}"
if [ -n "$LLM_BASE_URL" ]; then [ -n "$LLM_API" ] || { case "$LLM_BASE_URL" in *1143*) LLM_API=ollama ;; *) LLM_API=openai ;; esac; }
elif probe http://localhost:11434/api/tags; then LLM_API=ollama LLM_BASE_URL=http://localhost:11434
else for p in 8000 1234 8080; do probe "http://localhost:$p/v1/models" && { LLM_API=openai LLM_BASE_URL="http://localhost:$p/v1"; break; }; done; fi
if [ -n "$LLM_BASE_URL" ] && [ -z "$LLM_MODEL" ]; then
  if [ "$LLM_API" = ollama ]; then LLM_MODEL=$(curl -fsS -m 3 "$LLM_BASE_URL/api/tags" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["models"][0]["name"])' 2>/dev/null || true)
  else LLM_MODEL=$(curl -fsS -m 3 "$LLM_BASE_URL/models" | "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true); fi
fi
if [ -n "$LLM_BASE_URL" ]; then ok "$LLM_API $LLM_BASE_URL ${LLM_MODEL:-}"
else warn "찾지 못함 → 계산·그래프는 되고 'AI 질문·해설'만 안 됩니다"; LLM_API=ollama; fi

step "자가검증"
STAMP=$(cat app.py worker.py selftest.py | cksum | cut -d' ' -f1)-$VER
if [ "$(cat .selftest.ok 2>/dev/null)" = "$STAMP" ]; then skip "코드 변경 없음 — 이전 통과"
else env -u WORKSPACE "$PY" selftest.py >/dev/null 2>&1 || die "selftest 실패 (python3 selftest.py 로 확인)"; echo "$STAMP" > .selftest.ok
  ok "C-rate·충전·프로파일·수명(SEI+도금+LAM, 60사이클)·외삽·취소·해설 숫자 검사 (임시 폴더)"; fi

step "웹 서버"
[ -f .server.pid ] && kill "$(cat .server.pid)" 2>/dev/null || true
LLM_API=$LLM_API LLM_BASE_URL=$LLM_BASE_URL LLM_MODEL=$LLM_MODEL PORT=$PORT nohup "$PY" app.py > server.log 2>&1 & echo $! > .server.pid
wait_for "http://localhost:$PORT/api/health" 30 || { cat server.log; die "웹 서버 기동 실패 (server.log 확인)"; }
URL="http://localhost:$PORT"; ok "$URL"
case "$OS" in mac) open "$URL" ;; linux) [ -n "${WORKSPACE:-}" ] || { has xdg-open && xdg-open "$URL" >/dev/null 2>&1 || true; } ;; esac
printf '\n  %s준비 완료%s  %s%s%s   종료: bash setup.sh stop   로그: server.log\n\n' "$B" "$N" "$C" "$URL" "$N"
