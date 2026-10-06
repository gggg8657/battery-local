#!/usr/bin/env python3
"""battery local — PyBaMM 기반 리튬이온 배터리 성능·수명(열화) 예측. 웹 서버는 표준 라이브러리만, 계산은 PyBaMM 이 설치된
Python(기본 conda 환경 pybamm-inv)을 서브프로세스로 띄운다(worker.py). 외부 전송 없음.

  python3 app.py                                  # http://localhost:8785
  BATTERY_PY=/path/to/env/bin/python python3 app.py
  python3 app.py --cli job.json                   # 작업 하나를 돌려 결과 요약을 표준 출력으로

작업 큐: 긴 시뮬레이션은 백그라운드(동시 BATTERY_JOBS 개, 기본 2), 진행률·단계 결과(partial)를 화면에서 본다.
로컬 LLM(선택): 자연어 질문 → 실험 설정 JSON → 계산 → 계산 결과 수치만 근거로 해설(지어낸 숫자는 경고).
"""
import datetime
import glob
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WS = os.environ.get("WORKSPACE") or os.path.join(ROOT, "_workspace")  # 포털이 AGENT_DATA/<도구> 로 모아 줌
PORT = int(os.environ.get("PORT", "8785"))
HOST = os.environ.get("HOST", "0.0.0.0")
LLM_API = os.environ.get("LLM_API", "ollama")
LLM_BASE = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1" if LLM_API == "openai" else "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
LLM_KEY = os.environ.get("LLM_API_KEY", "")
NUM_CTX = int(os.environ.get("NUM_CTX", "16384"))
MAX_JOBS = max(1, int(os.environ.get("BATTERY_JOBS", "2")))           # 동시 계산 수
THREADS = os.environ.get("BATTERY_THREADS", "2")                       # 계산 하나당 BLAS/OpenMP 스레드
TIMEOUT = int(os.environ.get("BATTERY_TIMEOUT", "7200"))               # 작업 하나 최대 초
MAX_CYCLES = int(os.environ.get("BATTERY_MAX_CYCLES", "5000"))
WORKER = os.path.join(ROOT, "worker.py")
DISCLAIMER = ("물리 모델(PyBaMM) 기반 시뮬레이션입니다. 파라미터 세트의 셀과 실제 셀이 다르면 정량값(용량·온도·수명 사이클 수)은 달라집니다 — "
              "실측 데이터로 보정(파라미터 추정)이 필요합니다. 경향(조건 간 비교)을 보는 용도로 쓰세요.")


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def find_python():
    """PyBaMM 이 import 되는 Python: BATTERY_PY → conda 환경 pybamm-inv/pybamm → 현재 python3"""
    home = os.path.expanduser("~")
    cands = [os.environ.get("BATTERY_PY")]
    for base in ("miniforge3", "miniconda3", "anaconda3", "mambaforge"):
        for env in ("pybamm-inv", "pybamm", "battery"):
            cands.append(os.path.join(home, base, "envs", env, "bin", "python"))
    cands += [shutil.which("python3")]
    for c in cands:
        if c and os.path.exists(c):
            try:
                r = subprocess.run([c, "-I", "-c", "import pybamm;print(pybamm.__version__)"], capture_output=True, text=True,
                                   timeout=120, cwd=ROOT)
                if r.returncode == 0:
                    return c, r.stdout.strip()
            except Exception:
                pass
    return None, None


PY, PYBAMM_VER = ((os.environ.get("BATTERY_PY"), os.environ.get("BATTERY_PYBAMM_VER", "?")) if os.environ.get("BATTERY_SKIP_PROBE")
               else find_python())  # setup.sh 가 이미 확인했으면 다시 띄워 보지 않는다


def worker_env():
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = THREADS
    env["MPLBACKEND"] = "Agg"
    env["BATTERY_MAX_CYCLES"] = str(MAX_CYCLES)
    env.pop("PYTHONPATH", None)
    return env


def _nice():
    try:
        os.setsid()
        os.nice(10)  # 다른 사람 작업을 방해하지 않게 낮은 우선순위
    except Exception:
        pass


# ── 메타(설치된 PyBaMM 의 세트·지원 메커니즘) — 워커로 한 번 구해 캐시 ──────────────────────
_META = {"data": None, "err": None}
_META_LOCK = threading.Lock()


def meta(block=True):
    with _META_LOCK:
        if _META["data"] or not block:
            return _META["data"]
        cache = os.path.join(WS, "cache", "meta.json")
        try:
            d = json.loads(read(cache))
            if d.get("_py") == PY and d.get("_worker_mtime") == os.path.getmtime(WORKER):
                _META["data"] = d
                return d
        except Exception:
            pass
        if not PY:
            _META["err"] = "PyBaMM 이 설치된 Python 을 찾지 못했습니다 (BATTERY_PY=/경로/bin/python)"
            return None
        r = subprocess.run([PY, "-I", WORKER, "meta"], capture_output=True, text=True, timeout=600, env=worker_env(), cwd=ROOT,
                           preexec_fn=_nice)
        if r.returncode != 0:
            _META["err"] = (r.stdout + r.stderr)[-500:]
            return None
        d = json.loads(r.stdout.strip().splitlines()[-1])
        d.update(_py=PY, _worker_mtime=os.path.getmtime(WORKER))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
        _META["data"] = d
        return d


def set_info(name):
    for s in (meta() or {}).get("sets", []):
        if s["name"] == name:
            return s
    return None


# ── 작업 설정 검증 ──────────────────────────────────────────────────────
def _num(v, lo, hi, name, default=None):
    if v is None or v == "":
        if default is None:
            raise ValueError(f"{name} 값이 필요합니다")
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: 숫자가 아닙니다 ({v})")
    if not lo <= x <= hi:
        raise ValueError(f"{name}: {lo}~{hi} 범위여야 합니다 (입력 {x:g})")
    return x


STEP_RE = re.compile(r"^(Discharge|Charge|Rest|Hold)\b[\w\s./:<>%-]*$", re.I)


def validate(job):
    """화면·LLM 이 만든 작업 설정을 검사하고 기본값을 채운다. 틀리면 ValueError(한국어)."""
    kind = job.get("kind")
    if kind not in ("rate", "charge", "drive", "life"):
        raise ValueError("kind 는 rate|charge|drive|life")
    s = set_info(job.get("set") or "")
    if not s:
        raise ValueError(f"파라미터 세트 '{job.get('set')}' 가 설치된 PyBaMM 에 없습니다")
    out = {"kind": kind, "set": s["name"], "model": job.get("model") or ("SPM" if kind == "life" else "SPMe"),
           "temp_C": _num(job.get("temp_C"), -20, 60, "온도", 25.0)}
    if out["model"] not in ("SPM", "SPMe", "DFN"):
        raise ValueError("모델은 SPM|SPMe|DFN")
    if kind != "life":
        out["thermal"] = "lumped" if job.get("thermal", "lumped") == "lumped" and s["caps"]["thermal"] else "isothermal"
        if job.get("h") not in (None, ""):
            out["h"] = _num(job["h"], 0.1, 1000, "열전달계수")
    if kind in ("rate", "charge"):
        rates = job.get("c_rates") or []
        if isinstance(rates, str):
            rates = [x for x in re.split(r"[,\s]+", rates) if x]
        rates = [_num(r, 0.02, 10, "C-rate") for r in rates][:6]
        if not rates:
            raise ValueError("C-rate 를 하나 이상 넣으세요")
        out["c_rates"] = rates
    if kind == "charge":
        out["cv_cut_C"] = _num(job.get("cv_cut_C"), 0.005, 0.5, "CV 종료 전류", 0.05)
        out["soc0"] = _num(job.get("soc0"), 0, 90, "시작 SOC", 0.0)
    if kind == "drive":
        prof = job.get("profile") or []
        if not 2 <= len(prof) <= 50000:
            raise ValueError("전류 프로파일은 2~50000 점")
        out["profile"] = [[float(a), float(b)] for a, b in prof]
        out["unit"] = "C" if job.get("unit") == "C" else "A"
        out["soc0"] = _num(job.get("soc0"), 5, 100, "시작 SOC", 100.0)
        out["repeat"] = int(_num(job.get("repeat"), 1, 200, "반복 횟수", 1))
    if kind == "life":
        mechs = [m for m in (job.get("mechanisms") or ["sei"]) if m in ("sei", "plating", "lam")]
        bad = [m for m in mechs if not s["caps"].get(m)]
        if bad:
            raise ValueError(f"{s['name']} 세트는 {', '.join(bad)} 열화 파라미터가 없습니다 (지원: "
                             f"{', '.join(k for k in ('sei', 'plating', 'lam') if s['caps'].get(k)) or '없음'})")
        if not mechs:
            raise ValueError("열화 메커니즘을 하나 이상 고르세요")
        if "lam" in mechs and "sei" not in mechs:
            mechs.insert(0, "sei")
        out["mechanisms"] = mechs
        sm = job.get("sei_model") or "solvent-diffusion limited"
        if sm not in s["sei_models"]:
            raise ValueError(f"SEI 모델 '{sm}' 은 이 세트에서 쓸 수 없습니다")
        out["sei_model"] = sm
        out["sei_arrhenius"] = bool(job.get("sei_arrhenius", True))
        out["sei_scale"] = _num(job.get("sei_scale"), 0.01, 1000, "SEI 속도 배율", 1.0)
        out["max_cycles"] = int(_num(job.get("max_cycles"), 1, MAX_CYCLES, "최대 사이클", 500))
        out["eol"] = _num(job.get("eol"), 50, 99, "EOL 기준 SOH", 80.0)
        scs = job.get("scenarios") or [{}]
        if not 1 <= len(scs) <= 4:
            raise ValueError("시나리오는 1~4개")
        out["scenarios"] = []
        for i, sc in enumerate(scs):
            o = {"label": str(sc.get("label") or f"시나리오 {i + 1}")[:40],
                 "temp_C": _num(sc.get("temp_C", out["temp_C"]), -20, 60, f"시나리오 {i + 1} 온도")}
            if sc.get("steps"):
                steps = [str(x).strip() for x in sc["steps"] if str(x).strip()][:12]
                for st in steps:
                    if len(st) > 120 or not STEP_RE.match(st):
                        raise ValueError(f"실험 단계 형식 오류: '{st[:60]}' (예: Discharge at 1C until 2.5V)")
                o["steps"] = steps
            else:
                o["dis_C"] = _num(sc.get("dis_C"), 0.05, 5, f"시나리오 {i + 1} 방전 C-rate", 1.0)
                o["chg_C"] = _num(sc.get("chg_C"), 0.05, 5, f"시나리오 {i + 1} 충전 C-rate", 0.5)
                o["dod"] = _num(sc.get("dod"), 5, 100, f"시나리오 {i + 1} DoD", 100.0)
                o["cv_cut_C"] = _num(sc.get("cv_cut_C"), 0, 0.5, "CV 종료 전류", 0.05)
                o["rest_min"] = _num(sc.get("rest_min"), 0, 600, "휴지 시간", 5.0)
                if sc.get("vmax") not in (None, ""):
                    o["vmax"] = _num(sc["vmax"], s["vmin"], s["vmax"], f"시나리오 {i + 1} 충전 상한 전압")
            out["scenarios"].append(o)
    return out


def steps_preview(job):
    """화면 미리보기용: 시나리오별 한 사이클 PyBaMM 실험 문자열 (worker.cycle_steps 와 같은 규칙)"""
    s = set_info(job["set"])
    res = []
    for sc in job.get("scenarios") or []:
        if sc.get("steps"):
            res.append(sc["steps"])
            continue
        d, c, dod = sc["dis_C"], sc["chg_C"], sc["dod"]
        vmax = sc.get("vmax") or s["vmax"]
        st = [f"Discharge at {d:g}C until {s['vmin']:g}V" if dod >= 100 else
              f"Discharge at {d:g}C for {60 * dod / 100 / d:.4g} minutes or until {s['vmin']:g}V"]
        if sc["rest_min"] > 0:
            st.append(f"Rest for {sc['rest_min']:g} minutes")
        st.append(f"Charge at {c:g}C until {vmax:g}V")
        if sc["cv_cut_C"] > 0:
            st.append(f"Hold at {vmax:g}V until C/{round(1 / sc['cv_cut_C'])}")
        if sc["rest_min"] > 0:
            st.append(f"Rest for {sc['rest_min']:g} minutes")
        res.append(st)
    return res


# ── 작업 큐 ─────────────────────────────────────────────────────────────
JOBS_LOCK = threading.Lock()
QUEUE = []           # 대기 중인 id
RUNNING = {}         # id → Popen
WAKE = threading.Condition(JOBS_LOCK)


def job_dir(jid):
    if not re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", jid or ""):
        raise ValueError("잘못된 작업 id")
    return os.path.join(WS, "jobs", jid)


def write_json(p, d):
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(p + ".tmp", p)


def status_of(jid):
    return json.loads(read(os.path.join(job_dir(jid), "status.json")))


def set_status(jid, **kw):
    p = os.path.join(job_dir(jid), "status.json")
    with JOBS_LOCK:
        try:
            st = json.loads(read(p))
        except Exception:
            st = {}
        st.update(kw)
        write_json(p, st)
    return st


def title_of(job):
    s = job["set"]
    if job["kind"] == "rate":
        return f"방전 C-rate 비교 {', '.join(f'{c:g}C' for c in job['c_rates'])} · {s} · {job['model']} · {job['temp_C']:g}°C"
    if job["kind"] == "charge":
        return f"CC-CV 충전 {', '.join(f'{c:g}C' for c in job['c_rates'])} · {s} · {job['model']} · {job['temp_C']:g}°C"
    if job["kind"] == "drive":
        return f"전류 프로파일 {len(job['profile'])}점 · {s} · {job['model']} · {job['temp_C']:g}°C"
    return f"수명 {len(job['scenarios'])}개 시나리오 · {s} · {job['model']} · ≤{job['max_cycles']} 사이클"


def submit(job, origin="form", question=""):
    job = validate(job)
    now = datetime.datetime.now()
    jid = f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    d = job_dir(jid)
    os.makedirs(d)
    write_json(os.path.join(d, "job.json"), job)
    st = {"id": jid, "kind": job["kind"], "title": title_of(job), "state": "queued", "pct": 0, "msg": "대기 중",
          "created": now.isoformat(timespec="seconds"), "origin": origin, "question": question[:500]}
    write_json(os.path.join(d, "status.json"), st)
    with WAKE:
        QUEUE.append(jid)
        WAKE.notify()
    return jid


def run_job(jid):
    d = job_dir(jid)
    t0 = time.time()
    set_status(jid, state="running", started=datetime.datetime.now().isoformat(timespec="seconds"), msg="시작")
    log = open(os.path.join(d, "worker.log"), "w", encoding="utf-8")
    try:
        p = subprocess.Popen([PY, "-I", WORKER, "run", os.path.join(d, "job.json"), d], stdout=subprocess.PIPE,
                             stderr=log, text=True, env=worker_env(), cwd=ROOT, preexec_fn=_nice)
    except Exception as e:
        set_status(jid, state="error", msg=f"워커 실행 실패: {e}")
        return
    with JOBS_LOCK:
        RUNNING[jid] = p
    killer = threading.Timer(TIMEOUT, lambda: _kill(p))
    killer.start()
    err = None
    try:
        for line in p.stdout:
            if not line.startswith("@@ "):
                log.write(line)
                continue
            try:
                ev = json.loads(line[3:])
            except ValueError:
                continue
            if ev.get("stage") == "error":
                err = ev.get("msg")
            kw = {"msg": ev.get("msg", ""), "elapsed": round(time.time() - t0, 1)}
            if ev.get("pct") is not None:
                kw["pct"] = round(ev["pct"], 1)
            if ev.get("stage") == "partial":
                kw["partial"] = True
            set_status(jid, **kw)
        p.wait()
    finally:
        killer.cancel()
        log.close()
        with JOBS_LOCK:
            RUNNING.pop(jid, None)
    st = status_of(jid)
    el = round(time.time() - t0, 1)
    if st.get("state") == "canceled":
        return
    if p.returncode == 0 and os.path.exists(os.path.join(d, "result.json")):
        set_status(jid, state="done", pct=100, msg=f"완료 ({el:g} 초)", elapsed=el)
    elif time.time() - t0 >= TIMEOUT - 1:
        set_status(jid, state="error", msg=f"시간 초과({TIMEOUT} 초) — 사이클 수를 줄이거나 SPM 을 쓰세요", elapsed=el)
    else:
        tail = read(os.path.join(d, "worker.log"))[-400:]
        set_status(jid, state="error", msg=err or f"계산 실패 (exit {p.returncode}) {tail}", elapsed=el)


def _kill(p):
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def scheduler():
    while True:
        with WAKE:
            while not QUEUE or len(RUNNING) >= MAX_JOBS:
                WAKE.wait(1.0)
            jid = QUEUE.pop(0)
            RUNNING[jid] = None  # 자리 예약
        threading.Thread(target=_run_and_wake, args=(jid,), daemon=True).start()


def _run_and_wake(jid):
    try:
        run_job(jid)
    except Exception as e:
        set_status(jid, state="error", msg=f"{type(e).__name__}: {e}")
    finally:
        with WAKE:
            RUNNING.pop(jid, None)
            WAKE.notify()


def cancel(jid):
    with WAKE:
        if jid in QUEUE:
            QUEUE.remove(jid)
        p = RUNNING.get(jid)
    set_status(jid, state="canceled", msg="취소됨")
    if p:
        _kill(p)


def delete(jid):
    st = status_of(jid)
    if st.get("state") in ("queued", "running"):
        cancel(jid)
    shutil.rmtree(job_dir(jid), ignore_errors=True)


def list_jobs(limit=300):
    out = []
    for d in sorted(glob.glob(os.path.join(WS, "jobs", "*")), reverse=True)[:limit]:
        try:
            st = json.loads(read(os.path.join(d, "status.json")))
            out.append({k: st.get(k) for k in ("id", "kind", "title", "state", "pct", "msg", "created", "elapsed", "origin", "question")})
        except Exception:
            pass
    return out


def job_full(jid):
    d = job_dir(jid)
    r = {"status": status_of(jid), "job": json.loads(read(os.path.join(d, "job.json")))}
    for name in ("result", "partial", "explain"):
        p = os.path.join(d, name + ".json")
        if os.path.exists(p) and not (name == "partial" and os.path.exists(os.path.join(d, "result.json"))):
            try:
                r[name] = json.loads(read(p))
            except ValueError:
                pass
    if r["job"]["kind"] == "life":
        r["steps"] = steps_preview(r["job"])
    r["disclaimer"] = DISCLAIMER
    return r


def recover():
    """서버 재시작 전 대기·실행 중이던 작업은 중단 표시"""
    for d in glob.glob(os.path.join(WS, "jobs", "*")):
        try:
            st = json.loads(read(os.path.join(d, "status.json")))
            if st.get("state") in ("queued", "running"):
                st.update(state="error", msg="서버 재시작으로 중단됨 — 다시 실행하세요")
                write_json(os.path.join(d, "status.json"), st)
        except Exception:
            pass


# ── LLM ────────────────────────────────────────────────────────────────
def _clean(out):
    out = re.sub(r"<think>.*?</think>", "", out or "", flags=re.S).strip()
    out = re.sub(r"^```\w*\s*\n", "", out)
    return re.sub(r"\n?```\s*$", "", out).strip()


def llm(system, user, model=None, temperature=0.2, on_token=lambda t: None, json_mode=False):
    """스트리밍 채팅. Ollama /api/chat 또는 OpenAI 호환(vLLM 등)."""
    model = model or MODEL
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    buf = []
    try:
        if LLM_API == "openai":
            body = {"model": model, "stream": True, "temperature": temperature, "messages": msgs}
            hdr = {"Content-Type": "application/json", **({"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})}
            req = urllib.request.Request(LLM_BASE + "/chat/completions", json.dumps(body).encode(), hdr)
            with urllib.request.urlopen(req, timeout=1800) as r:
                for line in r:
                    line = line.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    tok = (json.loads(line[5:])["choices"][0].get("delta") or {}).get("content") or ""
                    if tok:
                        buf.append(tok)
                        on_token(tok)
            return "".join(buf)
        body = {"model": model, "stream": True, "think": False, "messages": msgs,
                "options": {"temperature": temperature, "num_ctx": NUM_CTX}}
        if json_mode:
            body["format"] = "json"
        for attempt in (0, 1):
            try:
                req = urllib.request.Request(LLM_BASE + "/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=1800) as r:
                    for line in r:
                        if not line.strip():
                            continue
                        j = json.loads(line)
                        if "error" in j:
                            raise RuntimeError(j["error"])
                        tok = j.get("message", {}).get("content", "")
                        if tok:
                            buf.append(tok)
                            on_token(tok)
                        if j.get("done"):
                            break
                return "".join(buf)
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                if attempt == 0 and "think" in msg:
                    body.pop("think")
                    continue
                raise RuntimeError(f"LLM HTTP {e.code}: {msg[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM 서버 연결 실패 ({LLM_BASE}): {e.reason}")


def models():
    try:
        if LLM_API == "openai":
            req = urllib.request.Request(LLM_BASE + "/models", headers={"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})
            with urllib.request.urlopen(req, timeout=3) as r:
                return [m["id"] for m in json.load(r)["data"]]
        with urllib.request.urlopen(LLM_BASE + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except Exception:
        return []


def plan_system():
    m = meta() or {"sets": []}
    sets = "\n".join(f"- {s['name']}: {s['chem']}, {s['cell']}, 공칭 {s['capacity']:g} Ah, 전압 {s['vmin']:g}~{s['vmax']:g} V, "
                     f"열화 지원: {', '.join(k for k in ('sei', 'plating', 'lam') if s['caps'].get(k)) or '없음(성능 전용)'}"
                     for s in m["sets"])
    return f"""너는 PyBaMM 배터리 시뮬레이션 설정 도우미다. 사용자의 자연어 질문을 아래 JSON 작업 설정 하나로 바꾼다. 계산·답변은 하지 않는다.
사용 가능한 파라미터 세트(이 이름만):
{sets}

JSON 형식(키 이름 그대로, 설명·주석·마크다운 없이 JSON 객체 하나만 출력):
{{"kind": "life" | "rate" | "charge",
 "set": 세트 이름, "model": "SPM" | "SPMe" | "DFN", "temp_C": 숫자,
 "c_rates": [숫자...]            # rate(방전 비교)·charge(충전 비교)일 때만, 최대 6개
 "mechanisms": ["sei","plating","lam"]  # life 일 때. 세트가 지원하는 것만
 "max_cycles": 정수, "eol": 80,   # life
 "scenarios": [{{"label": "짧은 한국어 이름", "temp_C": 숫자, "chg_C": 충전 C-rate, "dis_C": 방전 C-rate, "dod": 방전심도 %}}]  # life, 1~4개
 "reason": "이 설정을 고른 이유 한두 문장(한국어)"}}

규칙:
- 수명·열화·사이클·SOH·급속충전이 수명에 주는 영향 → kind "life". 방전 성능·용량·전압 곡선 → "rate". 충전 시간 → "charge".
- life 기본: set "OKane2022"(SEI·도금·LAM 결합 열화를 모두 지원), model "SPM", mechanisms 세트가 지원하는 전부, max_cycles 1000, eol 80.
- "얼마나 줄어/늘어", "비교" 질문이면 반드시 기준 시나리오(25°C, 충전 0.5C, 방전 1C, DoD 100)를 첫 번째로 넣고, 질문 조건을 두 번째 이후로 넣는다.
- 질문에 없는 조건은 기본값(25°C, 충전 0.5C, 방전 1C, DoD 100). 온도는 -20~60°C, C-rate 0.05~5.
- 사용자가 특정 화학계(LFP, NCA, LCO 등)를 말하면 그 화학계 세트를 고른다. 단 그 세트가 열화를 지원하지 않으면 life 대신 그 사실을 reason 에 적고 OKane2022 로 대체한다.
- rate/charge 기본: set "Chen2020", model "SPMe", temp_C 25."""


def parse_json(raw):
    raw = _clean(raw)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("모델 응답에서 JSON 을 찾지 못했습니다")
    return json.loads(m.group(0))


def plan(question, model=None, emit=lambda ev: None):
    if not question.strip():
        raise ValueError("질문을 적어 주세요")
    emit({"stage": "llm", "msg": f"{model or MODEL} · 질문 → 실험 설정"})
    raw = llm(plan_system(), question.strip()[:2000], model, 0.1, on_token=lambda t: emit({"token": t}), json_mode=True)
    try:
        spec = parse_json(raw)
    except Exception:
        emit({"stage": "llm", "msg": "형식 재시도", "reset": True})
        raw = llm(plan_system(), question.strip()[:2000] + "\n\nJSON 객체 하나만 출력한다.", model, 0.0,
                  on_token=lambda t: emit({"token": t}), json_mode=True)
        spec = parse_json(raw)
    reason = str(spec.pop("reason", ""))[:400]
    job = validate(spec)
    return job, reason


def facts_of(res, job):
    """해설에 넘길 계산 결과 수치 요약 (LLM 은 이 숫자만 쓸 수 있다). 비교 비율도 여기서 미리 계산."""
    f = {"세트": res.get("set"), "모델": res.get("model"), "PyBaMM": res.get("pybamm")}
    if res["kind"] == "life":
        rows = []
        base = None
        for sc in res["scenarios"]:
            e, fi = sc.get("eol") or {}, sc.get("final") or {}
            eol = e.get("simulated") or e.get("extrapolated")
            r = {"시나리오": sc["label"], "온도_C": sc["temp_C"], "한 사이클": " → ".join(sc["steps"]),
                 "계산한 사이클 수": sc["cycles_done"], "마지막 SOH_%": round(fi.get("soh") or 0, 2),
                 "100사이클당 용량감소_%p": fi.get("fade_per_100"),
                 f"EOL(SOH {e.get('threshold_pct', 80):g}%) 사이클": round(eol) if eol else None,
                 "EOL(사이클링 전류 방전 용량 기준) 사이클": round(e["simulated_dis"]) if e.get("simulated_dis") else None,
                 "EOL 구분": "시뮬레이션에서 도달" if e.get("simulated") else (f"외삽(계산 범위의 {e.get('extrap_ratio')}배)" if eol else "구할 수 없음"),
                 "리튬 손실 LLI_%": round(fi.get("LLI_pct") or 0, 3), "음극 활물질 손실 LAM_%": round(fi.get("LAM_ne_pct") or 0, 3),
                 "양극 활물질 손실 LAM_%": round(fi.get("LAM_pe_pct") or 0, 3),
                 "SEI 로 잃은 용량_Ah": fi.get("Q_SEI_Ah"), "리튬 도금으로 잃은 용량_Ah": fi.get("Q_plating_Ah"),
                 "균열면 SEI 로 잃은 용량_Ah": fi.get("Q_SEI_cracks_Ah"), "주된 열화": fi.get("dominant")}
            if base is None:
                base = eol
            elif base and eol:
                r["EOL_기준 시나리오 대비_%"] = round(100 * (eol - base) / base, 1)
            if sc.get("stop_reason"):
                r["중단 사유"] = sc["stop_reason"]
            rows.append({k: v for k, v in r.items() if v is not None})
        f["시나리오"] = rows
        f["열화 메커니즘"] = res.get("mechanisms")
    else:
        f["결과"] = [r["metrics"] for r in res["runs"]]
        f["온도_C"] = res.get("temp_C")
        if res.get("notes"):
            f["참고"] = res["notes"]
    return f


EXPLAIN_SYS = """너는 배터리 전기화학 연구자에게 PyBaMM 시뮬레이션 결과를 한국어로 해설한다.
절대 규칙:
- 숫자는 [계산 결과]에 있는 값만 그대로 쓴다. 새 숫자를 계산·추정·반올림 변경하지 않는다. 결과에 없는 사이클 수·퍼센트·온도를 만들지 않는다.
- '외삽'으로 표시된 EOL 은 반드시 "외삽"이라고 밝히고 불확실하다고 적는다.
- 메커니즘 설명(SEI 성장, 리튬 도금, 입자 균열에 따른 활물질 손실)은 일반 원리 수준으로만, 결과 수치와 연결해 설명한다.
- 결과가 직관과 다르면(예: 고온이 더 오래 감) 그대로 보고하고, 결과 수치(예: 도금 손실 vs SEI 손실)로 이유를 설명한다.
- 4~8문장 또는 짧은 목록. 마크다운 제목(#)·굵게(**) 쓰지 않는다. 마지막 면책 문구는 프로그램이 붙이니 쓰지 않는다."""

NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:\.\d+)?")


def _nums(text):
    return {x.replace(",", "") for x in NUM_RE.findall(text or "")}


def check_numbers(text, facts_json, question=""):
    """해설 속 숫자가 계산 결과(또는 질문)에 있는지. 반올림(유효숫자 줄임)은 허용."""
    allowed = _nums(facts_json) | _nums(question)
    vals = []
    for a in allowed:
        try:
            vals.append(float(a))
        except ValueError:
            pass
    bad = []
    for n in _nums(text):
        try:
            x = float(n)
        except ValueError:
            continue
        if n in allowed or x in (0, 1, 2, 3, 4) or (abs(x) < 10 and float(x).is_integer() and len(n) == 1):
            continue
        dec = len(n.split(".")[1]) if "." in n else 0
        if any(abs(round(v, dec) - x) < 1e-9 or (dec == 0 and abs(v - x) <= 0.5) for v in vals):
            continue
        bad.append(n)
    return sorted(set(bad), key=lambda s: -len(s))


def explain(jid, model=None, emit=lambda ev: None):
    d = job_dir(jid)
    res = json.loads(read(os.path.join(d, "result.json")))
    job = json.loads(read(os.path.join(d, "job.json")))
    st = status_of(jid)
    facts = facts_of(res, job)
    fj = json.dumps(facts, ensure_ascii=False, indent=1)
    q = st.get("question") or ""
    user = (f"[사용자 질문]\n{q}\n\n" if q else "") + f"[계산 결과]\n{fj}\n\n위 결과를 해설한다."
    emit({"stage": "llm", "msg": f"{model or MODEL} · 결과 해설"})
    raw = _clean(llm(EXPLAIN_SYS, user, model, 0.2, on_token=lambda t: emit({"token": t})))
    raw = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
    bad = check_numbers(raw, fj, q)
    out = {"text": raw, "unverified_numbers": bad, "facts": facts, "disclaimer": DISCLAIMER, "model": model or MODEL,
           "ts": datetime.datetime.now().isoformat(timespec="seconds")}
    write_json(os.path.join(d, "explain.json"), out)
    return out


# ── HTTP ───────────────────────────────────────────────────────────────
HTML = read(os.path.join(ROOT, "ui.html")) if os.path.exists(os.path.join(ROOT, "ui.html")) else "ui.html 없음"

# ── 저작권 표기 (LICENSE·NOTICE 참고) ─────────────────────────────────────
_SIG = __import__("base64").b64decode("wqkgMjAyNiBnZ2dnODY1NyDCtyBkb25nanVraW0uZGV2QGdtYWlsLmNvbQ==").decode()
_SIG_A = __import__("base64").b64decode("Z2dnZzg2NTcgPGRvbmdqdWtpbS5kZXZAZ21haWwuY29tPg==").decode()


def signed(html):
    """화면에 저작권 표기를 붙인다. ui.html 에서 지워져도 서버가 내보낼 때 다시 붙는다."""
    name, mail = _SIG.split(" · ")
    if 'name="author"' not in html:
        meta_ = f'<meta name="author" content="{name[7:]} <{mail}>">'
        html = html.replace("<head>", "<head>" + meta_, 1) if "<head>" in html else meta_ + html
    if "data-sig" not in html:
        tag = (f'<!-- {_SIG} --><div data-sig title="{mail}" style="text-align:center;font-size:11px;color:#9aa0a6;'
               f'opacity:.55;margin:28px 0 8px">{name}</div>')
        html = html.replace("</body>", tag + "</body>", 1) if "</body>" in html else html + tag
    return html


FILE_TYPES = {".csv": "text/csv; charset=utf-8", ".png": "image/png", ".json": "application/json"}


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        if "/api/jobs" in (a[0] if a else "") and "POST" in (a[0] if a else ""):
            super().log_message(fmt, *a)

    def _send(self, body, ctype="application/json", code=200, extra=None):
        b = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("X-Author", _SIG_A)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/health":
                return self._send({"ok": True, "python": PY, "pybamm": PYBAMM_VER})
            if path == "/api/meta":
                m = meta()
                if not m:
                    return self._send({"error": _META["err"] or "메타 정보를 못 구함"}, code=500)
                return self._send({**{k: v for k, v in m.items() if not k.startswith("_")}, "llm_model": MODEL,
                                   "max_cycles": MAX_CYCLES, "max_jobs": MAX_JOBS, "disclaimer": DISCLAIMER})
            if path == "/api/models":
                return self._send(models())
            if path == "/api/jobs":
                with JOBS_LOCK:
                    q = list(QUEUE)
                return self._send({"jobs": list_jobs(), "queue": q})
            m = re.fullmatch(r"/api/jobs/([\w-]+)", path)
            if m:
                return self._send(job_full(m.group(1)))
            m = re.fullmatch(r"/api/jobs/([\w-]+)/file/([\w.-]+)", path)
            if m:
                name = m.group(2)
                ext = os.path.splitext(name)[1]
                if ext not in FILE_TYPES or name.startswith("."):
                    raise ValueError("없는 파일")
                with open(os.path.join(job_dir(m.group(1)), name), "rb") as f:
                    data = f.read()
                return self._send(data, FILE_TYPES[ext], extra={"Content-Disposition": f'attachment; filename="battery-{m.group(1)}-{name}"'})
            if path.startswith("/api/"):
                raise ValueError("없는 경로")
            self._send(signed(HTML).encode(), "text/html; charset=utf-8")
        except (FileNotFoundError, ValueError):
            self._send({"error": "없음"}, code=404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, code=500)

    def _sse(self, fn):
        self.send_response(200)
        self.send_header("X-Author", _SIG_A)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(ev):
            self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()
        try:
            emit({"done": fn(emit)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try:
                emit({"error": str(e) if isinstance(e, (ValueError, RuntimeError)) else f"{type(e).__name__}: {e}"})
            except OSError:
                pass

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        except ValueError:
            return self._send({"error": "잘못된 요청"}, code=400)
        try:
            if path == "/api/jobs":
                jid = submit(req.get("job") or req)
                return self._send({"id": jid})
            if path == "/api/validate":
                job = validate(req.get("job") or req)
                return self._send({"job": job, "steps": steps_preview(job) if job["kind"] == "life" else None})
            m = re.fullmatch(r"/api/jobs/([\w-]+)/(cancel|delete)", path)
            if m:
                (cancel if m.group(2) == "cancel" else delete)(m.group(1))
                return self._send({"ok": True})
            if path == "/api/ask":  # 자연어 → 설정 (run=true 면 바로 큐에)
                q = (req.get("question") or "").strip()

                def go(emit):
                    job, reason = plan(q, req.get("model"), emit)
                    out = {"job": job, "reason": reason, "steps": steps_preview(job) if job["kind"] == "life" else None,
                           "title": title_of(job)}
                    if req.get("run", True):
                        out["id"] = submit(job, origin="ask", question=q)
                    return out
                return self._sse(go)
            m = re.fullmatch(r"/api/jobs/([\w-]+)/explain", path)
            if m:
                return self._sse(lambda emit: explain(m.group(1), req.get("model"), emit))
            return self._send({"error": "없는 경로"}, code=404)
        except ValueError as e:
            return self._send({"error": str(e)}, code=400)
        except FileNotFoundError:
            return self._send({"error": "없음"}, code=404)
        except Exception as e:
            return self._send({"error": f"{type(e).__name__}: {e}"}, code=500)


def start_background():
    os.makedirs(os.path.join(WS, "jobs"), exist_ok=True)
    recover()
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=meta, daemon=True).start()  # 첫 화면 전에 세트 정보 미리 구함


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--cli":
        start_background()
        jid = submit(json.loads(read(sys.argv[2])), origin="cli")
        while status_of(jid)["state"] in ("queued", "running"):
            time.sleep(1)
            print("\r" + status_of(jid).get("msg", "")[:100].ljust(100), end="", file=sys.stderr, flush=True)
        r = job_full(jid)
        print("\n" + json.dumps({"status": r["status"], "facts": facts_of(r["result"], r["job"]) if "result" in r else None},
                                ensure_ascii=False, indent=1))
        sys.exit(0 if r["status"]["state"] == "done" else 1)
    start_background()
    print(f"battery local → http://localhost:{PORT}  (python={PY} pybamm={PYBAMM_VER} jobs={MAX_JOBS}×{THREADS}thr "
          f"llm={LLM_API} {LLM_BASE} {MODEL}, workspace={WS})  {_SIG}", flush=True)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
