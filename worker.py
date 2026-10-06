#!/usr/bin/env python3
"""battery local 계산 워커 — PyBaMM 이 설치된 Python(기본: conda 환경 pybamm-inv)으로 app.py 가 서브프로세스로 띄운다.

  python -I worker.py meta                 # 설치된 PyBaMM 버전·파라미터 세트·세트별 지원 열화 메커니즘 → JSON(stdout)
  python -I worker.py run job.json outdir  # 작업 실행 → outdir/result.json (+ CSV·PNG). 진행률은 stdout 에 '@@ {json}' 줄로

작업 종류(job["kind"]):
  rate   정전류 방전 C-rate 비교 (전압-용량·온도·SOC)
  charge CC-CV 충전 프로토콜 비교 (충전 시간·80% 도달 시간)
  drive  사용자 전류 프로파일(시간 s, 전류 A 또는 C-rate; +방전/−충전)
  life   열화 서브모델을 켠 사이클 수명 예측 — 시나리오 1~4개, 단계적 사이클(50→200→…), EOL·LLI/LAM, 외삽은 '외삽'으로 표시
"""
import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402

T0 = time.time()


def emit(**ev):
    print("@@ " + json.dumps(ev, ensure_ascii=False), flush=True)


# ── 파라미터 세트 설명 (화학계는 각 세트 OCP 함수 이름으로 확인한 것) ───────────────────────────
SETS = {
    "OKane2022": {"chem": "NMC811 / 흑연-SiOx", "cell": "LG M50 21700 원통형",
                  "note": "Chen2020 기반 + SEI·리튬 도금·입자 균열·LAM 결합 열화 파라미터(O'Kane 2022). 수명 예측 권장"},
    "Chen2020": {"chem": "NMC811 / 흑연-SiOx", "cell": "LG M50 21700 원통형",
                 "note": "성능 시뮬레이션 기본 세트. SEI 는 문헌 예시값(활성화에너지 0)"},
    "ORegan2022": {"chem": "NMC811 / 흑연-SiOx", "cell": "LG M50 21700 원통형",
                   "note": "온도 의존 물성·열 파라미터를 정밀화한 세트. 열화 파라미터 없음(성능 전용)"},
    "Mohtat2020": {"chem": "NMC532 / 흑연", "cell": "파우치 셀(미시간대)", "note": "SEI 는 문헌 예시값"},
    "Ecker2015": {"chem": "NCO(니켈코발트산화물) / 흑연", "cell": "Kokam SLPB 75106100 파우치",
                  "note": "리튬 도금 파라미터 포함. 세트 정의 공칭 용량이 실제 셀보다 작게 축소되어 있음"},
    "Marquis2019": {"chem": "LCO / 흑연", "cell": "Kokam SLPB78205130H 파우치", "note": "SPM/SPMe 비교 논문 세트. 세트 정의 공칭 용량 0.68 Ah"},
    "Ai2020": {"chem": "LCO / 흑연", "cell": "Enertech 파우치", "note": "입자 응력(기계) 파라미터 포함 → 응력 기반 LAM 가능"},
    "NCA_Kim2011": {"chem": "NCA / 흑연", "cell": "파우치 셀(Kim 2011 Nominal Design)", "note": ""},
    "Ramadass2004": {"chem": "LCO / 흑연", "cell": "18650급 (여러 문헌 조합)", "note": "PyBaMM 문서상 '조합 세트 — 주의해서 사용'. 열 모델 불가"},
    "Prada2013": {"chem": "LFP / 흑연", "cell": "A123 26650 계열 LFP", "note": "SEI·열 파라미터 없음 → 등온 성능 전용"},
}
MODELS = {"SPM": "단일입자 모델 — 가장 빠름, 저율(≤1C)에서 적합",
          "SPMe": "전해질 포함 SPM — 빠르면서 2C 안팎까지 정확도 개선 (권장)",
          "DFN": "도일-풀러-뉴먼 P2D — 가장 정확, 수~수십 배 느림"}
SEI_MODELS = ["solvent-diffusion limited", "ec reaction limited", "reaction limited",
              "interstitial-diffusion limited", "electron-migration limited"]
SEI_LABEL = {"solvent-diffusion limited": "용매 확산 지배", "ec reaction limited": "EC 반응 지배",
             "reaction limited": "반응 지배", "interstitial-diffusion limited": "격자간 확산 지배",
             "electron-migration limited": "전자 이동 지배"}
MECH_LABEL = {"sei": "SEI 성장", "plating": "리튬 도금(부분 가역)", "lam": "활물질 손실(입자 응력·균열)"}
EA_SEI_DEFAULT = 38000.0  # OKane2022 세트의 SEI growth activation energy [J/mol]


def lazy_pybamm():
    import pybamm
    pybamm.set_logging_level("ERROR")
    return pybamm


def mech_options(mechs, sei_model="solvent-diffusion limited", thermal="isothermal"):
    o = {}
    if thermal and thermal != "isothermal":
        o["thermal"] = thermal
    if "sei" in mechs:
        o.update({"SEI": sei_model, "SEI porosity change": "true"})
    if "plating" in mechs:
        o.update({"lithium plating": "partially reversible", "lithium plating porosity change": "true"})
    if "lam" in mechs:
        o.update({"particle mechanics": ("swelling and cracking", "swelling only"), "loss of active material": "stress-driven"})
        if "sei" in mechs:
            o["SEI on cracks"] = "true"
    return o


def _ok(pybamm, name, opts):
    try:
        pybamm.ParameterValues(name).process_model(pybamm.lithium_ion.SPM(opts))
        return True
    except Exception:
        return False


def meta():
    pybamm = lazy_pybamm()
    have = set(pybamm.parameter_sets.keys())
    sets = []
    for name, d in SETS.items():
        if name not in have:
            continue
        pv = pybamm.ParameterValues(name)
        caps = {"thermal": _ok(pybamm, name, {"thermal": "lumped"}),
                "sei": _ok(pybamm, name, mech_options(["sei"])),
                "plating": _ok(pybamm, name, mech_options(["plating"])) and float(pv.get("Lithium plating kinetic rate constant [m.s-1]", 0) or 0) > 0,
                "lam": _ok(pybamm, name, mech_options(["sei", "lam"]))}
        sei_models = [m for m in SEI_MODELS if caps["sei"] and _ok(pybamm, name, mech_options(["sei"], m))]
        ea = pv.get("SEI growth activation energy [J.mol-1]", None)
        sets.append({"name": name, **d, "capacity": float(pv["Nominal cell capacity [A.h]"]),
                     "vmin": float(pv["Lower voltage cut-off [V]"]), "vmax": float(pv["Upper voltage cut-off [V]"]),
                     "caps": caps, "sei_models": sei_models, "sei_ea": None if ea is None else float(ea),
                     "doc": " ".join((pybamm.parameter_sets.get_docstring(name) or "").split())[:400]})
    return {"pybamm": pybamm.__version__, "python": sys.version.split()[0], "sets": sets, "models": MODELS,
            "sei_label": SEI_LABEL, "mech_label": MECH_LABEL, "ea_default": EA_SEI_DEFAULT}


# ── 공통 ─────────────────────────────────────────────────────────────
def fnum(x, nd=4):
    return float(f"{float(x):.{nd}g}") if x is not None and np.isfinite(x) else None


def decimate(n, k=400):
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def params(pybamm, job, T=None):
    pv = pybamm.ParameterValues(job["set"])
    T = job.get("temp_C", 25) if T is None else T
    pv["Ambient temperature [K]"] = 273.15 + float(T)
    pv["Initial temperature [K]"] = 273.15 + float(T)
    if job.get("h"):
        pv["Total heat transfer coefficient [W.m-2.K-1]"] = float(job["h"])
    return pv


def model_of(pybamm, name, opts):
    if name not in MODELS:
        raise ValueError(f"모델은 {list(MODELS)} 중 하나")
    return getattr(pybamm.lithium_ion, name)(opts or None)


def solver(pybamm):
    return pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-6, options={"num_threads": 1})


def var(sol, name, alt=None):
    try:
        return np.asarray(sol[name].entries, dtype=float)
    except Exception:
        if alt:
            return var(sol, alt)
        return None


def series(sol, cap_nom, soc0, mode):
    t = var(sol, "Time [s]")
    q = var(sol, "Discharge capacity [A.h]")
    v = var(sol, "Voltage [V]", "Terminal voltage [V]")
    i = var(sol, "Current [A]")
    T = var(sol, "Volume-averaged cell temperature [C]")
    if T is None:
        T = var(sol, "Volume-averaged cell temperature [K]")
        T = None if T is None else T - 273.15
    e = var(sol, "Discharge energy [W.h]")
    idx = decimate(len(t))
    soc = soc0 - q / cap_nom
    out = {"t_min": (t[idx] - t[0]) / 60, "q_Ah": q[idx] if mode != "charge" else -q[idx], "V": v[idx], "I_A": i[idx],
           "soc": soc[idx] * 100}
    if T is not None:
        out["T_C"] = T[idx]
    out = {k: [fnum(x, 6) for x in a] for k, a in out.items()}
    stats = {"t": t - t[0], "q": q, "v": v, "i": i, "T": T, "soc": soc, "e": e}
    return out, stats


def thermal_of(job, caps):
    return "lumped" if job.get("thermal", "lumped") == "lumped" and caps.get("thermal") else "isothermal"


def caps_of(pybamm, name):
    return {"thermal": _ok(pybamm, name, {"thermal": "lumped"})}


# ── 성능: C-rate 방전 비교 ───────────────────────────────────────────────
def run_rate(job, out):
    pybamm = lazy_pybamm()
    pv0 = params(pybamm, job)
    cap, vmin = float(pv0["Nominal cell capacity [A.h]"]), float(job.get("vmin") or pv0["Lower voltage cut-off [V]"])
    therm = thermal_of(job, caps_of(pybamm, job["set"]))
    rates = [float(c) for c in job.get("c_rates") or [0.5, 1, 2]][:6]
    model = model_of(pybamm, job.get("model", "SPMe"), {"thermal": therm} if therm != "isothermal" else {})
    res = []
    for k, c in enumerate(rates):
        emit(stage="sim", msg=f"{c}C 방전", pct=100 * k / len(rates))
        pv = params(pybamm, job)
        period = max(1.0, 3600 / c / 400)
        exp = pybamm.Experiment([f"Discharge at {c}C until {vmin}V"], period=f"{period:.3g} seconds")
        t1 = time.time()
        sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp, solver=solver(pybamm)).solve(initial_soc=1)
        s, st = series(sol, cap, 1.0, "rate")
        q = st["q"][-1] - st["q"][0]
        e = float(np.trapezoid(st["v"] * st["i"], st["t"]) / 3600)
        m = {"label": f"{c}C", "c_rate": c, "current_A": fnum(c * cap), "capacity_Ah": fnum(q), "capacity_pct": fnum(100 * q / cap),
             "energy_Wh": fnum(e), "avg_V": fnum(e / q if q else None), "duration_min": fnum(st["t"][-1] / 60),
             "end_V": fnum(st["v"][-1]), "sec": round(time.time() - t1, 2)}
        if st["T"] is not None:
            m["T_max_C"] = fnum(np.max(st["T"]))
            m["T_rise_C"] = fnum(np.max(st["T"]) - st["T"][0])
        res.append({"metrics": m, "series": s})
    notes = []
    for r in res:
        m = r["metrics"]
        if m["capacity_pct"] is not None and m["capacity_pct"] < 70:
            notes.append(f"{m['label']}: 하한 전압에 일찍 도달(공칭의 {m['capacity_pct']}%) — 고율에서 확산·전해질 한계. "
                         "이 세트·모델의 고율 한계일 수 있으니 DFN 으로 확인하세요")
    notes += heat_notes(res, pv0, therm)
    return {"kind": "rate", "cap_nom": cap, "thermal": therm, "runs": res, "notes": notes}


def heat_notes(res, pv, therm):
    if therm == "isothermal":
        return ["등온 계산(셀 온도 = 주변 온도 고정)"]
    hot = [r["metrics"] for r in res if (r["metrics"].get("T_max_C") or 0) > 60]
    h = float(pv["Total heat transfer coefficient [W.m-2.K-1]"])
    out = [f"집중(lumped) 열 모델, 열전달계수 {h:g} W/m²K"]
    if hot:
        out.append(", ".join(f"{m['label']} 최고 {m['T_max_C']}°C" for m in hot) +
                   " — 냉각 조건에 크게 좌우됨(열전달계수를 실제 냉각에 맞춰 조정)")
    return out


# ── 성능: CC-CV 충전 비교 ───────────────────────────────────────────────
def run_charge(job, out):
    pybamm = lazy_pybamm()
    pv0 = params(pybamm, job)
    cap, vmax = float(pv0["Nominal cell capacity [A.h]"]), float(job.get("vmax") or pv0["Upper voltage cut-off [V]"])
    cut = float(job.get("cv_cut_C") or 0.05)
    therm = thermal_of(job, caps_of(pybamm, job["set"]))
    rates = [float(c) for c in job.get("c_rates") or [0.5, 1, 2]][:6]
    model = model_of(pybamm, job.get("model", "SPMe"), {"thermal": therm} if therm != "isothermal" else {})
    soc0 = float(job.get("soc0", 0)) / 100 if float(job.get("soc0", 0)) > 1 else float(job.get("soc0", 0))
    res = []
    for k, c in enumerate(rates):
        emit(stage="sim", msg=f"{c}C CC-CV 충전", pct=100 * k / len(rates))
        pv = params(pybamm, job)
        exp = pybamm.Experiment([f"Charge at {c}C until {vmax}V", f"Hold at {vmax}V until C/{round(1 / cut)}"],
                                period=f"{max(1.0, 3600 / c / 400):.3g} seconds")
        t1 = time.time()
        sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp, solver=solver(pybamm)).solve(initial_soc=soc0)
        s, st = series(sol, cap, soc0, "charge")
        q_in = -(st["q"] - st["q"][0])
        soc = st["soc"]
        cc_end = sol.cycles[0].steps[0]["Time [s]"].entries[-1] - st["t"][0] if sol.cycles and sol.cycles[0].steps else None

        def t_at(target):
            j = np.argmax(soc >= target)
            return fnum(st["t"][j] / 60) if soc[j] >= target else None
        e_in = float(np.trapezoid(st["v"] * -st["i"], st["t"]) / 3600)
        m = {"label": f"{c}C", "c_rate": c, "charged_Ah": fnum(q_in[-1]), "energy_in_Wh": fnum(e_in),
             "t80_min": t_at(0.8), "t_cc_min": fnum(cc_end / 60) if cc_end is not None else None,
             "t_total_min": fnum(st["t"][-1] / 60), "cc_fraction_pct": fnum(100 * cc_end / st["t"][-1]) if cc_end else None,
             "sec": round(time.time() - t1, 2)}
        if st["T"] is not None:
            m["T_max_C"] = fnum(np.max(st["T"]))
            m["T_rise_C"] = fnum(np.max(st["T"]) - st["T"][0])
        res.append({"metrics": m, "series": s})
    return {"kind": "charge", "cap_nom": cap, "thermal": therm, "cv_cut_C": cut, "runs": res, "notes": heat_notes(res, pv0, therm)}


# ── 성능: 사용자 전류 프로파일 ────────────────────────────────────────────
def run_drive(job, out):
    pybamm = lazy_pybamm()
    pv = params(pybamm, job)
    cap = float(pv["Nominal cell capacity [A.h]"])
    prof = np.asarray(job["profile"], dtype=float)
    if prof.ndim != 2 or prof.shape[1] != 2 or len(prof) < 2:
        raise ValueError("프로파일은 [시간 s, 전류] 2열이어야 합니다")
    t, i = prof[:, 0] - prof[0, 0], prof[:, 1]
    if np.any(np.diff(t) <= 0):
        raise ValueError("시간 열이 증가하지 않습니다")
    if job.get("unit") == "C":
        i = i * cap
    reps = max(1, min(int(job.get("repeat", 1)), 200))
    if reps > 1:
        dt = t[-1] + (t[-1] - t[-2])
        t = np.concatenate([t + k * dt for k in range(reps)])
        i = np.tile(i, reps)
    therm = thermal_of(job, caps_of(pybamm, job["set"]))
    model = model_of(pybamm, job.get("model", "SPMe"), {"thermal": therm} if therm != "isothermal" else {})
    soc0 = float(job.get("soc0", 100)) / 100
    vmin, vmax = float(pv["Lower voltage cut-off [V]"]), float(pv["Upper voltage cut-off [V]"])
    step = pybamm.step.current(np.column_stack([t, i]), termination=[f"< {vmin}V", f"> {vmax}V"])
    exp = pybamm.Experiment([step])
    emit(stage="sim", msg=f"프로파일 {len(t)}점 · {t[-1] / 60:.1f}분", pct=5)
    sol = pybamm.Simulation(model, parameter_values=pv, experiment=exp, solver=solver(pybamm)).solve(initial_soc=soc0)
    s, st = series(sol, cap, soc0, "drive")
    done = st["t"][-1] >= t[-1] - 1e-6
    e_out = float(np.trapezoid(st["v"] * st["i"], st["t"]) / 3600)
    m = {"label": "프로파일", "duration_min": fnum(st["t"][-1] / 60), "profile_min": fnum(t[-1] / 60), "completed": bool(done),
         "V_min": fnum(st["v"].min()), "V_max": fnum(st["v"].max()), "soc_end_pct": fnum(100 * st["soc"][-1]),
         "net_Ah": fnum(st["q"][-1] - st["q"][0]), "net_energy_Wh": fnum(e_out), "I_max_A": fnum(i.max()), "I_min_A": fnum(i.min())}
    if st["T"] is not None:
        m["T_max_C"] = fnum(np.max(st["T"]))
        m["T_rise_C"] = fnum(np.max(st["T"]) - st["T"][0])
    notes = heat_notes([{"metrics": m}], pv, therm) + ([] if done else [f"전압 한계({vmin}~{vmax} V)에 닿아 {m['duration_min']}분에서 중단 — 프로파일 끝까지 못 감"])
    return {"kind": "drive", "cap_nom": cap, "thermal": therm, "runs": [{"metrics": m, "series": s}], "notes": notes}


# ── 수명 ────────────────────────────────────────────────────────────────
def cycle_steps(sc, vmin, vmax):
    """시나리오 설정 → 한 사이클의 PyBaMM 실험 문자열 목록"""
    if sc.get("steps"):
        return [str(s).strip() for s in sc["steps"] if str(s).strip()]
    d, c = float(sc.get("dis_C", 1)), float(sc.get("chg_C", 0.5))
    dod = float(sc.get("dod", 100))
    rest = float(sc.get("rest_min", 5))
    cut = float(sc.get("cv_cut_C", 0.05))
    vmax_s = float(sc.get("vmax") or vmax)
    vmin_s = float(sc.get("vmin") or vmin)
    dis = (f"Discharge at {d:g}C until {vmin_s:g}V" if dod >= 100 else
           f"Discharge at {d:g}C for {60 * dod / 100 / d:.4g} minutes or until {vmin_s:g}V")
    st = [dis]
    if rest > 0:
        st.append(f"Rest for {rest:g} minutes")
    st.append(f"Charge at {c:g}C until {vmax_s:g}V")
    if cut > 0:
        st.append(f"Hold at {vmax_s:g}V until C/{round(1 / cut)}")
    if rest > 0:
        st.append(f"Rest for {rest:g} minutes")
    return st


def stages_of(max_cycles):
    out = [s for s in (50, 200, 500, 1000, 2000, 3000, 5000) if s < max_cycles]
    return out + [max_cycles]


def fit_extrapolate(n, soh, thr):
    """용량 손실 L(n) = a·√n + b·n (a,b ≥ 0) 최소제곱 → L = 1-thr 가 되는 n. SEI(√n)+선형 열화 가정의 '외삽'."""
    n = np.asarray(n, float)
    loss = 1 - np.asarray(soh, float)
    k = n > 0
    if k.sum() < 5:
        return None
    A = np.column_stack([np.sqrt(n[k]), n[k]])
    try:
        from scipy.optimize import nnls
        (a, b), _ = nnls(A, loss[k])
    except Exception:
        a, b = np.linalg.lstsq(A, loss[k], rcond=None)[0]
        a, b = max(a, 0), max(b, 0)
    L = 1 - thr
    if a <= 0 and b <= 0:
        return None
    x = L / a if b <= 0 else (-a + math.sqrt(a * a + 4 * b * L)) / (2 * b)
    n_eol = x * x
    resid = loss[k] - A @ np.array([a, b])
    r2 = 1 - float(np.sum(resid ** 2)) / max(float(np.sum((loss[k] - loss[k].mean()) ** 2)), 1e-30)
    return {"a": float(a), "b": float(b), "n_eol": float(n_eol), "r2": r2, "fit_to": int(n[-1])}


def eol_cross(n, soh, thr):
    for j in range(1, len(soh)):
        if soh[j] <= thr < soh[j - 1]:
            return float(n[j - 1] + (soh[j - 1] - thr) / (soh[j - 1] - soh[j]) * (n[j] - n[j - 1]))
    return None


def discharge_curve(cyc):
    try:
        s = cyc.steps[0]
        q = np.asarray(s["Discharge capacity [A.h]"].entries, float)
        v = np.asarray(s["Voltage [V]"].entries, float)
        idx = decimate(len(q), 150)
        return {"q_Ah": [fnum(x, 5) for x in q[idx] - q[0]], "V": [fnum(x, 5) for x in v[idx]]}
    except Exception:
        return None


def run_scenario(pybamm, job, sc, si, nsc, write_partial):
    T = float(sc.get("temp_C", job.get("temp_C", 25)))
    pv = params(pybamm, job, T)
    cap = float(pv["Nominal cell capacity [A.h]"])
    vmin, vmax = float(pv["Lower voltage cut-off [V]"]), float(pv["Upper voltage cut-off [V]"])
    mechs = [m for m in job.get("mechanisms") or ["sei"] if m in MECH_LABEL]
    if not mechs:
        raise ValueError("열화 메커니즘을 하나 이상 고르세요")
    sei_model = job.get("sei_model") or "solvent-diffusion limited"
    notes = []
    if "sei" in mechs and job.get("sei_arrhenius", True):
        ea = pv.get("SEI growth activation energy [J.mol-1]", None)
        if ea is not None and float(ea) == 0:
            pv["SEI growth activation energy [J.mol-1]"] = EA_SEI_DEFAULT
            notes.append(f"이 세트의 SEI 활성화에너지가 0 → 온도 영향을 보려고 {EA_SEI_DEFAULT / 1000:g} kJ/mol(OKane2022 값) 적용")
    scale = float(job.get("sei_scale", 1) or 1)
    if "sei" in mechs and scale != 1:
        for k in ("SEI solvent diffusivity [m2.s-1]", "SEI kinetic rate constant [m.s-1]", "EC diffusivity [m2.s-1]",
                  "SEI lithium interstitial diffusivity [m2.s-1]", "SEI electron conductivity [S.m-1]"):
            if k in pv.keys():
                pv[k] = pv[k] * scale
        notes.append(f"SEI 성장 속도 상수 ×{scale:g} (사용자 보정)")
    model = model_of(pybamm, job.get("model", "SPM"), mech_options(mechs, sei_model))
    steps = cycle_steps(sc, vmin, vmax)
    full_dis = steps[0].lower().startswith("discharge") and " for " not in steps[0].lower() and "until" in steps[0].lower()
    try:
        pybamm.Experiment([tuple(steps)])
    except Exception as e:
        raise ValueError(f"실험 문자열 오류: {e}")
    max_cycles = max(1, min(int(job.get("max_cycles", 500)), int(os.environ.get("BATTERY_MAX_CYCLES", "5000"))))
    thr = float(job.get("eol", 80)) / 100
    chunk = int(job.get("chunk", 10))
    sims = {}
    rows, curves, start, stop_reason = [], [], None, ""
    stage_list = stages_of(max_cycles)
    c0, last_cyc = None, None
    t_sc = time.time()
    label = sc.get("label") or f"시나리오 {si + 1}"
    for stage in stage_list:
        while len(rows) < stage:
            k = min(chunk, stage - len(rows))
            if k not in sims:
                sims[k] = pybamm.Simulation(model, parameter_values=pv, experiment=pybamm.Experiment([tuple(steps)] * k),
                                            solver=solver(pybamm))
            try:
                sol = sims[k].solve(starting_solution=start, initial_soc=float(job.get("soc0", 100)) / 100 if start is None else None)
            except Exception as e:
                stop_reason = f"솔버 오류로 {len(rows)} 사이클에서 중단: {str(e)[:160]}"
                break
            new = [c for c in sol.cycles[-k:] if c is not None] if start is not None else [c for c in sol.cycles if c is not None]
            sv = sol.summary_variables
            got = min(len(new), k)
            if got == 0:
                stop_reason = f"솔버가 {len(rows) + 1} 사이클을 풀지 못해 중단"
                break
            def col(name, n=got):
                try:
                    return np.asarray(sv[name], float)[-n:]
                except Exception:
                    return np.full(n, np.nan)
            cap_e = col("Capacity [A.h]")
            lli = col("Loss of lithium inventory [%]")
            lam_n = col("Loss of active material in negative electrode [%]")
            lam_p = col("Loss of active material in positive electrode [%]")
            q_sei = col("Loss of capacity to negative SEI [A.h]")
            q_pl = col("Loss of capacity to negative lithium plating [A.h]")
            q_cr = col("Loss of capacity to negative SEI on cracks [A.h]")
            for j, cyc in enumerate(new[-got:]):
                try:
                    q = cyc.steps[0]["Discharge capacity [A.h]"].entries
                    qd = float(q[-1] - q[0])
                except Exception:
                    qd = float("nan")
                if c0 is None:
                    c0 = {"e": cap_e[j], "d": qd}
                    curves.append({"cycle": 1, **(discharge_curve(cyc) or {})})
                rows.append({"cycle": len(rows) + 1, "cap_esoh_Ah": fnum(cap_e[j], 6), "cap_dis_Ah": fnum(qd, 6),
                             "soh": fnum(100 * cap_e[j] / c0["e"], 6), "soh_dis": fnum(100 * qd / c0["d"], 6) if full_dis and c0["d"] else None,
                             "LLI_pct": fnum(lli[j], 5), "LAM_ne_pct": fnum(lam_n[j], 5), "LAM_pe_pct": fnum(lam_p[j], 5),
                             "Q_SEI_Ah": fnum(q_sei[j], 5), "Q_plating_Ah": fnum(q_pl[j], 5), "Q_SEI_cracks_Ah": fnum(q_cr[j], 5)})
            start = sol.last_state
            last_cyc = new[-1]
            done_frac = (si + len(rows) / max_cycles) / nsc
            emit(stage="life", msg=f"{label}: {len(rows)}/{max_cycles} 사이클 · SOH {rows[-1]['soh']:.2f}%",
                 pct=100 * done_frac, scenario=si, cycle=len(rows), soh=rows[-1]["soh"])
            if got < k:
                stop_reason = f"솔버가 {len(rows) + 1} 사이클에서 수렴하지 못해 중단(조건이 너무 가혹하거나 전압 한계 문제)"
                break
            if rows[-1]["soh"] is not None and rows[-1]["soh"] <= 100 * thr:
                stop_reason = f"SOH {100 * thr:g}% 도달"
                break
        if stop_reason:
            break
        curves.append({"cycle": len(rows), **(discharge_curve(last_cyc) or {})})
        write_partial(si, summarize(label, sc, steps, rows, curves, thr, notes, stop_reason, T, time.time() - t_sc, stage))
    if stop_reason and rows and (not curves or curves[-1]["cycle"] != len(rows)):
        curves.append({"cycle": len(rows), **(discharge_curve(last_cyc) or {})})
    return summarize(label, sc, steps, rows, curves, thr, notes, stop_reason, T, time.time() - t_sc, len(rows))


def summarize(label, sc, steps, rows, curves, thr, notes, stop_reason, T, sec, stage):
    n = np.array([r["cycle"] for r in rows], float)
    soh = np.array([r["soh"] for r in rows], float) / 100
    res = {"label": label, "scenario": sc, "steps": steps, "temp_C": T, "rows": rows, "curves": curves, "notes": list(notes),
           "stop_reason": stop_reason, "sec": round(sec, 1), "cycles_done": len(rows), "stage": stage}
    if not rows:
        return res
    last = rows[-1]
    eol = eol_cross(n, soh, thr)
    eol_d = None
    if rows[0].get("soh_dis") is not None:
        sd = np.array([r["soh_dis"] if r["soh_dis"] is not None else np.nan for r in rows], float) / 100
        eol_d = eol_cross(n, sd, thr)
    fit = None if eol else fit_extrapolate(n - 1, soh, thr)
    res["eol"] = {"threshold_pct": 100 * thr, "simulated": fnum(eol) if eol else None, "simulated_dis": fnum(eol_d) if eol_d else None,
                  "extrapolated": fnum(fit["n_eol"] + 1, 4) if fit else None, "fit": fit}
    if fit:
        k = fit["n_eol"] / max(n[-1], 1)
        res["eol"]["extrap_ratio"] = fnum(k, 3)
        res["notes"].append(f"EOL 은 계산한 {int(n[-1])} 사이클을 a·√n + b·n 으로 맞춘 외삽(계산 범위의 {k:.3g}배)"
                            + (" — 범위를 크게 벗어나 불확실성 큼. 최대 사이클을 늘리면 정확해짐" if k > 3 else ""))
        xs = np.linspace(n[-1], min(fit["n_eol"] + 1, n[-1] * 50), 40)
        res["eol"]["fit_curve"] = [[fnum(x, 5), fnum(100 * (1 - fit["a"] * math.sqrt(x - 1) - fit["b"] * (x - 1)), 5)] for x in xs]
    lli, ln, lp = last["LLI_pct"] or 0, last["LAM_ne_pct"] or 0, last["LAM_pe_pct"] or 0
    res["final"] = {"cycle": last["cycle"], "soh": last["soh"], "soh_dis": last.get("soh_dis"), "LLI_pct": lli, "LAM_ne_pct": ln,
                    "LAM_pe_pct": lp, "Q_SEI_Ah": last["Q_SEI_Ah"], "Q_plating_Ah": last["Q_plating_Ah"],
                    "Q_SEI_cracks_Ah": last["Q_SEI_cracks_Ah"],
                    "fade_per_100": fnum(100 * (1 - soh[-1]) / max(n[-1] - 1, 1) * 100, 4) if len(rows) > 1 else None}
    dom = max([("LLI(리튬 손실)", lli), ("LAM 음극", ln), ("LAM 양극", lp)], key=lambda x: x[1])
    res["final"]["dominant"] = dom[0] if dom[1] > 0 else None
    return res


def run_life(job, out):
    pybamm = lazy_pybamm()
    scs = (job.get("scenarios") or [{}])[:4]
    nsc = len(scs)
    partial = [None] * nsc
    results = []

    def write_partial(si, r):
        partial[si] = r
        doc = {"kind": "life", "partial": True, "scenarios": [x for x in results + partial[len(results):] if x]}
        with open(os.path.join(out, "partial.json.tmp"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        os.replace(os.path.join(out, "partial.json.tmp"), os.path.join(out, "partial.json"))
        emit(stage="partial", msg=f"{r['label']}: {r['cycles_done']} 사이클 단계 결과 저장", pct=None)

    for si, sc in enumerate(scs):
        r = run_scenario(pybamm, job, sc, si, nsc, write_partial)
        results.append(r)
        partial[si] = r
    pv = params(pybamm, job)
    return {"kind": "life", "cap_nom": float(pv["Nominal cell capacity [A.h]"]), "scenarios": results,
            "mechanisms": job.get("mechanisms"), "sei_model": job.get("sei_model") or "solvent-diffusion limited",
            "soh_basis": "평형(저율) 용량 — PyBaMM eSOH 'Capacity [A.h]' (기준성능시험 RPT 의 저율 용량에 해당)"}


# ── 그림(PNG, 영문 라벨 — 서버에 한글 글꼴이 없을 수 있음) ─────────────────────────────────
def plots(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    files = []
    kind = res["kind"]
    if kind in ("rate", "charge", "drive"):
        panels = [("q_Ah", "V", "Capacity [Ah]", "Voltage [V]")] if kind == "rate" else [("t_min", "V", "Time [min]", "Voltage [V]")]
        panels += [("t_min", "T_C", "Time [min]", "Temperature [C]"), ("t_min", "soc", "Time [min]", "SOC [%]")]
        if kind != "rate":
            panels.append(("t_min", "I_A", "Time [min]", "Current [A]"))
        fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
        for ax, (x, y, xl, yl) in zip(axs, panels):
            for r in res["runs"]:
                if y in r["series"]:
                    ax.plot(r["series"][x], r["series"][y], label=r["metrics"]["label"].replace("프로파일", "profile"))
            ax.set_xlabel(xl)
            ax.set_ylabel(yl)
            ax.grid(alpha=.3)
        axs[0].legend(fontsize=8)
    else:
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
        for k, sc in enumerate(res["scenarios"]):
            lbl = f"S{k + 1}"
            n = [r["cycle"] for r in sc["rows"]]
            line, = axs[0].plot(n, [r["soh"] for r in sc["rows"]], label=lbl)
            fc = (sc.get("eol") or {}).get("fit_curve")
            if fc:
                axs[0].plot([p[0] for p in fc], [p[1] for p in fc], "--", color=line.get_color(), alpha=.6)
            axs[1].plot(n, [r["LLI_pct"] for r in sc["rows"]], color=line.get_color(), label=f"{lbl} LLI")
            axs[1].plot(n, [r["LAM_ne_pct"] for r in sc["rows"]], ":", color=line.get_color(), label=f"{lbl} LAM-ne")
            axs[1].plot(n, [r["LAM_pe_pct"] for r in sc["rows"]], "-.", color=line.get_color(), label=f"{lbl} LAM-pe")
            for c in sc["curves"]:
                if c.get("q_Ah"):
                    axs[2].plot(c["q_Ah"], c["V"], color=line.get_color(), alpha=.35 + .65 * (c["cycle"] == sc["curves"][-1]["cycle"]))
        thr = res["scenarios"][0].get("eol", {}).get("threshold_pct", 80) if res["scenarios"] else 80
        axs[0].axhline(thr, color="gray", lw=.8, ls="--")
        axs[0].set_xlabel("Cycle")
        axs[0].set_ylabel("SOH [%] (dashed: extrapolation)")
        axs[1].set_xlabel("Cycle")
        axs[1].set_ylabel("Loss [%]")
        axs[2].set_xlabel("Discharge capacity [Ah]")
        axs[2].set_ylabel("Voltage [V]")
        axs[0].legend(fontsize=8)
        axs[1].legend(fontsize=6, ncol=2)
        for ax in axs:
            ax.grid(alpha=.3)
    fig.text(0.995, 0.01, "PyBaMM physics-based simulation; quantitative values need calibration to the real cell",
             ha="right", fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = os.path.join(out, "plot.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append("plot.png")
    return files


def write_csv(res, out):
    import csv
    files = []
    if res["kind"] in ("rate", "charge", "drive"):
        with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            keys = sorted({k for r in res["runs"] for k in r["metrics"]}, key=lambda k: (k != "label", k))
            w = csv.DictWriter(f, keys)
            w.writeheader()
            for r in res["runs"]:
                w.writerow(r["metrics"])
        with open(os.path.join(out, "curves.csv"), "w", newline="", encoding="utf-8-sig") as f:
            cols = ["t_min", "q_Ah", "V", "I_A", "soc", "T_C"]
            w = csv.writer(f)
            w.writerow(["run"] + cols)
            for r in res["runs"]:
                s = r["series"]
                for j in range(len(s["t_min"])):
                    w.writerow([r["metrics"]["label"]] + [s.get(c, [None] * (j + 1))[j] if c in s else "" for c in cols])
        files += ["summary.csv", "curves.csv"]
    else:
        cols = ["cycle", "soh", "soh_dis", "cap_esoh_Ah", "cap_dis_Ah", "LLI_pct", "LAM_ne_pct", "LAM_pe_pct",
                "Q_SEI_Ah", "Q_plating_Ah", "Q_SEI_cracks_Ah"]
        with open(os.path.join(out, "cycles.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["scenario"] + cols)
            for sc in res["scenarios"]:
                for r in sc["rows"]:
                    w.writerow([sc["label"]] + [r.get(c) for c in cols])
        with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "temp_C", "cycles_done", "soh_final", "eol_simulated", "eol_extrapolated(외삽)",
                        "LLI_pct", "LAM_ne_pct", "LAM_pe_pct", "Q_SEI_Ah", "Q_plating_Ah", "stop_reason", "steps"])
            for sc in res["scenarios"]:
                fi, e = sc.get("final") or {}, sc.get("eol") or {}
                w.writerow([sc["label"], sc["temp_C"], sc["cycles_done"], fi.get("soh"), e.get("simulated"), e.get("extrapolated"),
                            fi.get("LLI_pct"), fi.get("LAM_ne_pct"), fi.get("LAM_pe_pct"), fi.get("Q_SEI_Ah"),
                            fi.get("Q_plating_Ah"), sc["stop_reason"], " | ".join(sc["steps"])])
        files += ["summary.csv", "cycles.csv"]
    return files


RUN = {"rate": run_rate, "charge": run_charge, "drive": run_drive, "life": run_life}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "meta":
        print(json.dumps(meta(), ensure_ascii=False))
        return
    if len(sys.argv) < 4 or sys.argv[1] != "run":
        sys.exit(__doc__)
    with open(sys.argv[2], encoding="utf-8") as f:
        job = json.load(f)
    out = sys.argv[3]
    os.makedirs(out, exist_ok=True)
    emit(stage="load", msg="PyBaMM 불러오는 중", pct=0)
    if job.get("kind") not in RUN:
        raise SystemExit("kind 는 rate|charge|drive|life")
    res = RUN[job["kind"]](job, out)
    res.update(set=job["set"], model=job.get("model"), temp_C=job.get("temp_C", 25), sec=round(time.time() - T0, 1),
               pybamm=lazy_pybamm().__version__)
    emit(stage="files", msg="CSV·그림 저장", pct=99)
    res["files"] = write_csv(res, out) + plots(res, out)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    emit(stage="done", msg=f"완료 {res['sec']} s", pct=100)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit(stage="error", msg=f"{type(e).__name__}: {e}")
        sys.exit(1)
