#!/usr/bin/env python3
"""자가검증 (실제 PyBaMM 짧은 계산 + 가짜 LLM): 설정 검증·실험 문자열 → 작업 큐(rate·charge·drive·life 짧게) → 결과 파싱(EOL·LLI/LAM·
외삽 표시·CSV·PNG) → 자연어 설정(가짜 LLM) → 해설 숫자 검사 → HTTP(화면·메타·파일 다운로드·저작자 표기).
WORKSPACE 는 임시 폴더로 바꿔 실데이터 폴더에 흔적을 남기지 않는다. 수십 초 걸린다.   python3 selftest.py"""
import http.server, json, os, shutil, sys, tempfile, threading, time, urllib.request

TMP = tempfile.mkdtemp(prefix="battery-selftest-")
os.environ["WORKSPACE"] = TMP
os.environ.setdefault("BATTERY_JOBS", "2")
for _k in ("PORTAL_JOBS_URL", "PORTAL_JOBS_TOKEN", "PORTAL_JOBS_TOOL"):  # 포털 작업 알림은 꺼진 채로 시작
    os.environ.pop(_k, None)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app  # noqa: E402

T0 = time.time()


def wait(jid, limit=300):
    t = time.time()
    while time.time() - t < limit:
        st = app.status_of(jid)
        if st["state"] not in ("queued", "running"):
            return st
        time.sleep(0.5)
    raise AssertionError(f"{jid} 시간 초과")


def fake_plan(system, user, model=None, temperature=0.2, on_token=lambda t: None, json_mode=False):
    if "[계산 결과]" in user:  # 해설: 결과에 있는 숫자 + 지어낸 숫자 1234.5
        out = "급속충전 시나리오의 EOL 이 더 짧습니다. 지어낸 값 1234.5 사이클."
    else:  # 설정 만들기: 코드 펜스·설명 섞어서
        out = ('설정입니다:\n```json\n{"kind":"life","set":"OKane2022","model":"SPM","mechanisms":["sei","plating"],"max_cycles":20,'
               '"scenarios":[{"label":"기준","temp_C":25,"chg_C":0.5},{"label":"45°C 2C","temp_C":45,"chg_C":2}],'
               '"reason":"기준과 비교"}\n```')
    for i in range(0, len(out), 9):
        on_token(out[i:i + 9])
    return out


try:
    assert app.WS == TMP, "WORKSPACE 가 임시 폴더가 아님"
    assert app.PY, "PyBaMM 이 설치된 Python 을 찾지 못함 (BATTERY_PY=...)"
    app.start_background()
    m = app.meta()
    names = [s["name"] for s in m["sets"]]
    assert "Chen2020" in names and "OKane2022" in names, names
    ok = app.set_info("OKane2022")
    assert ok["caps"]["sei"] and ok["caps"]["plating"] and ok["caps"]["lam"], ok["caps"]
    assert not app.set_info("Chen2020")["caps"]["plating"]

    # 1) 설정 검증·실험 문자열
    for bad, msg in [({"kind": "rate", "set": "Nope"}, "없습니다"),
                     ({"kind": "life", "set": "Chen2020", "mechanisms": ["plating"]}, "파라미터가 없습니다"),
                     ({"kind": "life", "set": "OKane2022", "scenarios": [{"steps": ["rm -rf /"]}]}, "형식 오류"),
                     ({"kind": "rate", "set": "Chen2020", "c_rates": "50"}, "범위"),
                     ({"kind": "life", "set": "OKane2022", "max_cycles": 10 ** 7}, "범위")]:
        try:
            app.validate(bad)
            raise AssertionError(f"검증 통과하면 안 됨: {bad}")
        except ValueError as e:
            assert msg in str(e), (bad, e)
    j = app.validate({"kind": "life", "set": "OKane2022", "mechanisms": ["lam"], "scenarios": [{"chg_C": 2, "dod": 50}]})
    assert j["mechanisms"] == ["sei", "lam"], j["mechanisms"]  # 균열면 SEI 를 위해 SEI 자동 포함
    st = app.steps_preview(j)[0]
    assert st == ["Discharge at 1C for 30 minutes or until 2.5V", "Rest for 5 minutes", "Charge at 2C until 4.2V",
                  "Hold at 4.2V until C/20", "Rest for 5 minutes"], st
    assert app.validate({"kind": "rate", "set": "Chen2020", "c_rates": "0.5, 1 2"})["c_rates"] == [0.5, 1.0, 2.0]

    # 2) 실제 짧은 계산 — 큐에 동시에 넣는다
    ids = {
        "rate": app.submit({"kind": "rate", "set": "Chen2020", "model": "SPM", "c_rates": [0.5, 2]}),
        "charge": app.submit({"kind": "charge", "set": "Chen2020", "model": "SPM", "c_rates": [1]}),
        "drive": app.submit({"kind": "drive", "set": "Chen2020", "model": "SPM", "unit": "C", "soc0": 90,
                             "profile": [[t, 1.5 if (t // 60) % 2 == 0 else -0.5] for t in range(0, 600, 5)]}),
        "life": app.submit({"kind": "life", "set": "OKane2022", "model": "SPM", "mechanisms": ["sei", "plating", "lam"],
                            "max_cycles": 60, "scenarios": [{"label": "0.5C", "chg_C": 0.5}, {"label": "2C", "chg_C": 2}]}),
        "custom": app.submit({"kind": "life", "set": "Mohtat2020", "model": "SPM", "mechanisms": ["sei"], "max_cycles": 12,
                              "eol": 99, "sei_scale": 400,
                              "scenarios": [{"label": "직접", "steps": ["Discharge at 1C until 2.8V", "Charge at 1C until 4.2V",
                                                                       "Hold at 4.2V until C/20"]}]}),
    }
    sts = {k: wait(v) for k, v in ids.items()}
    for k, s in sts.items():
        assert s["state"] == "done", (k, s)
    import portal_jobs
    assert portal_jobs.notify("owner", "done", id="x") is False and app.notify_job(sts["rate"]) is False  # 알림 꺼짐 → 아무 일도 안 함

    r = app.job_full(ids["rate"])["result"]
    a, b = [x["metrics"] for x in r["runs"]]
    assert 4.5 < a["capacity_Ah"] < 5.5 and b["capacity_Ah"] < a["capacity_Ah"], (a, b)  # 고율일수록 용량 감소
    assert b["T_max_C"] > a["T_max_C"] and b["avg_V"] < a["avg_V"]                      # 발열 증가·평균 전압 감소
    assert set(r["runs"][0]["series"]) >= {"t_min", "q_Ah", "V", "soc", "T_C"} and len(r["runs"][0]["series"]["V"]) <= 400
    assert set(r["files"]) == {"summary.csv", "curves.csv", "plot.png"}
    c = app.job_full(ids["charge"])["result"]["runs"][0]["metrics"]
    assert c["t80_min"] and c["t_total_min"] > c["t80_min"] and 4.5 < c["charged_Ah"] < 5.5, c
    d = app.job_full(ids["drive"])["result"]["runs"][0]["metrics"]
    assert d["completed"] and 0 < d["soc_end_pct"] < 90, d

    L = app.job_full(ids["life"])
    res = L["result"]
    s1, s2 = res["scenarios"]
    assert s1["cycles_done"] == 60 and len(s1["rows"]) == 60 and s1["rows"][-1]["cycle"] == 60
    assert s1["steps"][2] == "Charge at 0.5C until 4.2V" and s2["steps"][2] == "Charge at 2C until 4.2V"
    for s in (s1, s2):
        f = s["final"]
        assert 95 < f["soh"] < 100 and f["LLI_pct"] > 0 and f["Q_plating_Ah"] > 0 and f["Q_SEI_Ah"] > 0, f
        assert s["eol"]["simulated"] is None and s["eol"]["extrapolated"], s["eol"]   # 60 사이클로는 80% 못 감 → 외삽
        assert any("외삽" in n for n in s["notes"]), s["notes"]
        assert [c["cycle"] for c in s["curves"]][:2] == [1, 50] and s["curves"][-1]["cycle"] == 60  # 단계(50) 곡선 저장
    assert s2["final"]["Q_plating_Ah"] > s1["final"]["Q_plating_Ah"], "2C 충전의 도금 손실이 더 커야 함"
    assert s2["final"]["LLI_pct"] > s1["final"]["LLI_pct"]
    assert os.path.exists(os.path.join(app.job_dir(ids["life"]), "partial.json"))
    assert set(res["files"]) == {"summary.csv", "cycles.csv", "plot.png"}
    with open(os.path.join(app.job_dir(ids["life"]), "cycles.csv"), encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    assert lines[0].startswith("scenario,cycle,soh") and len(lines) == 121, len(lines)
    cu = app.job_full(ids["custom"])["result"]["scenarios"][0]
    assert cu["steps"][0] == "Discharge at 1C until 2.8V" and cu["rows"][0]["soh_dis"] is not None
    assert any("×400" in n for n in cu["notes"]) and any("38 kJ/mol" in n for n in cu["notes"]), cu["notes"]

    # 3) 취소 — 포털 작업 알림을 가짜 서버로 받아 상태 순서·내용 확인
    got = []

    class P(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            got.append((self.headers.get("X-Portal-Jobs-Token"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200); self.send_header("Content-Length", "0"); self.end_headers()
    ps = http.server.ThreadingHTTPServer(("127.0.0.1", 0), P)
    threading.Thread(target=ps.serve_forever, daemon=True).start()
    os.environ.update(PORTAL_JOBS_URL=f"http://127.0.0.1:{ps.server_address[1]}/api/internal/jobs", PORTAL_JOBS_TOKEN="tok",
                      PORTAL_JOBS_TOOL="battery-local")
    try:
        jn = app.submit({"kind": "rate", "set": "Chen2020", "model": "SPM", "c_rates": [1]})
        jc = app.submit({"kind": "life", "set": "OKane2022", "model": "DFN", "max_cycles": 500})
        time.sleep(3)
        app.cancel(jc)
        assert wait(jc, 30)["state"] == "canceled" and wait(jn)["state"] == "done"
        assert portal_jobs.flush()
    finally:
        for _k in ("PORTAL_JOBS_URL", "PORTAL_JOBS_TOKEN", "PORTAL_JOBS_TOOL"):
            os.environ.pop(_k, None)
        ps.shutdown()
    seq = lambda jid: [q["status"] for t, q in got if q["id"] == jid and q["status"] != "running"]
    assert seq(jn) == ["queued", "done"] and seq(jc) == ["queued", "canceled"], got
    assert [q["status"] for t, q in got if q["id"] == jn][1] == "running", got
    pn = next(q for t, q in got if q["id"] == jn and q["status"] == "done")
    assert all(t == "tok" for t, _ in got) and pn["tool"] == "battery-local" and pn["user"] == "owner" and pn["link"] == "/?job=" + jn, pn
    assert pn["title"].startswith("방전 C-rate 비교") and pn["message"].startswith("완료"), pn
    for done_id in (jc, ids["life"]):  # 끝난 작업은 다시 취소 안 됨(409) — 상태도 그대로
        before = app.status_of(done_id)["state"]
        try:
            app.cancel(done_id); raise AssertionError("끝난 작업 취소가 통과함")
        except app.Conflict:
            pass
        assert app.status_of(done_id)["state"] == before
    assert app.hide_paths("x /home/a/b/worker.py y 1C/2") == "x worker.py y 1C/2"

    # 4) 자연어 → 설정 → 해설(가짜 LLM) · 숫자 검사
    app.llm = fake_plan
    job, reason = app.plan("45도에서 2C 급속충전하면 수명 얼마나 줄어?")
    assert job["kind"] == "life" and len(job["scenarios"]) == 2 and job["scenarios"][1]["temp_C"] == 45 and reason == "기준과 비교"
    ex = app.explain(ids["life"])
    assert ex["unverified_numbers"] == ["1234.5"], ex["unverified_numbers"]
    assert "EOL_기준 시나리오 대비_%" in ex["facts"]["시나리오"][1]
    ok_text = f"0.5C 의 SOH 는 {round(s1['final']['soh'], 2)}% 입니다."
    assert app.check_numbers(ok_text, json.dumps(ex["facts"], ensure_ascii=False)) == []

    # 5) HTTP
    srv = app.ThreadingHTTPServer(("127.0.0.1", 0), app.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    with urllib.request.urlopen(base + "/") as rr:
        html = rr.read().decode()
        assert "data-sig" in html and 'name="author"' in html and rr.headers["X-Author"]
    assert "cdn" not in html.lower() and "https://" not in html
    mm = json.load(urllib.request.urlopen(base + "/api/meta"))
    assert mm["disclaimer"] and mm["sets"]
    with urllib.request.urlopen(f"{base}/api/jobs/{ids['life']}/file/plot.png") as rr:
        assert rr.read(4) == b"\x89PNG"
    jl = json.load(urllib.request.urlopen(base + "/api/jobs"))
    assert len(jl["jobs"]) == 7
    req = urllib.request.Request(base + "/api/jobs", json.dumps({"kind": "rate", "set": "X"}).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req)
        raise AssertionError("잘못된 세트가 통과")
    except urllib.error.HTTPError as e:
        assert e.code == 400
    try:
        urllib.request.urlopen(f"{base}/api/jobs/{ids['life']}/file/..%2Fjob.json")
        raise AssertionError("경로 탈출")
    except urllib.error.HTTPError as e:
        assert e.code == 404
    # 브라우저가 떠나도(첫 진행 이벤트 뒤 쓰기가 BrokenPipeError) 작업은 끝까지 — 결과는 작업의 해설(explain.json)에 남고, 서버 오류·트레이스백 없음
    import io, threading, urllib.request
    from http.server import ThreadingHTTPServer


    class Cut:
        def __init__(s, f, n):
            s.f, s.n = f, n

        def write(s, b):
            s.n -= 1
            if s.n < 0:
                raise BrokenPipeError(32, "Broken pipe")
            return s.f.write(b)

        def flush(s):
            pass

        def __getattr__(s, k):
            return getattr(s.f, k)


    class CutH(app.H):
        def setup(s):
            super().setup()
            s.wfile = Cut(s.wfile, 2)  # 응답 머리 + 첫 이벤트까지만 닿음


    ERRS = []


    class Srv(ThreadingHTTPServer):
        def handle_error(s, req, addr):
            ERRS.append(sys.exc_info()[1])


    def cut_post(path, body, user="dc-test"):
        cut = Srv(("127.0.0.1", 0), CutH)
        threading.Thread(target=cut.serve_forever, daemon=True).start()
        err0, sys.stderr = sys.stderr, io.StringIO()
        try:
            r = urllib.request.Request(f"http://127.0.0.1:{cut.server_address[1]}{path}", json.dumps(body).encode(),
                                       {"Content-Type": "application/json", "X-Agent-User": user})
            got, log = urllib.request.urlopen(r, timeout=60).read().decode(), sys.stderr.getvalue()
        finally:
            sys.stderr = err0
            cut.shutdown()
            cut.server_close()
        assert got.count("data: ") == 1 and not ERRS and "Traceback" not in log and log.count("연결 끊김") == 1, (got, ERRS, log)
        return os.path.join(app.WS, "u", user)
    xp = os.path.join(app.job_dir(ids["life"]), "explain.json")
    os.remove(xp)
    cut_post(f"/api/jobs/{ids['life']}/explain", {}, user="owner")
    assert json.load(open(xp, encoding="utf-8"))["unverified_numbers"] == ["1234.5"]

    srv.shutdown()
    app.delete(ids["drive"])
    assert not os.path.exists(app.job_dir(ids["drive"]))
    print(f"selftest OK ({time.time() - T0:.0f} s, PyBaMM {app.PYBAMM_VER}, {app.PY})")
finally:
    shutil.rmtree(TMP, ignore_errors=True)
