# THIRD_PARTY_NOTICES — battery-local

이 저장소의 코드(`app.py`·`worker.py`·`ui.html`·`selftest.py`·`setup.sh`)는 DongJu Kim(gggg8657)이 작성했으며 PolyForm Noncommercial 1.0.0 입니다. 웹 서버와 화면은 Python 표준 라이브러리와 자체 SVG 차트만 쓰고, 제3자 코드·글꼴·데이터를 동봉하지 않습니다. 계산은 서버에 이미 설치된 PyBaMM conda 환경의 python 으로 `worker.py` 를 서브프로세스로 실행합니다.

조사 기준: 2026-10-12. 실제 파일(import 문, `setup.sh`)을 보고, 라이선스는 설치된 패키지(`~/miniforge3/envs/pybamm-inv`)의 dist-info 와 원 저장소에서 확인했습니다. 버전은 이 서버의 값입니다(고정 아님).

## 구성요소

| 이름 | 버전 | 쓰임 | 라이선스 | 링크 | 상업 이용 | 비고 |
|---|---|---|---|---|---|---|
| PyBaMM | 26.3.0 | 설치(별도 conda 환경), 서브프로세스 | BSD-3-Clause (© 2018 the PyBaMM team) | https://github.com/pybamm-team/PyBaMM | 가능 | 인용: Sulzer et al., *J. Open Res. Softw.* 9(1):14 (2021) |
| PyBaMM 내장 파라미터 세트: Chen2020, OKane2022, ORegan2022, Mohtat2020, Ecker2015, Marquis2019, Ai2020, NCA_Kim2011, Ramadass2004, Prada2013 | (PyBaMM 26.3.0 안) | PyBaMM 에서 이름으로 불러 씀(값을 복사하지 않음) | PyBaMM 과 같이 BSD-3-Clause 로 배포. 값의 출처는 각 원 논문 | `pybamm.parameter_sets.get_docstring(<이름>)` | 가능(인용 권장) | 결과를 발표할 때 해당 논문과 PyBaMM 을 인용. Ramadass2004 는 PyBaMM 문서상 여러 문헌을 섞은 세트 |
| SEI 활성화에너지 기본값 38 kJ/mol (`worker.py` `EA_SEI_DEFAULT`) | — | 코드 상수 | OKane2022 세트의 값(사실 데이터) | O'Kane et al., *PCCP* 24, 7909 (2022) | 가능 | 출처를 주석에 적음 |
| pybammsolvers (IDAKLU) | 0.6.0 | 설치(PyBaMM 의존성) | BSD-3-Clause (© 2024 PyBaMM Team) | https://github.com/pybamm-team/pybammsolvers | 가능 | 휠 안에 SUNDIALS(BSD-3), SuiteSparse KLU·AMD·COLAMD·BTF(LGPL-2.1 / BSD-3), OpenBLAS(BSD-3), libgfortran·libgomp(GPL+GCC Runtime Library Exception) 공유 라이브러리 포함 |
| CasADi | 3.7.2 | 설치(PyBaMM 의존성) | LGPL-3.0-or-later | https://github.com/casadi/casadi | 조건부 | 수정 없이 동적 라이브러리로 씀. 재배포할 때는 LGPL 고지·교체 가능성 보장. 이 저장소는 동봉 안 함 |
| NumPy / SciPy | 2.3.5 / 1.17.1 | 설치(의존성) | BSD-3-Clause | https://numpy.org , https://scipy.org | 가능 | |
| Matplotlib | 3.10.8 | 설치(선택, PNG 저장) | Matplotlib License (PSF 기반) | https://matplotlib.org | 가능 | |
| 수명 외삽식 L(n) = a·√n + b·n | — | `worker.py` | 일반적인 경험식(SEI 확산 √t + 선형 항) | — | 해당 없음 | 특정 코드를 옮기지 않음 |
| Google Gemma 4 `gemma4:31b` (Ollama GGUF 빌드) | — | 외부 서비스(OpenAI 호환 API, 질문 → 계산 계획·설명) | Apache-2.0 (Ollama `api/show` 의 license 필드로 확인) | https://ai.google.dev/gemma | 가능 | 가중치 동봉 안 함 |
| Ollama | — | 외부 서비스(LLM 서버) | MIT | https://github.com/ollama/ollama | 가능 | 동봉 안 함 |

같은 서버의 PyBaMM_Inverse 연구 코드는 읽기만 했고 복사하지 않았습니다(열화 옵션 조합·도금 파라미터 출처 확인에 참고).

## 확인 필요 (상업 이용 전)

현재 상업 이용 확인이 필요한 항목은 찾지 못했습니다. CasADi·SuiteSparse 일부(LGPL)는 이 저장소에 동봉하지 않고 설치된 것을 그대로 쓰므로, 이 도구를 묶어 재배포할 때만 LGPL 고지를 챙기면 됩니다.

이 문서는 법률 자문이 아니며, 라이선스는 각 원문이 우선합니다.
