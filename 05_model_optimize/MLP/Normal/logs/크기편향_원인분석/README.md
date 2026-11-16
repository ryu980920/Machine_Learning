# 숫자·문자 정확도 역전 원인분석 — EXP유010·011·012

CoM-only, Size-only, Size→CoM에서 숫자와 문자 정확도가 역전되는 원인을 **세 가설에 대한 하나의 연구 시리즈**로 관리한다. 이 폴더는 `logs/가설A/`, `logs/가설B/`와 **동일한 깊이**에 위치한다. 연구 번호 010·011·012는 그대로 유지한다.

## 연구 목록

| 실험 | 검증 가설 | 원본 실측의 핵심 | 연구 기록 / 수치 |
|---|---|---|---|
| **EXP유010** | Training과 Hand-made의 크기 분포 차이 | 공식 B/D/F 긴 변 24px, 손글씨 20px 확인 | [실험일지](EXP유010_크기분포_분석.md) · [summary.csv](EXP유010_summary.csv) |
| **EXP유011** | Size-only의 숫자 붕괴는 CoM 및 재학습과 관련되는가 | 같은 Size 모델에서 평가 입력을 CoM으로 바꾸자 숫자 36.46%→73.81% | [실험일지](EXP유011_Size_CoM_교차분석.md) · [summary.csv](EXP유011_summary.csv) |
| **EXP유012** | MLP가 크기 관련 특징을 숫자·문자 구분 단서로 쓰는가 | 동일 이미지 0.8~1.2배 개입 시 숫자/문자 판단 경향 역전 | [실험일지](EXP유012_크기분류단서.md) · [summary.csv](EXP유012_summary.csv) |

각 실험일지는 **사전 계획, 실험 절차, 실제 결과, 해석 및 한계, 재현 방법**을 한 문서에 통합한다. 각 CSV는 이전 GitHub 실험 폴더의 파일 내용을 **변경 없이 옮긴 핵심 지표 요약**이다.

## 재현성 및 데이터 경계

- **실제 완료된 공통 GitHub Actions:** [EXP유010~012 원본 분석 #43801194796](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796)
- **재사용된 EXP유006의 9개 모델:** [원본 학습 #43785390739](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739)
- **공통 원본 코드:** [EXP010 크기 분포](../../scripts/exp010_geometry_distribution.py) · [EXP011/012 모델 고정·크기 개입](../../scripts/exp011_012_causal_probes.py) · [종합 집계](../../scripts/exp010_012_aggregate.py)
- **전체 전수 결과:** 위 GitHub Actions의 Artifacts (보존 기간 제한). 요약 CSV는 영구 GitHub 기록이나 전체 예측 결과를 대체하지 않는다.
- **분석 조건:** Official Train 76,800장, Official Validation 19,200장, 기존 작성자 4명의 Hand-made 2,560장; Seed 42·43·44.
- **안전 경계:** 기존 체크포인트를 고정해 평가했으며 새 학습, 공식 Original Test·교수님 Test 접근, 최종 Pipeline 변경은 없다.

[Normal 메인 안내](../../README.md) · [가설 A](../가설A/README.md) · [가설 B](../가설B/README.md)

## Shortcut 직접검증과 해결 학습 — 후속 실험

- [EXP유013 — 기하학적 특징만으로 숫자·문자 이진분류](EXP유013_기하특징_이진분류.md): 원본/CoM/Size/Size→CoM에서 4개 Geometry Features만 사용한 별도 이진분류기로 Shortcut 단서의 존재 측정. [완료·원본 실측](https://github.com/ryu980920/Machine_Learning/actions/runs/43809791116).
- [EXP유014 — 크기 편향 완화를 위한 Class-balanced 학습](EXP유014_Shortcut_완화_학습.md): Training 전용 Scale 및 Stroke 다양화 3조건 × 3 Seed MLP를 기존 EXP유006 CoM / Size→CoM과 비교. [재실행 실측 확인](https://github.com/ryu980920/Machine_Learning/actions/runs/43810141307).

기존 코드는 `Normal/scripts/` 아래 유지하며 이 연구는 브랜치에서만 실행하고 최종 모델 또는 main을 변경하지 않는다.

### EXP유013~014 완료 판정 (2026-11-16)

- **EXP유013:** 원본 공식 Validation에서 Size 하나만의 숫자/문자 이진분류 Balanced Accuracy **99.7639%**; Size→CoM 이후 **51.0931%**. 다만 Width/Height/Mass 포함 4변수에서는 정규화 후 **64.3972%**로 잔존 구별력 확인.
- **EXP유014:** 기존 CoM MLP의 1.2배 숫자 family flip **71.60%** → Class-balanced Scale **8.96%**, Scale+Stroke **6.83%**. Hand-made 숫자/문자 동시 개선 확인.
- **채택 보류:** 신규 A/B의 공식 Validation 각각 **95.6962% / 95.6007%**, 기존 CoM **97.0521%**보다 낮아 기준 **97.30%**에 미달. 재학습 모델은 실험 브랜치 Artifact에만 보존하고 `main` 또는 최종 학습 Pipeline을 변경하지 않음.
- **원본 실측 및 3-Seed 집계:** [GitHub Actions #43810141307](https://github.com/ryu980920/Machine_Learning/actions/runs/43810141307) · `Normal-EXP013-014-measured-shortcut-review`.


## EXP유015 — 크기 Shortcut 재발 방지 우선 증강 최적화 (완료)

[실측 실험일지](EXP유015_증강비율_획강도_최적화.md) · [Gate 합격 9개 후보 요약 CSV](EXP유015_shortcut_gate_통과9조건_요약.csv) · [OfficialVal 크기 변형 진단 CSV](EXP유015_OfficialVal_크기변형_요약.csv).

- **78개 새 MLP, 26개 크기 비율×획 강도 조합, 3 Seed**. 사전 고정 Shortcut Gate를 통과한 9개 후보 중 OfficialVal이 가장 높은 조건은 **30% 크기 증강 / 획 강도 28% / 전체 CoM**, OfficialVal **96.1806%**, Hand-made 숫자 **88.8125%**, 문자 **87.7778%**.
- Hand-made 숫자1.2배·문자0.8배 Family flip: **9.44% / 9.93%**. OfficialVal 같은 개입: **5.00% / 5.23%** (기존 CoM은 **75.67% / 49.42%**).
- **중요한 잔존 문제:** Hand-made B·8·D는 개별 크기 변경에서 여전히 20%대 Family flip이 나타남. 크기 Shortcut **완전 제거 증명 아님**, 목표 OfficialVal 97.30%도 미달.
- 실측 원본: [1차 45모델](https://github.com/ryu980920/Machine_Learning/actions/runs/43814205670) · [2차 33모델 및 통합 Gate](https://github.com/ryu980920/Machine_Learning/actions/runs/43815571141) · [OfficialVal 동결모델 9개 교차 진단](https://github.com/ryu980920/Machine_Learning/actions/runs/43817096445).
- 전체 실행은 [Draft PR #13](https://github.com/ryu980920/Machine_Learning/pull/13)에 보존하고 기존 `main`/최종 모델은 그대로 유지.


### EXP유016 전 로그 동기화 확인 (2026-11-16)

EXP유010~015 MD·요약 CSV 반영 상태를 원본 브랜치에서 검증했다. EXP유015는 전체 26개 조건의 **[합격/탈락 전체 CSV](EXP유015_26조건_전체판정.csv)**와 **[B·8·D 여섯 크기 개입 세부 CSV](EXP유015_26조건_B8D_전환율_전체.csv)**를 추가 영구 보존했다. 3-Seed 상세 CSV/NPZ·체크포인트는 [1차 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43814205670), [2차 Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/43815571141), [OfficialVal Audit](https://github.com/ryu980920/Machine_Learning/actions/runs/43817096445) Artifact에 보존한다. 아직 `main`에 병합하지 않았으며 새로운 실험은 별도 브랜치로 진행한다.
