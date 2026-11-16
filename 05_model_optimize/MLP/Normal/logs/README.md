# MLP / Normal — 실험 로그

연구 흐름은 **초기분석 → 가설 B → 가설 A** 순서로 관리한다. 각 폴더의 README에서 검증 목적과 연구 현황을 확인할 수 있다.

| 구분 | 실험 로그 | 목적 |
|---|---|---|
| [초기분석](초기분석/README.md) | [EXP유001](초기분석/EXP유001_Pixcel_Attribution.md) · [EXP유002](초기분석/EXP유002_Input_Sensitivity.md) | 가설 A/B를 설정하고 픽셀 기여도·입력 민감도를 탐색 |
| [가설 B](가설B/README.md) | [EXP유003](가설B/EXP유003_Spatial_Response.md) · [EXP유004](가설B/EXP유004_Automatic_Centering.md) · [EXP유005](가설B/EXP유005_CoM_Translation_Augmentation.md) | 위치·공간 분포 의존성과 CoM 정렬·±1px 위치 증강 |
| [가설 A](가설A/README.md) | [EXP유001](가설A/EXP유001_픽셀판단근거_분석.md) · [EXP유002](가설A/EXP유002_형태대좌표_분리검증.md) · [EXP유003](가설A/EXP유003_형태위치_혼동원인_통합진단.md) | 모델의 픽셀 판단 기준과 획의 형태·좌표 반응 분리 분석 |

**번호 규칙:** 초기분석·가설 B의 EXP유001~005는 기존 번호를 유지한다. 가설 A는 **독립적인 실험 번호 EXP유001부터 시작한다.** 같은 번호를 언급할 때는 가설명을 명시한다.

- [05 최적화 공통 양식](../../../EXPERIMENT_TEMPLATE.md) · [기존 상세 양식](../../../../03_experiments/EXPERIMENT_TEMPLATE.md)
- 기록 순서: **문제 발견 → 가설 → 통제 조건 → 실측 결과 → 한계 → 후속 연구**.
- 기존 `scripts/`, `results/`, 모델 체크포인트는 변경하지 않는다. 이번에는 로그 파일과 관련 README를 정리한다.

## Normal 최상위 독립 원인 규명 실험 (2026-11-16)

가설 A/B 일반 연구 로그와 분리하여 Normal 루트에서 관리한다.

- [EXP유010 — 글자 크기 분포](../EXP010_Geometry_Distribution/README.md)
- [EXP유011 — CoM·재학습의 분리](../EXP011_CoM_Effect/README.md)
- [EXP유012 — 크기 관련 Shortcut 민감도](../EXP012_Size_Shortcut/README.md)

각 실험 폴더에 계획·결과 통합 `README.md`, 실측 `summary.csv`, `RESULTS.md`, 재현용 `RUN.md`가 있다. 기존 코드는 Normal `scripts/`에 보존한다.
