# Combined MLP — 전체 연구 흐름 (EXP-CMB-001~007)

> **번호 규칙:** Combined의 모든 실험은 `EXP-CMB-001`부터 순차 표기한다. 과거 `EXP유001~004`라는 폴더별 번호는 사용하지 않는다. 같은 실험 안의 문서는 `01_Plan → 02_Interim → 03_Final → 04_Review` 순서다. 이 번호는 **진행 단계**이며 별도 실험이 아니다.
>
> **연구 상태:** CMB-001~007 실측·분석 기록 완료. 최고 성능 조건 탐색은 개발용 Hand-made OOF/OfficialVal 기준이며 **최종 배포 모델은 결정하지 않았다. Original Test/교수님 Test는 이 연구의 조건 선택에 사용하지 않았다.**

## 1. 연구 진행 순서

| 순서 | 구분 | 실험과 핵심 질문 | 주요 관측·판단 | 보고서 |
|---|---|---|---|---|
| 1 | 기준선 | **CMB-001** — 공식 데이터만 학습한 Normal과 Hand-made를 포함한 Combined의 작성자 일반화 비교 | 4-Writer 보류 OOF **81.250%** (Normal 62.578%) | [Baseline](EXP-CMB-001_Baseline_4Writer_OOF.md) |
| 2 | 원인 탐색 | **CMB-002** — 왜 0↔D, 8↔B 오류가 발생하는가? | 공간·형태 관련 반례 확인, **가설 B 우선 / 가설 A 조건부 보류** | [Error Analysis](초기분석/EXP-CMB-002_Error_Analysis.md) |
| 3 | 모델 고정 진단 | **CMB-003** — 이동·크기·획 등의 단일 입력 변형 효과와 부작용은? | 획 얇게25% 개선 경향 / 95% 축소는 작성자 편중, 후속 재학습 필요 | [Input Sensitivity](초기분석/EXP-CMB-003_Input_Sensitivity.md) |
| 4 | 가설 B — 정규화 | **CMB-004** — Size와 Center(CoM)의 효과를 분리할 수 있는가? | Size-only 80.781%, CoM-only 85.859%, Size+CoM **88.945%** OOF. OfficialVal 손실 | [계획](가설B/EXP-CMB-004_01_Plan_Geometry_Normalization.md) · [최종 결과](가설B/EXP-CMB-004_02_Final_Geometry_Normalization.md) |
| 5 | 가설 B — 저비율 증강 | **CMB-005** — Size 0~30% × Stroke 0/14/28%의 오분류·크기 편향 변화는? | **10조건·120모델**, 30%-28% OOF **90.716%** | [계획](가설B/EXP-CMB-005_01_Plan_Size00_30_Stroke.md) · [중간](가설B/EXP-CMB-005_02_Interim_Size00_10.md) · [최종](가설B/EXP-CMB-005_03_Final_Size00_30.md) |
| 6 | 가설 B — 비율 확장 | **CMB-006** — Size 40~50%를 추가하면 개선이 이어지는가? | **6조건·72모델**, 기존 10조건과 합쳐 16조건 비교 | [사전 계획](가설B/EXP-CMB-006_01_Plan_Size40_50.md) · [16조건 원본 결과](../results/Balanced_Scale_Stroke_Extension/full_0to50_comparison/SUMMARY.md) |
| 7 | 가설 B — 상한·반례 검증 | **CMB-007** — Size 60~100%에서 정확도와 Family Flip의 상충은? | **15조건·180모델**, 과거 데이터 포함 **31조건 비교**. 8↔B 방향별 오류 교환·100% 증강 반례 | [계획](가설B/EXP-CMB-007_01_Plan_Size60_100.md) · [중간](가설B/EXP-CMB-007_02_Interim_Size60_80.md) · [최종](가설B/EXP-CMB-007_03_Final_Size60_100.md) · [후속 오분류율 분석](가설B/EXP-CMB-007_04_Review_Top3_OOF_Error.md) |

## 2. 가설별 연결

- **[초기분석](초기분석/README.md) — CMB-002·003:** 오류 위치와 원인을 찾고, 모델 고정 입력 변형으로 후속 실험의 방향을 정한다.
- **[가설 B](가설B/README.md) — CMB-004~007:** 기하 정규화 → 저비율 증강 → 중비율 확장 → 고비율 반례 분석으로 이어지는 **하나의 연속 연구**다.
- **[가설 A](가설A/README.md):** 획 연결 구조의 인과적 기여는 추가 근거가 부족하여 독립 후속 실험 보류. 가설 B 연구 완료와 혼동하지 않는다.

## 3. 개발용 성능과 최종 선정은 구분

전체 31조건의 **OOF 정확도 상위 3개**는 Size/Stroke **30%-28%, 30%-14%, 30%-0%**로 오분류율이 각각 **9.284%, 9.310%, 9.427%**다. 이는 학습 조건의 개발용 순위이며, **독립 신규 작성자 일반화나 최종 배포 모델의 선택을 의미하지 않는다.**

[원본 결과 모음](../results/README.md) · [재현 코드](../scripts/README.md) · [Combined 개요](../README.md) · [공통 실험 양식](../../../EXPERIMENT_TEMPLATE.md).
