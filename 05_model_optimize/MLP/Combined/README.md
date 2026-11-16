# MLP / Combined — 작성자 보류 OOF 최적화 연구

> **현재 상태:** **EXP-CMB-001~007**의 실험·결과를 기록했고, 후속 **EXP-CMB-008 하이퍼파라미터 최적화는 설계·사전등록 단계**다. 실험 번호는 하나의 `EXP-CMB-###` 체계이며, 계획/중간/최종/후속 분석은 **동일 실험의 문서 단계**다. **Size30/Stroke28/CoM의 학습 증강 조건은 선정했으나 최종 MLP 하이퍼파라미터·배포 체크포인트는 미확정.**

## 연구 흐름 (순서대로 읽기)

```text
CMB-001   Combined Baseline (4명 Writer Leave-One-Out)
   ↓
CMB-002   오분류 원인 분석 (0↔D, 8↔B)
   ↓
CMB-003   기존 모델 고정 입력 민감도 진단
   ↓  가설 B: 크기·중심·획 분포의 영향 검증
CMB-004   Geometry Normalization (Size / CoM / Size+CoM)
   ↓
CMB-005   Size 0~30% × Stroke (10조건)
   ↓
CMB-006   Size 40~50% 추가 (누적 16조건)
   ↓
CMB-007   Size 60~100% 추가 (누적 31조건)
           → 정확도·클래스별 안정성·오분류 방향 분석
   ↓  전처리 탐색 종료: Size30 / Stroke28 / CoM 선정
CMB-008   MLP 구조 → L2 → 학습률 → 조기 종료 순차 최적화
           → 실험 전 사전등록 완료, 신규 HP 실측 미실행
```

[**전체 실험 일지 및 단계별 문서 링크**](logs/README.md) · [가설 B 통합 설명](logs/가설B/README.md)

## 연구 단계별 핵심 결과

| 실험 | 수행 내용 | Hand-made OOF 및 판단 | 상세 |
|---|---|---|---|
| **CMB-001** | Baseline, 4-Writer 보류 OOF | Normal 62.578% → Combined **81.250%** | [로그](logs/EXP-CMB-001_Baseline_4Writer_OOF.md) |
| **CMB-002** | 오류·형태/공간 단서 분석 | 가설 B 우선, 가설 A 조건부 보류 | [로그](logs/초기분석/EXP-CMB-002_Error_Analysis.md) |
| **CMB-003** | 고정 MLP 입력 민감도 비교 | 95% 축소의 작성자 편중과 얇게25% 후보 확인 | [로그](logs/초기분석/EXP-CMB-003_Input_Sensitivity.md) |
| **CMB-004** | Size / CoM 정규화 기전 비교 | Size+CoM **88.945%**, 공식 Validation 손실 | [최종 보고](logs/가설B/EXP-CMB-004_02_Final_Geometry_Normalization.md) |
| **CMB-005** | Size 0~30% × Stroke, 120개 MLP | 전체 10조건 중 30%-28% **90.716%** | [최종 보고](logs/가설B/EXP-CMB-005_03_Final_Size00_30.md) |
| **CMB-006** | Size 40~50%, 72개 MLP 추가 | 누적 16조건 비교, 30%-28% 우위 유지 | [계획](logs/가설B/EXP-CMB-006_01_Plan_Size40_50.md) · [16조건 결과](results/Balanced_Scale_Stroke_Extension/full_0to50_comparison/SUMMARY.md) |
| **CMB-007** | Size 60~100%, 180개 MLP 추가 | 누적 **31조건** 비교, 최고 OOF 갱신 실패, 8↔B 비대칭 오류 관측 | [최종 보고](logs/가설B/EXP-CMB-007_03_Final_Size60_100.md) · [오분류율 후속 분석](logs/가설B/EXP-CMB-007_04_Review_Top3_OOF_Error.md) |
| **CMB-008** | **Size30/Stroke28/CoM 고정** 후 단계별 MLP HP 최적화 | **사전등록 완료, 신규 학습 미실행** | [왜 이 HP를 다음에 시험하고 어떻게 확정할지](logs/하이퍼파라미터_최적화/EXP-CMB-008_01_Plan_30_28_Frozen_HP.md) |

## CMB-007까지의 관측과 한계

- 전체 31조건에서 OOF 정확도 상위 세 조합은 **Size30%-Stroke28%, 30%-14%, 30%-0%**이며 오분류율은 **9.284%, 9.310%, 9.427%**.
- 고비율 증강은 평균 Family Flip을 낮추지만 **8→B 오류 증가 / B→8 오류 감소**, OfficialVal 성능 하락, 특정 클래스 최악 반례를 동반했다.
- 사후 순위가 독립 데이터에서의 우월성 또는 최종 모델 선정 근거인 것은 아니다. **Original Test·교수님 Test·Normal 신규 480장은 CMB-007 모델 선택에 사용하지 않았다.**

## 후속 연구에서 고정된 결정

- **선정한 학습 증강:** Official Training의 30%를 클래스 균형 Size·Stroke 강도 28%로 대체하며, 학습·보류 Writer·OfficialVal의 중심 정렬은 CoM. **추론 시 Size를 30% 줄이거나 Stroke를 28% 변경한다는 뜻이 아니다.**
- **다음 변수:** [CMB-008 HP 단계별 사전등록](logs/하이퍼파라미터_최적화/EXP-CMB-008_01_Plan_30_28_Frozen_HP.md) 참조. S1 너비부터 검증하고, 각 단계에서 실패 가드레일 및 동결 이유를 기록한다.

## 자료 찾기

[실험일지](logs/README.md) · [실측 CSV/그림](results/README.md) · [재현 코드](scripts/README.md) · [가설 A 보류 근거](logs/가설A/README.md) · [실험 공통 양식](../../EXPERIMENT_TEMPLATE.md).

**번호 구분:** Normal 실험의 `EXP유###` 번호는 별개이며 Combined의 `EXP-CMB-###`와 혼용하지 않는다.
