# MLP / Combined — 작성자 보류 OOF 최적화 연구

## 최종 선정·동결 (2026-11-19)

**기존 Combined MLP `EXP-CMB-011/base` 최종 유지 확정.** 공식 Training + 기존 3 Writer Hand-made만 학습하며, 신규 7·9·8·B 480장은 최종 학습에 **추가하지 않는다**.

- **고정 조건:** `MLPClassifier(hidden_layer_sizes=(256,), activation="relu", solver="adam", alpha=0.0001, learning_rate_init=0.001, batch_size=128, early_stopping=False, max_iter=200)`; 학습 공식 Size30%-Stroke28%-CoM, 손글씨 CoM-only. 추론은 CoM → /255.
- **CMB011 기존 Writer OOF:** 92.3568%, Macro F1 0.923571, OfficialVal 97.0781%. 신규 7·9 / B·8 / 전체 480장 추가 모두 Writer OOF에서 기준선 대비 소폭 하락.
- **별도 438장:** 기존 87.7283%, 전체 480장 추가 87.9376%(+0.2093%p). 소폭 이득이지만 재사용 개발셋·Bootstrap 95% 구간의 0 포함·클래스별 상충으로 신규 추가 조건 미채택.
- [**최종 모델 동결 선언**](logs/EXP-CMB-012_Final_Combined_Model_Freeze_20261119.md) · [**438장 실측 보고서**](logs/학습데이터_추가/EXP-CMB-011_03_External438_and_Selection.md) · [**동결 데이터·모델 조건 JSON**](results/Final_Model/FINAL_COMBINED_FREEZE.json).
- **정확한 범위:** 기존 12개 CV 가중치는 검증 근거이며, 새 단일 최종 추론 모델을 생성·교체한 것이 아님. Original Test·교수님 Test 미사용. 과거 CMB008 S6 가드레일 결과를 소급 변경하지 않음.

> 아래 2026-11-18의 '개발용 우선 후보/배포 미확정' 설명과 CMB010 작성 시점의 Draft PR 표기는 **과거 단계의 기록**이다. 이 날짜 이후의 최종 결정은 위 CMB012 선언이 우선한다.


> **현재 연구 상태 (2026-11-18):** EXP-CMB-001~009 개발용 실험 완료. 공식 Training 76,800장에 Fold별 학습용 Hand-made 1,920장을 합치는 **Combined**의 특징을 유지한다. Size30%·Stroke28%·CoM, MLP (256,), alpha=1e-4, LR=.001, Early Stopping=False, **Batch 128을 후속 우선 후보**로 둔다. 단, 기존 S6의 Batch 128 안전 가드레일 미통과 및 새 438장 선택 편향은 남아 있다. 최종 배포 모델은 미변경.

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
           → S1 너비 256 선정, S2 두 번째 층 추가 탈락
   ↓
           S3 alpha 강도 검증 완료 → alpha=1e-4 유지
   ↓
           S4 초기 LR 비교 → 실제 LR 선정 후 S5 Early Stopping 비교
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
| **CMB-008** | **Size30/Stroke28/CoM 동결** 후 HP 탐색 | **S1 36개 학습 완료:** 은닉층 256 OOF **92.214%**·OfficialVal **97.044%**. 4방향 <5% 미달, 과적합 가드레일 통과한 최고 정확도 대안. **S2 단층 유지·S3 alpha=1e-4 유지 실측 완료, S4→S5 연속 실험 계획 준비** | [S1 선정·탈락 이유](logs/하이퍼파라미터_최적화/EXP-CMB-008_02_Stage1_Width_Results.md) · [S2 최종 결정](logs/하이퍼파라미터_최적화/EXP-CMB-008_04_Stage2_Depth_Results.md) · [S3 L2 실측](logs/하이퍼파라미터_최적화/EXP-CMB-008_06_Stage3_Alpha_Results.md) · [S4 계획](logs/하이퍼파라미터_최적화/EXP-CMB-008_07_Plan_S4_LearningRate.md) · [S5 계획](logs/하이퍼파라미터_최적화/EXP-CMB-008_09_Plan_S5_EarlyStopping.md) |

## 2026-11-18 최신 후속 연구 — Batch Size 및 추가 전처리

**Normal과 Combined의 핵심 차이:** Normal은 **공식 MNIST·EMNIST 데이터만 학습**하고 기존 Hand-made는 진단/검증에 사용한다. 반면 Combined는 **공식 Training에 별도 손글씨 학습 데이터를 추가할 수 있다.** 지금의 4-Writer 보류 평가에서는 학습에 공식 76,800장 + 다른 Writer 3명 1,920장을 넣고, 보류한 Writer 640장은 학습에 넣지 않는다. 앞으로 오류가 큰 클래스에 새로운 손글씨를 추가하는 방안은 **추가 논의 전까지 실험·반영하지 않았다.**

- **CMB-008 S6 역사적 판단:** 36모델 사전등록 결과에서 Batch 200만 가드레일 통과. [20번 기존 실측 로그](logs/하이퍼파라미터_최적화/EXP-CMB-008_20_Stage6_Batch_Size_Results.md)는 변경하지 않는다.
- **CMB-008 21번 후속:** 별도 438장 진단 평가(24모델)에서 Batch 128 정확도 **87.519%**, Batch 200 **86.301%**, Macro F1 **0.8690 vs 0.8567**. Batch 128을 *정확도 중심 개발용 우선 후보*로 상향. [21번 후속 로그](logs/하이퍼파라미터_최적화/EXP-CMB-008_21_Stage6_External438_Followup_Results.md). 해당 438장은 사후 검토된 개발용 데이터이며 독립 Test가 아니다.
- **CMB-009:** Local Warp, Rotation ±3°/±5°, Shear ±0.05/±0.10을 Batch 128에서 각각 평가(총 72개 모델). Rotation ±5°의 OOF가 **92.3698%**로 기준선 92.2786%보다 소폭 증가했으나 9→7 악화와 사전 가드레일 미통과. Shear ±0.05는 7↔9를 개선했으나 OOF 저하. **추가 전처리 자동 채택·조합 학습 없음.** [CMB-009 실험 로그](logs/가설B/EXP-CMB-009_01_Sequential_Preprocessing_Results.md) · [정량 CSV](results/Sequential_Preprocessing/standalone_table.csv).
- **모델 적용 범위:** Batch 128 및 기존 전처리는 **후속 연구의 기준 후보**이며, 기존 저장된 최종 배포 체크포인트/Original Test·교수님 Test는 변경하지 않았다.

## CMB-007까지의 관측과 한계

- 전체 31조건에서 OOF 정확도 상위 세 조합은 **Size30%-Stroke28%, 30%-14%, 30%-0%**이며 오분류율은 **9.284%, 9.310%, 9.427%**.
- 고비율 증강은 평균 Family Flip을 낮추지만 **8→B 오류 증가 / B→8 오류 감소**, OfficialVal 성능 하락, 특정 클래스 최악 반례를 동반했다.
- 사후 순위가 독립 데이터에서의 우월성 또는 최종 모델 선정 근거인 것은 아니다. **Original Test·교수님 Test·Normal 신규 480장은 CMB-007 모델 선택에 사용하지 않았다.**

## 후속 연구에서 고정된 결정

- **선정한 학습 증강:** Official Training의 30%를 클래스 균형 Size·Stroke 강도 28%로 대체하며, 학습·보류 Writer·OfficialVal의 중심 정렬은 CoM. **추론 시 Size를 30% 줄이거나 Stroke를 28% 변경한다는 뜻이 아니다.**
- **다음 변수:** [CMB-008 HP 단계별 사전등록](logs/하이퍼파라미터_최적화/EXP-CMB-008_01_Plan_30_28_Frozen_HP.md) 참조. S1~S3에서 은닉층 (256,), alpha=1e-4를 잠정 선정했으며, 다음 S4·S5에서 학습률과 Early Stopping을 순차 검증한다. 최종 배포 모델은 미확정이다.

## 자료 찾기

[실험일지](logs/README.md) · [실측 CSV/그림](results/README.md) · [재현 코드](scripts/README.md) · [가설 A 보류 근거](logs/가설A/README.md) · [실험 공통 양식](../../EXPERIMENT_TEMPLATE.md).

**번호 구분:** Normal 실험의 `EXP유###` 번호는 별개이며 Combined의 `EXP-CMB-###`와 혼용하지 않는다.

<!-- CMB008_S4S5_FINAL_START -->
**CMB-008 S1~S5 검증 완료:** (256,), alpha=1e-4, LR **0.001**, Early Stopping **False**, OOF **92.214%**, OfficialVal **97.044%**. [선정·탈락 근거](logs/하이퍼파라미터_최적화/EXP-CMB-008_11_Final_HP_Decision.md). 최종 Test·배포 미변경.
<!-- CMB008_S4S5_FINAL_END -->

## 발표 자료 — S5 Learning Curve 및 현재 결정

**Early Stopping=False 유지 (개발용 HP).** 같은 실행에서 관측한 Training vs Validation Accuracy·Loss 그래프의 과적합 징후와, 별도 S5 Writer OOF 평가에서 False가 True보다 좋았던 근거를 구분했다. [**PPT용 16:9 그래프 및 발표 문장**](logs/하이퍼파라미터_최적화/EXP-CMB-008_18_S5_Presentation_LearningCurve_ESFalse.md). Test 평가·최종 배포는 미실시.

## EXP-CMB-010 — 7·8·9·B 추가 Training 통제 실험 (실측 완료, main 미반영)

- 사용자가 새로 제공한 실제 손글씨 480장(7 160·9 160·8 80·B 80)을 QA 검증 후 단일 신규 작성자 미확인 코호트로 보존했다. [원본 NPZ](../../../02_data/Hand_made_extra/CMB010_7_8_9_B_480.npz) · [품질 검사](logs/학습데이터_추가/EXP-CMB-010_01_Input_QA_and_Train_Plan.md).
- 기준선 78,720장과 추가 학습 79,200장을 **동일 Batch 128·MLP (256,)·Size30/Stroke28/CoM**으로 Seed 3 × 기존 Writer 4 × 2조건 = 24모델 통제 비교했다.
- Writer OOF **92.4089% → 92.2917%**, OfficialVal **97.0972% → 97.0677%**로 하락. D→0, B→8은 개선됐지만 7↔9, 0→D는 악화돼 **480장 전체 보강은 최종 채택하지 않았다**. [실측 결과](logs/학습데이터_추가/EXP-CMB-010_02_Added_480_Training_Results.md) · [실측 CSV/JSON](results/Targeted_Handmade_480/README.md).
- 이번 내용은 [Draft PR #38](https://github.com/ryu980920/Machine_Learning/pull/38)에서만 보존한다. **main의 기존 학습 데이터/모델 및 독립 Test는 변경하지 않았다.**
