# Combined MLP — 전체 연구 흐름 (EXP-CMB-001~009)

## 최종 Combined MLP 선정 (2026-11-19)

**EXP-CMB-012: 신규 480장 추가 이전의 `CMB011/base`를 최종 모델 구성으로 동결.** [선정 근거·고정 사양](EXP-CMB-012_Final_Combined_Model_Freeze_20261119.md) · [438장 실측·후속 판단](학습데이터_추가/EXP-CMB-011_03_External438_and_Selection.md) · [동결 Manifest](../results/Final_Model/FINAL_COMBINED_FREEZE.json).

CMB011의 7·9·B·8 데이터 추가 연구는 최종 학습 구성으로 채택하지 않았으며 원본·성공/실패 기록은 모두 보존한다. 기존 Writer OOF·OfficialVal과 재사용 개발용 438장을 함께 검토한 결정이다. 새 단일 모델 가중치를 생성한 것은 아니고 Original Test·교수님 Test는 미사용이다.


> **번호 규칙:** Combined의 모든 실험은 `EXP-CMB-001`부터 순차 표기한다. 과거 `EXP유001~004`라는 폴더별 번호는 사용하지 않는다. 같은 실험 안의 문서는 `01_Plan → 02_Interim → 03_Final → 04_Review` 순서다. 이 번호는 **진행 단계**이며 별도 실험이 아니다.
>
> **연구 상태 (2026-11-18):** CMB-001~009 실측·분석 완료. Size30/Stroke28/CoM + MLP (256,), alpha=1e-4, LR=.001, Early Stopping=False, **Batch 128은 후속 개발용 우선 후보**. 최종 배포는 미실시. Combined는 공식 Training에 **추가 Hand-made 학습 데이터**를 포함할 수 있으며, 작성자 보류 CV로 학습/평가 겹침을 방지한다.

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
| 8 | 하이퍼파라미터 최적화 | **CMB-008** — Size30/Stroke28/CoM 고정 후 구조·규제·학습률을 단계적으로 검증 | **S1 36개 학습 완료**, (256,) OOF 92.214%·OfficialVal 97.044% / 4방향 <5% 목표는 아직 미달 / **S2·S3 실측 완료 / (256,), alpha=1e-4 유지 / S4→S5 사전등록** | [S1 실측·채택 이유](하이퍼파라미터_최적화/EXP-CMB-008_02_Stage1_Width_Results.md) → [S2 결과](하이퍼파라미터_최적화/EXP-CMB-008_04_Stage2_Depth_Results.md) → [S3 실측](하이퍼파라미터_최적화/EXP-CMB-008_06_Stage3_Alpha_Results.md) → [S4 계획](하이퍼파라미터_최적화/EXP-CMB-008_07_Plan_S4_LearningRate.md) → [S5 계획](하이퍼파라미터_최적화/EXP-CMB-008_09_Plan_S5_EarlyStopping.md) |

## 2026-11-18 최신 상태와 다음 연구 경계

- **CMB-008 20번 [기존 S6 실측](하이퍼파라미터_최적화/EXP-CMB-008_20_Stage6_Batch_Size_Results.md):** 사전 가드레일에 따라 Batch 200 유지 추천(역사적 판단 보존).
- **CMB-008 21번 [438장 후속 평가](하이퍼파라미터_최적화/EXP-CMB-008_21_Stage6_External438_Followup_Results.md):** Batch 128의 평균 438장 정확도 87.519%, Batch 200 86.301%. Batch 128을 정확도 중심 우선 후보로 재검토하되, 과거 가드레일 통과로 바꾸어 적지 않음.
- **CMB-009 [추가 전처리 72모델 실측](가설B/EXP-CMB-009_01_Sequential_Preprocessing_Results.md):** Local Warp·Rotation·Shear 모두 사전 조합 가드레일 미통과 → **기존 Size30/Stroke28/CoM + Batch 128 유지**, 조합 학습은 조건 미충족으로 실행하지 않음. [정량 CSV](../results/Sequential_Preprocessing/standalone_table.csv).
- **Combined의 차별성:** Normal은 공식 데이터만 학습하지만 Combined는 **공식 학습 + Hand-made 학습**이 가능함. 4명 보류 CV에서는 **다른 3명 1,920장만 학습**, 보류 Writer 640장만 OOF 평가한다. 오분류가 많은 클래스에 새 손글씨를 추가하는 연구는 **추후 논의 예정, 아직 데이터 편입·재학습 없음**.
- 438장은 사용자 검토를 거친 개발 데이터로 최종 Test가 아니며 Original Test·교수님 Test 및 최종 배포 모델은 변경하지 않았다.

## 2. 가설별 연결

- **[초기분석](초기분석/README.md) — CMB-002·003:** 오류 위치와 원인을 찾고, 모델 고정 입력 변형으로 후속 실험의 방향을 정한다.
- **[가설 B](가설B/README.md) — CMB-004~007:** 기하 정규화 → 저비율 증강 → 중비율 확장 → 고비율 반례 분석으로 이어지는 **하나의 연속 연구**다.
- **[가설 A](가설A/README.md):** 획 연결 구조의 인과적 기여는 추가 근거가 부족하여 독립 후속 실험 보류. 가설 B 연구 완료와 혼동하지 않는다.

## 3. 전처리 조합 선정과 신규 HP 연구

[EXP-CMB-008 사전등록](하이퍼파라미터_최적화/EXP-CMB-008_01_Plan_30_28_Frozen_HP.md): 30%-28%는 후속 학습 증강 조건으로 **선정**했으며, 최종 MLP HP는 모든 단계 실측 후 별도 결정한다. S1은 (256,)을 S2 대조 기준으로 동결했고, D→0 / B→8 / 8→B / 0→D **각각 5% 미만 목표**는 미달했으므로 S2에서 단층을 유지하고 S3에서 alpha=1e-4를 유지한 뒤 S4·S5로 이어진다. **너비 → 깊이 → alpha → LR → Early Stopping**의 단계마다 왜 변경했는지, 무엇이 개선·악화되었는지를 기록한다.

## 4. 개발용 성능과 최종 선정은 구분

전체 31조건의 **OOF 정확도 상위 3개**는 Size/Stroke **30%-28%, 30%-14%, 30%-0%**로 오분류율이 각각 **9.284%, 9.310%, 9.427%**다. 이는 학습 조건의 개발용 순위이며, **독립 신규 작성자 일반화나 최종 배포 모델의 선택을 의미하지 않는다.**

[원본 결과 모음](../results/README.md) · [재현 코드](../scripts/README.md) · [Combined 개요](../README.md) · [공통 실험 양식](../../../EXPERIMENT_TEMPLATE.md).

<!-- CMB008_S4S5_FINAL_START -->
**CMB-008 S1~S5 완료:** (256,), alpha=1e-4, LR **0.001**, Early Stopping **False**, OOF **92.214%**, OfficialVal **97.044%**. [S4 결과](하이퍼파라미터_최적화/EXP-CMB-008_08_Stage4_LearningRate_Results.md) → [S5 결과](하이퍼파라미터_최적화/EXP-CMB-008_10_Stage5_EarlyStopping_Results.md) → [최종 결정](하이퍼파라미터_최적화/EXP-CMB-008_11_Final_HP_Decision.md). 최종 Test·배포 미변경.
<!-- CMB008_S4S5_FINAL_END -->

## 발표용 Learning Curve 및 최종 Early Stopping 결정

**EXP-CMB-008 S1~S5 개발용 HP 선정 완료:** Size30/Stroke28/CoM, (256,), alpha=1e-4, LR=.001, **Early Stopping=False**. [발표용 Accuracy·Loss Learning Curve 및 결정 이유](하이퍼파라미터_최적화/EXP-CMB-008_18_S5_Presentation_LearningCurve_ESFalse.md). Training/Validation 간 격차와 Validation Loss 증가로 **과적합 징후는 있으나**, 기존 S5 실제 Writer OOF 및 핵심 오분류 비교에서는 False를 유지하는 것이 적절하다. 4방향 각 5% 미만 목표는 미달, 최종 Test·배포 미사용.

## 09B 강의자료 후속 — Batch Size 실측

- [**CMB-008 S6 Batch Size 200 / 128 / 64 비교 결과**](하이퍼파라미터_최적화/EXP-CMB-008_20_Stage6_Batch_Size_Results.md): 36개 실제 학습, Batch 128의 OOF 정확도가 가장 높았으나 네 방향·취약 클래스 비악화 Gate를 통과하지 못함 → 기존 Batch 200 유지 추천. 원본 S5과 새 S6 기준선의 재현성 차이는 별도로 기록. 최종 HP/모델 배포 변경 없음.

## EXP-CMB-010 — 손글씨 480장 추가 Training 실험 (2026-11-18)

- [**01 품질 검사·데이터 분할 계획**](학습데이터_추가/EXP-CMB-010_01_Input_QA_and_Train_Plan.md): 새 NPZ 480장(7 160/9 160/8 80/B 80)의 직접 구조·육안 검사 후 원본 보존. 미확인 작성자이므로 신규 480장 전체를 하나의 학습용 코호트로 취급.
- 공식 Training 76,800장 + 기존 3 Writer 1,920장의 **78,720장 기준선**과, 여기에 **480장을 추가한 79,200장**을 동일 Batch128·Size30/Stroke28/CoM 설정으로 3 Seed × 4 Writer 비교한다.
- 신규 480장은 학습용으로만 쓰고, 기존 held-out Writer 640장·OfficialVal 19,200장·Original Test·교수님 Test에 추가하지 않는다.
- **이 단계는 클래스별 실제 손글씨 다양성 보강의 최초 통제 실험**이며, 새로운 모델의 개선·배포를 사전에 확정한 것은 아니다.


### CMB-010 실측 결과 — 480장 전체 보강 미채택

[02 추가 Training 24모델 실측](학습데이터_추가/EXP-CMB-010_02_Added_480_Training_Results.md): 기존 78,720장 대비 7/9/8/B 480장을 추가한 79,200장 학습에서 Writer OOF **92.4089%→92.2917%**, OfficialVal **97.0972%→97.0677%**. B→8·D→0 개선과 7↔9·0→D 악화가 동반돼, **원본 보존·추가 480장 전체 채택 보류**. [16클래스 CSV](../results/Targeted_Handmade_480/class_error_comparison.csv). 기존 `main`/배포 가중치는 변경하지 않는다.
