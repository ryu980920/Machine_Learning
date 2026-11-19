# MLPClassifier / Combined — 실측 결과

## 최종 모델 구성 및 별도 438장 검증 결과 (2026-11-19)

- [**Final_Model — 최종 기존 Combined 선정 결과·동결 Manifest·438장 7조건 CSV**](Final_Model/README.md)
- [**CMB011 438장 실제 저장 모델 84개 평가 로그**](../logs/학습데이터_추가/EXP-CMB-011_03_External438_and_Selection.md)
- 고정 조건은 기존 Combined `CMB011/base`이며 신규 480장 Training 제외. **최종 설정 고정과 새로운 단일 추론 가중치 생성/배포는 구분**한다.


실제 수행한 실험의 지표·오류 이미지·Confusion Matrix·학습/추론 시간·메타데이터를 각각의 폴더에서 관리하고, [logs](../logs/README.md)와 [scripts](../scripts/README.md)에 연결한다.

| 결과 폴더 | 관련 실험 | 상태 |
|---|---|---|
| [EXP_CMB_001/](EXP_CMB_001/README.md) | [EXP유001 Baseline](../logs/EXP-CMB-001_Baseline_4Writer_OOF.md) | **원본 학습·OOF 결과 보존** |
| [Error_Analysis/](Error_Analysis/README.md) | [초기분석 CMB-002](../logs/초기분석/EXP-CMB-002_Error_Analysis.md) | **진단 실행 완료** — 가설 B 우선, A 보류 |
| [Input_Sensitivity/](Input_Sensitivity/README.md) | [초기분석 CMB-003](../logs/초기분석/EXP-CMB-003_Input_Sensitivity.md) | **실측 완료** — 10개 변형 OOF 및 공식 Validation, 후속 후보는 얇게25% |
| [Geometry_Normalization/](Geometry_Normalization/README.md) | [가설 B EXP유001](../logs/가설B/EXP-CMB-004_02_Final_Geometry_Normalization.md) | **4방법×4-Fold 새 학습 + 4-way 설명 대조 완료**, 공식 Validation 손실로 최종 채택 보류 |

기존 EXP-CMB-001의 모델 ZIP·OOF·Training Loss·Viridis CM와 학습/검증 분할은 바꾸지 않는다. 후속 분석은 새 실험 코드와 파일만 추가하며, OOF 개발평가를 독립 Test 성능으로 보고하지 않는다.


## 2026-11-18 후속 실측 자료

- [EXP-CMB-008 S6 438장 후속 집계 CSV](Hyperparameter_Optimization/S6_Batch_Size/External438_followup_summary.csv) · [21번 후속 로그](../logs/하이퍼파라미터_최적화/EXP-CMB-008_21_Stage6_External438_Followup_Results.md): Batch 128 개발용 우선 후보를 신규 438장과 S6 재실행으로 비교. 438장 원본 NPZ·이미지는 공개 저장소에 게시하지 않는다.
- [EXP-CMB-009 추가 전처리 결과](Sequential_Preprocessing/README.md) · [원본 정량 CSV](Sequential_Preprocessing/standalone_table.csv) · [조합 후보 선정 JSON](Sequential_Preprocessing/selection.json): Local Warp·Rotation·Shear 72모델 비교 완료, 단독 가드레일 통과 조건 없음. 모델 배포는 변경하지 않는다.
