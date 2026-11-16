# MLPClassifier / Combined — 실측 결과

실제 수행한 실험의 지표·오류 이미지·Confusion Matrix·학습/추론 시간·메타데이터를 각각의 폴더에서 관리하고, [logs](../logs/README.md)와 [scripts](../scripts/README.md)에 연결한다.

| 결과 폴더 | 관련 실험 | 상태 |
|---|---|---|
| [EXP_CMB_001/](EXP_CMB_001/README.md) | [EXP유001 Baseline](../logs/EXP-CMB-001_Baseline_4Writer_OOF.md) | **원본 학습·OOF 결과 보존** |
| [Error_Analysis/](Error_Analysis/README.md) | [초기분석 CMB-002](../logs/초기분석/EXP-CMB-002_Error_Analysis.md) | **진단 실행 완료** — 가설 B 우선, A 보류 |
| [Input_Sensitivity/](Input_Sensitivity/README.md) | [초기분석 CMB-003](../logs/초기분석/EXP-CMB-003_Input_Sensitivity.md) | **실측 완료** — 10개 변형 OOF 및 공식 Validation, 후속 후보는 얇게25% |
| [Geometry_Normalization/](Geometry_Normalization/README.md) | [가설 B EXP유001](../logs/가설B/EXP-CMB-004_02_Final_Geometry_Normalization.md) | **4방법×4-Fold 새 학습 + 4-way 설명 대조 완료**, 공식 Validation 손실로 최종 채택 보류 |

기존 EXP-CMB-001의 모델 ZIP·OOF·Training Loss·Viridis CM와 학습/검증 분할은 바꾸지 않는다. 후속 분석은 새 실험 코드와 파일만 추가하며, OOF 개발평가를 독립 Test 성능으로 보고하지 않는다.
