# MLPClassifier / Combined — 실측 결과

각 실험의 **실제 실행으로 생성된** Accuracy·Precision·Recall·F1 CSV, Confusion Matrix, 이미지 Grid, Learning Curve, 학습/추론 시간 및 재현 메타데이터는 이 폴더에서 실험별로 관리한다. 해당 [실험일지](../logs/README.md) 및 [코드](../scripts/README.md)를 연결한다.

| 결과 폴더 | 연결 실험 | 상태 |
|---|---|---|
| **[EXP_CMB_001/](EXP_CMB_001/README.md)** | [EXP유001 Baseline](../logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md) | **실측 결과 보존 완료** |
| **[Error_Analysis/](Error_Analysis/README.md)** | [초기분석 EXP유002](../logs/초기분석/EXP유002_Error_Analysis.md) | **계획용 README만 존재 / 미실시** |
| **[Input_Sensitivity/](Input_Sensitivity/README.md)** | [초기분석 EXP유003](../logs/초기분석/EXP유003_Input_Sensitivity.md) | **계획용 README만 존재 / 미실시** |

### 기존 Baseline 보존

`EXP_CMB_001/`의 실제 OOF 2,560개 예측 및 지표, 분할 인덱스·manifest, 학습 모델 ZIP, Viridis **Count CM 2개 + Row-normalized CM 2개**, Training Loss PNG 1개는 그대로 유지한다. 원본 OOF에서 재계산 가능한 표는 결과 복제를 최소화하며 출처를 명시한다.

**예정 결과 폴더를 실측 결과로 간주하지 않는다.** 후속 실험이 성공적으로 실행되고 CSV·PNG·해시·모델/입력/분할 검증이 끝난 경우에만 실제 산출물을 추가한다. 기존 EXP-CMB-001의 파일을 변경하거나 덮어쓰지 않는다.
