# EXP유002 추가 비교 — 새 42번 기준 Rotation 보정 전·후

- 본 폴더의 **모델 5개×전후 각각 2,560장** 데이터와 16×16 Confusion Matrix 10개는 [EXP유002 실험일지 8절](../../logs/EXP유002_5개모델_Normal_Baseline.md)을 뒷받침한다.
- `rotation_review_provenance.json`에 **41·43·44번 기존 검증 예측 재사용, 새 42번 신규 추론**의 출처를 정확히 구분하였다. 검토본의 결과는 모델 5개×2,560장을 모두 독립 전량 재실행한 결과가 아니다.
- `rotation_before_after_5models.csv`: 5개 모델별 Accuracy, Macro F1, 정답 수 및 변화폭.
- `rotation_before_after_by_writer.csv`: 작성자 4명별 전후 지표.
- `rotation_before_predictions.csv`, `rotation_after_predictions.csv`: 모델별·작성자별·이미지별 예측값.
- `confusion_matrix_{svc,hgb,mlp,extra_trees,knn}_{before,after}.csv/.png`: 모델별 전후 16×16 Confusion Matrix. 실제 레이블은 행, 예측 레이블은 열.
- 42·43번은 전후 동일 이미지. 41·44번에만 **반시계 90° Rotation 보정** 적용.
- 전체 모델 재실행 코드는 [`run_rotation_comparison.py`](../../scripts/experiment3_handmade/run_rotation_comparison.py). 모델 joblib ZIP은 로컬 별도 보관. 스크립트 실행은 비신뢰 모델 직렬화 파일을 불러오지 않도록 주의한다.
