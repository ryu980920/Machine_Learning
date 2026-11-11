# EXP유003 — 새 42번 원본 기준 Hand-made Rotation 평가 결과

[EXP유003 실험일지](../../logs/EXP유003_Handmade_방향보정_성능평가.md) · [2026-11-11 전체 재실행 검증](REVALIDATION_20261111.md).

## 데이터·평가 조건

- 5개 **기존 학습 모델**: SVC(RBF), HistGradientBoosting, MLP, Extra Trees, KNN. 모델을 재학습하지 않았다.
- 현재 표준 **41·42·43·44번 각 640장, 총 2,560장**. 41·44번에만 회전 보정 전·후 차이가 있다. **42번은 새 Grayscale 원본(256단계) 그대로**, 43번도 무회전.
- 결과는 2026-11-09 검토본에서 일부 기존 예측을 재사용해 집계했으나, **2026-11-11에는 5×4×2 총 40회 새 추론으로 전체 재검증**했고 모든 결과가 일치했다.
- 기존 `rotation_review_provenance.json`은 **최초 검토본의 예측 출처 기록**으로 보존한다. **새 재실행의 근거는 `REVALIDATION_20261111.md`**이며, 두 실행의 provenance를 혼용하지 않는다.

## 수치·시각자료

- [rotation_before_after_5models.csv](rotation_before_after_5models.csv): 5개 모델 전체 Accuracy/Macro F1, 전·후 정답 수·증가폭.
- [rotation_before_after_by_writer.csv](rotation_before_after_by_writer.csv): 41·42·43·44번 작성자별 전후 결과(**42·43은 동일 값**).
- `rotation_before_predictions.csv`, `rotation_after_predictions.csv`: 전후 각 **12,800건**, 총 **25,600건**. 새 재실행과 키 기준 동일함을 검증.
- `confusion_matrix_{svc,hgb,mlp,extra_trees,knn}_{before,after}.csv/.png`: **총 10개** 16×16 Confusion Matrix. 실제 레이블은 행, 예측 레이블은 열. 각 Matrix 총 2,560, 클래스별 160장.
- `rotation_confusion_long.csv`, `rotation_per_class.csv`: 전체 클래스별 혼동 수·Precision/Recall/F1.
- `rotation_accuracy_before_after.png`: 전체 Accuracy 그래프. 재실행 수치가 완전히 일치하여 기존 동일 데이터 그래프를 유지.
- **재현 코드:** [run_rotation_comparison.py](../../scripts/experiment3_handmade/run_rotation_comparison.py) (신뢰된 모델 ZIP이 별도로 필요).

**역사 기록 보존:** 2026-11-09 최초 집계에서는 저장된 과거 예측값 일부를 재사용했으므로, 2026-11-11 **독립 전체 재실행**과 정확히 구분해 기록한다. 원본 EXP유002 8절과 구 42번 EXP유003은 [archive 폴더](../../logs/archive/EXP유003_20261109_구42번_원문.md)에 보존한다.
