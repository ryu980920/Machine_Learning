# EXP유002 — Combined Error Analysis (실험 계획)

> **실험 ID:** EXP-CMB-002 · **상태: 계획 수립 / 미실시**. 아래 수치는 완료된 [EXP-CMB-001 Baseline](../EXP유001_Combined_MLP_4Fold_CV_Baseline.md)의 원본 OOF를 재집계한 값이다. 이미지 특징 추출·IG 계산·후속 모델 추론은 아직 수행하지 않았다.

## 1. 문제 발견

- Combined Hand-made OOF: **2,080/2,560장 정답, 81.2500%**, 오답 480장.
- `0→D` 30장, `D→0` 29장: 합계 59장. `8→B` 27장, `B→8` 26장: 합계 53장.
- `0→D`는 작성자 43번에서 **28/30장**, `D→0`은 41·42번에서 **27/29장** 발생했다.
- Normal 오답→Combined 정답 **540장**, Normal 정답→Combined 오답 **62장**.
- 기존 실측 근거: [OOF_predictions.csv](../../results/EXP_CMB_001/OOF_predictions.csv), [metrics_summary.csv](../../results/EXP_CMB_001/metrics_summary.csv).

## 2. 원인 가설

- **가설 B(공간·명암):** 동일 클래스에서도 작성자별 중심 X/Y, Bounding Box, 잉크 비율, 밝기, 획 굵기 차이가 오류와 연관되어 있다.
- **가설 A(판단 근거):** 특히 `0/D`, `8/B`의 유사 획·연결 구조와 MLP 입력 픽셀의 조합이 경쟁 클래스 간 점수에 영향을 준다.
- **학습 효과 분리:** 새로 발생한 오답을 Normal에서도 지속되던 오답과 구별하여 학습 후 판단 변화의 영향을 관찰한다.

## 3. 통제 및 수행 계획

1. 보관된 [experiment_manifest.json](../../results/EXP_CMB_001/experiment_manifest.json) 해시·Fold와 [OOF_predictions.csv](../../results/EXP_CMB_001/OOF_predictions.csv) 2,560행·원본 CM·Accuracy의 일치를 검증한다.
2. `ml26_41.npz`~`ml26_44.npz`의 실제 이미지와 `writer`, `index_in_writer`를 결합한다. 작성자별 실제 16클래스×40장 구성 유지.
3. 모든 Hand-made 이미지의 Center Offset, Bounding Box 크기·종횡비, Ink Ratio, 평균·총 밝기 및 획 굵기 추정치를 기록한다. 전경 Threshold 32를 기본으로 하고 64·128에 대한 안정성을 확인한다.
4. 정답·오답·교정·신규 오류 집단을 **작성자·실제 클래스별로 구분**하여 비교한다. `0↔D`, `8↔B`와 대응하는 정답 사례의 실제 이미지 Grid를 만든다.
5. [trained_models.zip](../../results/EXP_CMB_001/trained_models.zip)의 **Normal 1개와 작성자별 보류 Combined 4개**를 고정하여, 동일 이미지에서 클래스 간 Logit Margin 및 대표 Signed Integrated Gradients를 계산한다. 각 작성자는 해당 작성자를 보류한 Combined 모델만 사용한다.
6. 구조·위치 특징의 연관성을 해석하고 [가설 A](../가설A/README.md)·[가설 B](../가설B/README.md)의 후속 단일 변수 검증 질문을 도출한다.

**금지:** 이 실험에서 모델 재학습, 보류 작성자의 Training 편입, 최종 Test 사용, 새 전처리 채택을 수행하지 않는다. 복수 작성자를 합산한 차이만으로 원인을 확정하지 않는다.

## 4. 산출물 계획 (미생성)

- [결과 폴더 안내](../../results/Error_Analysis/README.md) — 실제 CSV/PNG는 **아직 없음**.
- 계획 산출물: 전체/작성자별 Viridis CM, 오분류 원본 Grid, `image_features.csv`, `error_transitions.csv`, `writer_metrics.csv`, `per_class_metrics.csv`, `feature_comparison.csv`, `margin_ig_diagnostics.csv`, `run_manifest.json`.
- 완료 후에만 실측표·이미지·실행 ID·결론을 이 로그에 추가하며, 기존 EXP-CMB-001 파일은 덮어쓰지 않는다.

## 5. 현재 판단 및 다음 실험

지금은 **오류 위치와 전환 유형까지 확인된 상태**다. 공간/형태 가설은 미검증이므로 원인을 단정하지 않는다. 원인 후보를 정리한 후 [EXP유003 — Input Sensitivity](EXP유003_Input_Sensitivity.md)에서 모델을 고정한 조작 실험으로 검증한다.
