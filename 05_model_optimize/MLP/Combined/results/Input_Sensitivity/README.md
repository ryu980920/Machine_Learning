# EXP-CMB-003 — Combined Input Sensitivity 실측 결과

**상태: 실험 실행 완료 · 연구 목적 판정 완료 · 새 모델 학습 없음.**

[완성된 실험일지](../../logs/초기분석/EXP-CMB-003_Input_Sensitivity.md) · [실행 코드](../../scripts/exp_cmb003_input_sensitivity.py) · [성공 Actions #43745629093](https://github.com/ryu980920/Machine_Learning/actions/runs/43745629093)

## 실험 조건과 요약

- 기존 EXP-CMB-001 저장 Normal 1개 + Combined 보류 작성자 4개를 고정. Hand-made OOF 원본 2,560건 100% 재현.
- 원본 + 위치 ±1px(4) + 크기 95/105%(2) + 획 ±25%(2) + 밝기 90/110%(2) = **11조건**.
- 실측: Combined Baseline **81.2500%**, 95% 축소 **83.8672%** (교정132/신규65), 획 얇게25% **82.3047%** (교정50/신규23).
- 95% 축소 순개선은 **작성자 43에 집중**, 공식 개발 Validation은 Fold별 **−0.3750~−0.9375%p**. 얇게25%도 41번과 문자 악화, 공식 개발 Validation Fold별 **−0.0521~−0.2031%p**.
- 다음 재학습 후보 선별 기준에는 **획 얇게25%만 통과**. 실측은 어디까지나 *가중치 고정된 입력 진단*, 신규 작성자에서의 검증이나 재학습 성공을 의미하지 않음.
- 공식 Original Test 및 교수님 최종 Test 미사용. 원본 모델·Baseline CSV·분할·과거 산출물은 변경하지 않음.

## 정량 CSV·재현 자료

| 파일 | 목적 |
|---|---|
| [SUMMARY.md](SUMMARY.md) | 11조건 전체 Combined 비교 및 후속 후보 스크리닝 |
| [oof_metrics.csv](oof_metrics.csv) | Normal/Combined OOF Accuracy·Macro F1·교정·신규·Flip·McNemar/BH |
| [writer_metrics.csv](writer_metrics.csv) | 작성자 41–44별 정확도 및 신규·교정 |
| [digit_letter_metrics.csv](digit_letter_metrics.csv) | 숫자 10개 / 문자 6개 클래스 그룹별 Accuracy·Macro F1 |
| [per_class_metrics.csv](per_class_metrics.csv) | 16개 클래스별 Precision/Recall/F1, 교정·신규 오류 |
| [pair_confusion_changes.csv](pair_confusion_changes.csv) | 0→D/D→0/8→B/B→8 원본 대비 변화 (작성자별 포함) |
| [focus_margin_diagnostics.csv](focus_margin_diagnostics.csv) | 43번 0/D·8/B, 41번 D/0·B/8의 경쟁 Logit 변화 |
| [official_validation_metrics.csv](official_validation_metrics.csv) | 공식 개발 Validation 19,200장, Normal 및 Combined 4개 모델의 조건별 성능·추론 시간 |
| [transform_diagnostics.csv](transform_diagnostics.csv) | 변환 후 픽셀 질량 변화, 비동일 비율, 추론시간 |
| [paired_predictions.csv.gz](paired_predictions.csv.gz) | 원본→변형 이미지별 동일 2,560건의 Normal·Combined 대응 예측 |
| [run_manifest.json](run_manifest.json) | 모델 원본 SHA, 11조건, 선별 기준, 미사용 Test, 환경 버전 |

## 이미지

- [변형 전후 실제 Hand-made 그리드](figures/transform_examples.png)
- [교정 vs 신규 오류 그래프](figures/corrections_vs_induced.png)
- [Combined Baseline Count CM](figures/original_count.png) / [Row-normalized](figures/original_normalized.png)
- [95% 축소 Count CM](figures/scale_095_count.png) / [Row-normalized](figures/scale_095_normalized.png)
- [획 얇게25% Count CM](figures/stroke_thin_25_count.png) / [Row-normalized](figures/stroke_thin_25_normalized.png)
- 다른 8개 변형도 각각 Viridis Count/Normalized CM를 저장함.

**모든 신규 CSV·그림은 EXP-CMB-003만의 산출물이며, EXP-CMB-001 및 EXP-CMB-002 파일은 덮어쓰지 않았다.**
