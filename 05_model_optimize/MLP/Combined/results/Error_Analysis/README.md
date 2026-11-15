# EXP-CMB-002 — Combined Error Analysis 실제 검증 결과

**실험 상태:** 실행·검증 완료, **가설 B 우선 / 가설 A 별도 진행 보류**. 현재 결과는 실험 브랜치의 검토용 자료이며, 사용자 승인 전까지 main에 병합하지 않는다.

[전체 실험일지와 근거 판단](../../logs/초기분석/EXP유002_Error_Analysis.md) · [실행 스크립트 1](../../scripts/exp_cmb002_evidence.py) · [실행 스크립트 2](../../scripts/exp_cmb002_followup.py)

**실행 성공:** [GitHub Actions #43742381093](https://github.com/ryu980920/Machine_Learning/actions/runs/43742381093), [#43742905999](https://github.com/ryu980920/Machine_Learning/actions/runs/43742905999), [#43743082289](https://github.com/ryu980920/Machine_Learning/actions/runs/43743082289).

## 재현 조건

- [EXP-CMB-001 원본](../EXP_CMB_001/README.md)의 2,560건 OOF 완전 재현, 기존 Normal 1개와 Combined 보류 작성자 모델 4개만 사용.
- 기존 모델 SHA 및 공식·손글씨 원본 SHA 재검증, **새 MLP 학습 0회**, 공식 Original Test·교수님 Test 미사용.
- 원본의 정답/오답 480개와 교정 540개·신규 62개를 다시 분류. 0/D/8/B의 실제 클래스 이미지 640장을 집중 분석.
- Writer × Class 층화 순열 검정 2,500/3,000회, 탐색적 집단 검정 2,000회, 오류쌍별 임계값 안정성 4,000회. 다중 비교는 Benjamini–Hochberg로 보정.
- 픽셀 교환 시 0–255 전체 밝기 히스토그램/총 밝기/Bounding Box/전경 수 보존, CoM 이동 <=0.3px; 원본 이미지 61개에서 유효한 대조.

## 결과 파일

| 파일 | 내용 |
|---|---|
| [SUMMARY.md](SUMMARY.md) · [FOLLOWUP_SUMMARY.md](FOLLOWUP_SUMMARY.md) | 자동 생성된 1차·보강 실측 요약 |
| [evidence_summary.json](evidence_summary.json) · [FOLLOWUP_EVIDENCE.json](FOLLOWUP_EVIDENCE.json) | 수치 원본·검증 조건·한계 기록 |
| [image_features.csv](image_features.csv) | 2,560장 전체 중심·너비·Ink Ratio·명암·획 굵기 대리지표 및 threshold 32/64/128 |
| [morphology_features.csv](morphology_features.csv) | 2,560장 형태/공간 질량 분포 11개 지표 |
| [writer_metrics.csv](writer_metrics.csv) · [per_class_metrics.csv](per_class_metrics.csv) | 작성자별·16클래스별 성능 |
| [error_transitions.csv](error_transitions.csv) | Normal→Combined 상세 정답·오답 전환 |
| [feature_associations.csv](feature_associations.csv) · [morphology_associations.csv](morphology_associations.csv) | 640장 전체의 Writer/Class 층화 순열 검정 |
| [feature_comparison.csv](feature_comparison.csv) · [targeted_writer_class_tests.csv](targeted_writer_class_tests.csv) | 작성자×실제 클래스 정오답 차이와 탐색적 90개 비교 |
| [pair_specific_threshold_sensitivity.csv](pair_specific_threshold_sensitivity.csv) | 43번 8→B, 41번 D→0, 43번 0→D, 41번 B→8의 직접 오답·정답 및 threshold 안정성 비교 |
| [margin_ig_diagnostics.csv](margin_ig_diagnostics.csv) · [region_ablation_controls.csv](region_ablation_controls.csv) | Signed IG 89장×2모델, 고정 영역 제거 vs 동일 픽셀 수 무작위 제거 |
| [pixel_swap_trials.csv](pixel_swap_trials.csv) · [pixel_swap_per_image.csv](pixel_swap_per_image.csv) · [topology_contrasts.csv](topology_contrasts.csv) | 밝기·Bounding Box 보존 픽셀 교환 원본/이미지 단위 대조/95% Bootstrap |

## 실제 이미지·Confusion Matrix

- [0/D 실제 정오답 이미지 Grid](figures/error_grid_0D.png) · [8/B 실제 정오답 이미지 Grid](figures/error_grid_8B.png)
- [pooled Count CM](figures/writer_confusion_matrices/pooled_count.png) · [pooled Row-normalized CM](figures/writer_confusion_matrices/pooled_normalized.png)
- 작성자별 Count/Normalized CM: [41](figures/writer_confusion_matrices/writer_41_count.png) · [42](figures/writer_confusion_matrices/writer_42_count.png) · [43](figures/writer_confusion_matrices/writer_43_count.png) · [44](figures/writer_confusion_matrices/writer_44_count.png).

**판정:** 중심/너비/잉크 특성이 특히 43번 8→B·41번 D→0에서 정답과 관련된다는 근거는 확보. 형태 인접성만 의도적으로 크게 훼손한 픽셀 교환에서 대조군 대비 추가 영향은 검증되지 않았다. **가설 B만 다음 우선 실험 대상으로 선택하고 가설 A는 조건부 보류**한다.

이 결과는 **개발용 Hand-made의 탐색 연구**이며, 유의한 정오답 차이를 외부 일반화 성능이나 인과적 전처리 효과로 표현하지 않는다.
