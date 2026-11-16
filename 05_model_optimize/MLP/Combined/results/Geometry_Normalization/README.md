# EXP-CMB-004 — Geometry Normalization: 실제 실험 결과

**상태: 4개 전처리×4-Fold 재학습 16개 모델, 추가 4-way 통제, 결과 집계 완료.** 현재는 **실험 브랜치 검토 자료**이며 사용자 승인 전까지 main에 병합하지 않는다.

- **[실험 전 고정한 사전등록 기준](../../logs/가설B/EXP유001_Geometry_Normalization.md)**: Accuracy 하한 없음. Size와 CoM의 **효과 차이와 이유를 설명할 수 있는가**로 판단.
- **[최종 실험일지와 증거 판정](../../logs/가설B/EXP유001_Geometry_Normalization_Results.md)**: Size와 CoM의 기하 변화, 0/D·8/B 방향별 오류, 재학습 효과와 공식 Validation 손실까지 설명.

## 재현 조건 및 실제 수치

| 방법 | OOF Accuracy | Macro F1 | 순증가 정답 수 |
|---|---:|---:|---:|
| EXP-CMB-001 원본 | 81.2500% | 0.8120 | — |
| BBox Center(참고 대조) | 80.3125% | 0.8036 | −24 |
| **BBox Size** | **80.7813%** | 0.8096 | −12 |
| **CoM Center** | **85.8594%** | 0.8593 | +118 |
| **Size+CoM** | **88.9453%** | 0.8898 | +197 |

- 공식 Training 76,800장에만 근거한 **BBox long-side 중앙값=20px**; 전경 threshold=32, 목표 CoM=(13.5,13.5).
- 각 Fold 공식 Train 76,800 + 다른 작성자 Hand-made 1,920, Held-out 640; 완전히 새 MLPClassifier(random_state=42), PixelScaler255, 동일 Train/Dev/OOF 변형.
- 공식 Original Test·교수님 Test 미사용. 기존 EXP-CMB-001~003 모델과 원본 로그/CSV 불변.
- **Hand-made OOF 최고 Size+CoM도 공식 Validation은 네 Fold 모두 하락**하므로, 후속 모델 선정·배포 근거로 바로 사용하지 않는다.

## 모든 비교 자료

| 파일 | 내용 |
|---|---|
| [SUMMARY.md](SUMMARY.md) | 자동 집계 OOF, 주요 오류쌍 및 기하 변화 |
| [method_comparison.csv](method_comparison.csv) | Baseline/4방법 Accuracy·Macro F1·교정·신규·시간 |
| [writer_comparison.csv](writer_comparison.csv) · [group_comparison.csv](group_comparison.csv) | 작성자별·숫자/문자별 성능 |
| [main_error_pairs.csv](main_error_pairs.csv) | 0→D, D→0, 8→B, B→8 전후 혼동 |
| [geometry_comparison.csv](geometry_comparison.csv) | Size·CoM의 실제 정규화량/분포·손실 |
| [mechanism_by_stratum.csv](mechanism_by_stratum.csv) | 작성자/클래스/오류전환별 Size·CoM 및 Logit Margin |
| [factorial_interaction.csv](factorial_interaction.csv) | E−C−D+A (서로 다른 재학습 모델의 설명적 대비) |
| [CROSS_4WAY_SUMMARY.md](CROSS_4WAY_SUMMARY.md) | 기존 모델/원본→기존 모델/변형→재학습 모델/변형 비교 |
| [cross_4way_summary.csv](cross_4way_summary.csv) · [cross_4way_writers.csv](cross_4way_writers.csv) | 4-way 전체·작성자별 |
| [cross_4way_digit_letter.csv](cross_4way_digit_letter.csv) · [cross_4way_pair_errors.csv](cross_4way_pair_errors.csv) | 4-way 숫자·문자, 혼동쌍 |
| [cross_4way_focus_margins.csv](cross_4way_focus_margins.csv) · [cross_4way_paired_predictions.csv](cross_4way_paired_predictions.csv) | 4-way 동일 이미지 Logit Margin·원본 대응 예측 |
| [AGGREGATE_MANIFEST.json](AGGREGATE_MANIFEST.json) | 16개 신규 모델 SHA·입력 및 실행 내역 |
| [accuracy_comparison.png](accuracy_comparison.png) | 전체 OOF 비교 시각화 |

## 각 방법의 독립 결과

- [bbox_center/](bbox_center/run_manifest.json): 기존 Normal의 Bounding Box 중심 정렬 재현용 참고 대조.
- [bbox_size/](bbox_size/run_manifest.json): Bounding Box 긴 변을 20px로 정규화.
- [com_center/](com_center/run_manifest.json): Center of Mass 목표 정렬.
- [size_then_com/](size_then_com/run_manifest.json): 먼저 Size 정규화, 이어서 CoM 정렬.

각 하위 폴더에는 **작성자별·공식 Validation 지표, 학습 Loss Curve, 핵심 혼동쌍, 기하 특징, 대응 예측, Viridis OOF/작성자별 CM, 원본/변형 이미지 Grid, 재학습 모델 4개 ZIP과 개별 SHA**가 존재한다.

## 실행과 코드

- [4-Fold 학습 코드](../../scripts/exp_cmb004_geometry_normalization.py)
- [방법 간 결과 비교 코드](../../scripts/exp_cmb004_compare.py)
- [이미지 변화 vs 재학습 가중치 분리 코드](../../scripts/exp_cmb004_cross_diagnostic.py)
- [학습 성공 GitHub Actions #43747924168](https://github.com/ryu980920/Machine_Learning/actions/runs/43747924168)
- [추가 대조 성공 GitHub Actions #43748490480](https://github.com/ryu980920/Machine_Learning/actions/runs/43748490480)

**실험 목적 달성**은 설명 가능성의 판정이며, 원본 Test에 대한 일반화 또는 실제 배포용 성능 채택을 뜻하지 않는다.
