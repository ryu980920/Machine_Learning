# EXP유004 — Automatic Centering: 실제 실행 산출물

**상태: 2026-11-15 KST 실행·검증 완료, 연구 결과 보존용으로 사용자 승인 후 정식 배포.** 원래 실험은 `review-only`로 실행했으므로 `SUMMARY.md`와 `exp004_run_metadata.json`에는 원본 생성 당시의 `review-only` 상태가 그대로 남아 있다. 원본 수정 없이 ZIP에서 복사·검증했으며, **최종 Pipeline 채택은 별개**다.

- [연구 실험 로그 — EXP유004_Automatic_Centering.md](../../logs/EXP유004_Automatic_Centering.md)
- [학습·평가 코드](../../scripts/Automatic_Centering_Experiment.py) · [stateless 정렬 Transformer](../../scripts/Automatic_Centering_Transformer.py)
- [GitHub Actions 원본 성공 실행 #43638979287](https://github.com/ryu980920/Machine_Learning/actions/runs/43638979287), Artifact **14131562043** (`normal-mlp-exp004-automatic-centering-review`)

## 성능 요약

| 모델 | 공식 Validation | Hand-made | Hand-made 기존 대비 |
|---|---:|---:|---:|
| Frozen Normal MLP | **97.3177%** | 62.8516% | 기준 |
| **Center of Mass** | 96.9167% | **77.1094%** | **+14.2578%p** |
| Bounding Box Center | 97.3073% | 61.6016% | −1.2500%p |

**중요한 상충 결과:** Hand-made 숫자는 **56.31% → 83.19%**, 문자는 **73.75% → 66.98%** (Normal→Center of Mass). 4명의 작성자 데이터를 이미 여러 차례 분석했으므로 독립 Test 개선을 주장하거나 후보를 곧바로 채택하지 않는다.

## 사용자 승인 후 추가 배포: CoM 판단근거 후속 분석

EXP유004의 기존 결과 10개를 유지한 채, **같은 Normal MLP를 고정하고 이미지만 CoM 이동한 조건**, **같은 CoM 입력에서 CoM 재학습 모델로 변경한 조건**을 분리한 후속 분석을 추가했다.

- [후속 연구 로그](../../logs/EXP유004_CoM_판단근거_후속분석.md)
- [후속 실측 원본 9개, 그래프 및 해시](CoM_Decision_Review/README.md)
- [후속 실행 기록 #43686466736](https://github.com/ryu980920/Machine_Learning/actions/runs/43686466736)
- CoM 이동만으로도 숫자 **+25.69%p**, 문자 **−21.35%p**로 달라졌다. **결과 배포는 신규 전처리 채택과 무관**하다.

## 원본 파일 10개

| 파일 | 내용 | 검증 SHA-256 앞 12자 |
|---|---|---|
| [SUMMARY.md](SUMMARY.md) | 성공 실행 중 생성한 실제 결과 요약 (생성 당시 review-only 상태 보존) | `a218e64f267a` |
| [exp004_group_metrics.csv](exp004_group_metrics.csv) | Accuracy·Macro F1·Precision·Recall·오류 교정: 전체/16개 클래스/작성자/숫자·문자, **144행** | `1810fa279353` |
| [exp004_predictions.csv.gz](exp004_predictions.csv.gz) | 공식 Validation 19,200 + Hand-made 2,560, **21,760행**; 세 모델의 같은 이미지별 정답·예측 | `0dbdbe369a73` |
| [exp004_confusions.csv](exp004_confusions.csv) | 공식/손글씨 × 세 모델 × 16×16 혼동행렬, **1,536행** | `5bf4cfd74f97` |
| [exp004_typical_confusions.csv](exp004_typical_confusions.csv) | 5→F, B→8, 0→D 등 대표 오분류쌍, **36행** | `ff5607175a05` |
| [exp004_alignment_stats.csv](exp004_alignment_stats.csv) | Train/Validation/Hand-made × 중심 방법별 오차·이동·획손실, **6행** | `66ff7aa03dd3` |
| [accuracy_comparison.png](accuracy_comparison.png) | 두 데이터셋에 대한 세 모델 정확도 그래프 | `64201b7036cf` |
| [exp004_run_metadata.json](exp004_run_metadata.json) | 기준 모델·데이터 SHA, 동일 학습 조건, 정렬 규칙, 2개 학습 시간과 환경, Test 미사용 플래그 | `265899d0c35d` |
| [exp004_mass.joblib](exp004_mass.joblib) | Center of Mass **학습 완료 후보 Pipeline** | `3c1aac362d75` |
| [exp004_bbox.joblib](exp004_bbox.joblib) | Bounding Box **학습 완료 후보 Pipeline** | `09976f913cf4` |

**재현/주의사항:** Python 3.13.18, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.7. 모델을 불러올 때 `Automatic_Centering_Transformer.py`와 기존 `pixel_transform.PixelScaler255`가 import 가능한 환경이 필요하다. **신뢰하지 않는 joblib 파일을 역직렬화하면 안 된다.** 원본 Test, 교수님 Test는 읽지 않았고 Hand-made를 학습에 사용하지 않았다.

실험 실행과 원본 결과 검증은 성공한 GitHub Actions에 남겨 두었으며, 기존 EXP유001·002·003과 baseline은 변경하지 않았다.
