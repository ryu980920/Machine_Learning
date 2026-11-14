# EXP유002 — Input_Sensitivity 실제 실험 결과 (Normal MLP)

**실험 범위:** 입력 위치(8)·밝기(4)·획 굵기(4)·크기(4) 총 20개 변형 + 원본(1). 고정된 MLP Baseline에 **공식 Validation 19,200장과 개발용 Hand-made 2,560장**을 적용하여 전체 **456,960행** 예측하였다. **공식 Test 및 Hand-made Training 미사용**, 모델 개선/튜닝이 아닌 민감도 진단.

- [실험 로그 — EXP유002_Input_Sensitivity.md](../../logs/EXP유002_Input_Sensitivity.md)
- [실험 원본 코드 — Input_Sensitivity.py](../../scripts/Input_Sensitivity.py)
- [정량 비교 그래프 재생성 코드](../../scripts/plot_input_sensitivity.py)
- [성공한 GitHub Actions 실행 #43629898144](https://github.com/ryu980920/Machine_Learning/actions/runs/43629898144) — 2026-11-15 KST; `normal-mlp-input-sensitivity-stage3` Artifact

## 원본 실행 검증

| 검증 항목 | 값 |
|---|---:|
| Official Validation 원본 Accuracy | **97.3177083%** |
| Hand-made 원본 Accuracy | **62.8515625%** |
| 기존 공식 CM / Hand-made 예측 불일치 | **0 / 0** |
| 이전 픽셀 기여도 사례 98장 예측 차이 | **0** |
| 원본 포함 조건 | **21개** |
| 전체 예측 | **456,960행** |
| 집중 분석 | **106장 × 21조건 = 2,226행** |

## 보존 파일

| 파일 | 내용 |
|---|---|
| [SUMMARY.md](SUMMARY.md) | 성공 실행의 자동 요약 및 상위 민감도 조건 |
| [stage3_run_metadata.json](stage3_run_metadata.json) | 데이터·도구 버전/변형값/재현 검사/학습 조건 |
| [stage3_sensitivity_summary.csv](stage3_sensitivity_summary.csv) | 공식·손글씨 × 21조건의 Accuracy, 예측 변경률, 오류 교정/발생 |
| [stage3_per_class.csv](stage3_per_class.csv) | 데이터별 16개 클래스×21조건 |
| [stage3_per_writer.csv](stage3_per_writer.csv) | Hand-made 작성자 41~44번×21조건 |
| [stage3_error_pairs.csv](stage3_error_pairs.csv) | 선택 오류쌍의 감소·악화 및 정답 전환 건수 |
| [stage3_focus_106.csv](stage3_focus_106.csv) | 핵심 106개 샘플의 모든 21개 입력 조건별 분류 |
| [stage3_predictions.csv.gz](stage3_predictions.csv.gz) | **원본 포함 전체 456,960행**의 이미지별 결과 — gzip 압축 CSV, 원래 모델의 학습 데이터는 포함하지 않음 |
| [normal_mlp_refit_stage3.joblib](normal_mlp_refit_stage3.joblib) | 성공한 실행에서 사용한 모델 체크포인트. **신뢰할 수 있는 파일만 역직렬화** |
| [stage3_example_grid.png](stage3_example_grid.png) | 오류 이미지 원본/변형에 대한 예측 비교 |
| [stage3_probability_curves.png](stage3_probability_curves.png) | 대표 오류 샘플의 조건별 클래스 확률 |
| [STAGE3_Accuracy_Change_Comparison.png](STAGE3_Accuracy_Change_Comparison.png) | 공식 vs Hand-made 정확도 증감 비교 |
| [STAGE3_Digit_Letter_Tradeoff.png](STAGE3_Digit_Letter_Tradeoff.png) | 손글씨 숫자·문자 각각의 정확도 변화 |
| [STAGE3_Per_Writer.png](STAGE3_Per_Writer.png) | 손글씨 작성자별 민감도 |

## 결과 해석 규칙

- **Flip Rate**는 예측 클래스가 변형 전과 달라진 비율이며, 오류가 고쳐진 비율과 다르다.
- **Error Correction**은 오답→정답, **Error Induction**은 정답→오답을 의미한다. 둘을 함께 계산해야 순수 Accuracy 변화를 해석할 수 있다.
- 위치·크기·획 굵기 입력 조작은 기하학적 형태도 변형할 수 있다. **형태를 전혀 사용하지 않는다는 근거는 아니다.**
- 조건별 수치는 현재 개발용 평가 결과이다. **최종 전처리 또는 독립 Test 성능을 제시하지 않는다.**
- 재실행 시 과거 내부 가중치까지 완전히 같다는 보장은 없으므로, 보관한 `.joblib` 및 `stage3_run_metadata.json`으로 동일 실행의 출처를 확인한다.

**재현:** 5개 공용 Baseline 파일은 변경하지 않는다. 저장소 루트에서 아래를 실행한다.

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python 05_model_optimize/MLP/Normal/scripts/Input_Sensitivity.py \
  --out 05_model_optimize/MLP/Normal/results/Input_Sensitivity --batch-size 1024

python 05_model_optimize/MLP/Normal/scripts/plot_input_sensitivity.py \
  --result-dir 05_model_optimize/MLP/Normal/results/Input_Sensitivity
```
