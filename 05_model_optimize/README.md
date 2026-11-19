# 05_model_optimize — 모델별 최적화 연구 기록

## Combined MLP 최종 선정 — 2026-11-19

**기존 Combined `CMB011/base` 최종 확정·하이퍼파라미터 및 Training 구성 동결.** 신규 480장 추가는 미채택. [최종 선정 근거](MLP/Combined/logs/EXP-CMB-012_Final_Combined_Model_Freeze_20261119.md) · [438장 외부 개발 평가](MLP/Combined/logs/학습데이터_추가/EXP-CMB-011_03_External438_and_Selection.md) · [최종 고정값](MLP/Combined/results/Final_Model/FINAL_COMBINED_FREEZE.json).

공식 Train 76,800장 + 기존 다른 작성자 3명 1,920장의 Writer 보류 Training, MLP (256,), alpha=1e-4, LR=.001, Batch=128, Early Stopping=False, Size30%-Stroke28%-CoM 구성. 학습 가중치를 재학습·교체한 것이 아니라 **최종 선정 설정과 근거를 기록**한 것이다. 독립 Test는 미사용.


5개 모델 각각의 **Normal**과 **Combined**를 독립적으로 최적화하고, 가설·실험 코드·결과물을 같은 규칙으로 축적하는 공간이다.

## 폴더 구조

| 모델 | Normal | Combined |
|---|---|---|
| SVC (RBF) | [Normal](SVC/Normal/README.md) | [Combined](SVC/Combined/README.md) |
| HistGradientBoosting | [Normal](HGB/Normal/README.md) | [Combined](HGB/Combined/README.md) |
| MLPClassifier | [Normal](MLP/Normal/README.md) | [Combined](MLP/Combined/README.md) |
| Extra Trees | [Normal](Extra_Trees/Normal/README.md) | [Combined](Extra_Trees/Combined/README.md) |
| KNeighborsClassifier | [Normal](KNN/Normal/README.md) | [Combined](KNN/Combined/README.md) |

각 조건의 하위 폴더:
- `logs/`: 실험별 Markdown 기록(문제 → 원인 가설 → 선택한 강의 기법 → 비교 조건 → 실측 결과 → 결론 → 다음 실험)
- `scripts/`: 실험을 실제 실행한 Python / Notebook 파일과 실행 방법
- `results/`: 실측 CSV, Confusion Matrix, Learning Curve, 대표 오류 이미지 등 결과 근거

빈 폴더도 GitHub에서 보이도록 각 하위 폴더에 `README.md`를 둔다.

## 공통 연구 흐름

1. **오류 분석:** Original Validation과 개발용 Hand-made의 Confusion Matrix를 확인하고, 클래스를 정규화해 오분류 경향·대표 이미지를 조사한다.
2. **원인 가설:** 관찰한 오류에 대해 기울기·중심·획 굵기·극성·대비·필체 분포 또는 모델의 일반화 문제 등 실제로 검증할 가설을 작성한다.
3. **강의 내용에 근거한 개선:** 전처리 Pipeline, Training-only Augmentation, 데이터 구성, Hyperparameter 등을 선정한다. 변경한 요소와 고정한 요소를 분리한다.
4. **실험과 검증:** 동일한 개발용 Validation 기준에서 Accuracy·Macro/Weighted Precision·Recall·F1, Confusion Matrix, 학습·전처리·추론 시간, 해당 모델에 적합한 Learning Curve를 기록한다.
5. **판단과 반복:** 가설의 채택·보류·실패를 수치·이미지와 연결하고, 다음 실험을 기록한다.

## Normal / Combined 구분

- **Normal:** 공식 MNIST·EMNIST Training만 학습에 사용. 공식 Training 96,000장을 기존 그룹 분할에 따라 Train **76,800장 / Validation 19,200장**으로 나눈 설정을 Baseline 비교 기준으로 유지한다. 개발용 Hand-made는 오분류 분석/검증에 사용할 수 있으나 학습 데이터에 편입하지 않는다.
- **Combined:** 공식 MNIST·EMNIST Training에 별도로 지정한 Hand-made 학습용 데이터를 추가한다. **학습에 사용한 이미지는 검증에 중복 사용하지 않으며**, 독립 작성자 대상 평가가 필요하다면 실제 작성자 ID에 따른 Group CV / Leave-One-Writer-Out 또는 별도 새 작성자 Validation을 사용한다. CV는 Fold마다 처음부터 모델을 다시 학습한다.
- 각 모델은 **Normal/Combined별로 독립 최적 모델과 Pipeline을 선정**한다. 동일 알고리즘·동일 설정에서 데이터 추가 효과만 비교하는 통제 실험은 별도로 구분한다.

**평가 데이터 사용 원칙:** 공식 Original Test 16,000장과 교수님 최종 2차 Test는 모델 선택·튜닝에 사용하지 않는다. 교수님 1차는 모델 고정 후 최초 Test로 평가하며, 성능이 미흡할 때에만 교수님 허용에 따라 개발용 Validation으로 전환할 수 있다. 전환 후의 1차 성능을 독립 Test 결과라 부르지 않는다.

## 기록하는 방법

- 첫 번째 실험부터 각 조건의 `logs/`에 `EXP[작성자코드][번호]_[주제].md` 형태로 저장한다. 동일한 번호가 다른 모델/조건과 겹쳐 혼동되지 않도록 본문에 **모델·Normal/Combined·실행일·데이터 버전·실험 ID**를 명시한다.
- [최적화 전용 간략 양식](EXPERIMENT_TEMPLATE.md) 또는 [기존 공통 상세 실험일지](../03_experiments/EXPERIMENT_TEMPLATE.md)를 사용한다.
- 초기 미학습 Baseline 코드는 [03_experiments/model_optimization](../03_experiments/model_optimization/README.md)을 참조한다. **기존 초기 코드·W01/W02 실험 원본은 수정하거나 이동하지 않는다.**
- 현재 새 폴더의 모든 실험은 **미실시·미등록 상태**다. 과거 측정값을 이 위치에서 새로 실행한 값으로 표시하지 않는다.

[연구 흐름도](../01_project/RESEARCH_FLOW.md) · [데이터 사용 규칙](../02_data/HANDMADE_SPLIT_PLAN.md).


## 2026-11-18 최신 Combined MLP 현황

**Normal**은 공식 MNIST·EMNIST Training만 사용해 학습한다. **Combined**는 같은 공식 Training에 **별도 수집한 Hand-made 손글씨를 학습 데이터로 추가할 수 있는 모델**이다. 현재 Combined Writer-held-out Fold는 공식 76,800장에 다른 작성자 3명의 손글씨 **1,920장**을 추가하고, 보류 Writer 640장은 학습에 넣지 않는다.

[Combined CMB-008 438장 후속 비교](MLP/Combined/logs/하이퍼파라미터_최적화/EXP-CMB-008_21_Stage6_External438_Followup_Results.md)에서 **Batch 128**을 정확도 중심 개발용 우선 후보로 정했고, [CMB-009 추가 전처리 연구](MLP/Combined/logs/가설B/EXP-CMB-009_01_Sequential_Preprocessing_Results.md)에서 Local Warp·Rotation·Shear의 사전 가드레일 통과가 없음을 확인하여 기존 Size30%-Stroke28%-CoM을 유지했다. **최종 배포는 미변경**이다.

앞으로 오류가 높은 클래스의 손글씨를 추가하는 연구는 **후속 협의 대상**이며, 이번 main 반영에서 손글씨 원본 추가·학습 데이터 변경·새 모델 학습을 수행하지 않았다.
