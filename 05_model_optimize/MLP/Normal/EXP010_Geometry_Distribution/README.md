# Normal MLP · 가설 B · EXP유010 — 학습·손글씨 글자 크기 분포 불일치 (계획·실행 통합)

> **상태: 분석 완료 · 가설 1의 분포 불일치 확인 · 원인 단독 확정은 보류**
>
> **성공한 실제 분석:** https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796
>
> **출발 기준:** 가설 B EXP유006 CoM-only 문자 66.4931%, 숫자 84.4167%; Size→CoM 문자 91.7708%, 숫자 84.0625% (3 Seed 평균).

## 1. 사전 연구 질문과 판단 기준

**가설 1:** 공식 Training의 MNIST 숫자·EMNIST 문자는 크기 분포가 서로 다르고, 특히 공식 문자가 실제 Hand-made 문자보다 커서 크기 정규화가 문자 오인식을 완화한다.

원본 Official Train/Validation/Hand-made에서 클래스별 긴 변·너비·높이·무게중심·질량을 측정하고, EXP유006과 동일한 Size-only 및 Size→CoM 변환 후 재측정한다. 변환 전후 Official Train vs Hand-made의 Wasserstein-1(W1) 거리를 비교한다. 사전 판단은 크기 격차 감소와 같은 클래스 정확도 개선이 동반되는지 보는 것으로, 분포 거리 감소만으로 인과관계라고 단정하지 않는다.

## 2. 실행 방법 및 데이터 경계

- 2026-11-16 성공 실행. Official Training 76,800장, Official Validation 19,200장, Hand-made 2,560장(41~44 작성자), 16개 클래스의 분포를 실제 측정.
- **코드는 EXP유006에서 쓰던 원본**: Combined의 image_geometry / apply_size, foreground threshold=32, 공식 Training 긴 변 중앙값=20px, bilinear affine 변환, BBox 중심 고정. CoM은 threshold32, (13.5,13.5) 목표 정수 이동.
- 원본(raw), Size-only, Size→CoM의 **동일 이미지 기하학**을 측정. 이미지 크기 정규화의 기준점은 Official Training만에서 추출. 모든 전처리 메서드는 라벨과 작성자 ID를 사용하지 않음.
- Original Test/교수님 Test 미사용, Hand-made 학습 미사용, 새 MLP 학습 없음.
- 산출물: geometry_summary.csv, geometry_distance.csv, handmade_geometry_per_image.csv, 검증된 종합 결과 EXP010–012 CSV.

## 3. 원본 데이터의 실제 BBox 분포

| 클래스 | 공식 Train 긴 변 중앙값 | Hand-made 긴 변 중앙값 | 원본 W1 (px) | Size 후 W1 (px) |
|---|---:|---:|---:|---:|
| B | 24px | 20px | 3.632 | **0.019** |
| D | 24px | 20px | 3.392 | **0.016** |
| F | 24px | 20px | 3.457 | **0.029** |
| 8 | 20px | 20px | 1.056 | **0.019** |
| 0 | 20px | 20px | 1.043 | **0.032** |
| 5 | 20px | 20px | 0.956 | **0.025** |

원본 숫자 그룹의 BBox 긴 변 중앙값은 공식 Train·Hand-made 모두 20px이었지만, 문자 그룹은 공식 Train 24px / Hand-made 20px이었다. Size 후 모두 중앙값 약 20px로 정렬됐다.

**중요한 구분:** 긴 변 정규화가 형상 전체를 같게 만들지는 않는다. B의 BBox **너비 중앙값**은 공식 Train 22px / Hand-made 14px에서 Size 후 **18px / 14px**로 바뀌었으며, 가로폭 차이는 여전히 남는다. D도 22px / 15px → 18px / 15px이다. W1 감소는 알고리즘이 긴 변을 맞추도록 설계된 당연한 결과이기도 하므로 이것 하나만으로 개선 원인을 증명하지 않는다.

**가로폭 분포도 별도로 확인했다.** 공식 Train / Hand-made B의 BBox 폭 중앙값은 Raw **22/14px** → Size **18/14px**, W1은 **6.6267→3.2023px**로 감소했다. D는 Raw **22/15px** → Size **18/15px**, W1 **5.6281→2.4769px**였다. 한편 숫자 8의 폭 W1은 **1.8483→2.2110px**로 오히려 증가했다. 따라서 긴 변을 같게 맞추더라도 클래스별 가로·세로비 불일치는 남을 수 있다.

## 4. 정확도 변화와 연결

동일한 EXP유006 학습 조건과 실제 보관된 체크포인트에서 계산한 문자별 CoM-only → Size→CoM 정확도 변화 (3 Seed 평균):

| 클래스 | CoM-only Recall | Size→CoM Recall | 변화 |
|---|---:|---:|---:|
| A | 72.71% | 97.08% | +24.37%p |
| **B** | 35.83% | **84.17%** | **+48.33%p** |
| C | 78.75% | 95.42% | +16.67%p |
| **D** | 54.79% | **83.75%** | **+28.96%p** |
| E | 80.83% | 95.00% | +14.17%p |
| F | 76.04% | 95.21% | +19.17%p |

문자 전반에서 큰 개선이 나타났으며, B·D의 개선 폭이 특히 크다. 반면 숫자 0~9의 CoM-only → Size→CoM 변화는 클래스별로 상승·하락이 혼재했다.

## 5. 가설 1 판정

**확인:** 공식 Training 문자의 긴 변은 손글씨 문자보다 중앙값 4px 컸고, 숫자는 중앙값이 비슷했다. Size 변환이 이 차이를 해소했으며 문자 정확도가 함께 크게 개선됐다. 특히 B/D의 큰 분포 차이와 정확도 개선이 동시에 관찰됐다.

**미확정:** 크기 분포를 가깝게 만든 것만이 성능 개선의 유일한 원인은 아니다. Scale 변형은 획의 픽셀 강도·두께·보간과 모델의 판단 기준에도 영향을 미친다. EXP유011(동일 모델의 CoM 입력 변화), EXP유012(동일 이미지의 크기 개입)과 결합해서 판단한다.

## 6. 재현 가능한 자료

- 분석 코드: `05_model_optimize/MLP/Normal/scripts/exp010_geometry_distribution.py`
- 종합 코드: `05_model_optimize/MLP/Normal/scripts/exp010_012_aggregate.py`
- Actions run 43801194796의 `exp010-geometry`와 `Normal-EXP010-012-reviewed-causal-results` Artifact: 원본 분포/클래스별 W1/전수 CSV.
- 가설 B [EXP유011](../EXP011_CoM_Effect/README.md), 가설 A [EXP유012](../EXP012_Size_Shortcut/README.md).

> **판정 범위:** 반복 사용한 4명의 개발용 손글씨 결과이며, 최종 Test 일반화나 모델 채택 근거는 아니다.


---

## 독립 실험 폴더 구성

- **실험일지:** 현재 `README.md` — 기존 계획과 실행 결과 원문 수치 보존
- **핵심 측정 요약:** [RESULTS.md](RESULTS.md) 및 [summary.csv](summary.csv)
- **재현 절차와 코드 위치:** [RUN.md](RUN.md)
- **공통 분석 코드:** [exp010_geometry_distribution.py](../scripts/exp010_geometry_distribution.py)
- **실제 완료된 GitHub Actions 실행:** https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796

이 연구는 기존 EXP유006에서 학습한 모델을 고정해 분석했으며 최종 모델을 교체하지 않았다.
