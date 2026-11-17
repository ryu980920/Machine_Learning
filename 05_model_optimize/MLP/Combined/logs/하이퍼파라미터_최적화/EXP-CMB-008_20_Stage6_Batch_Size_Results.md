# EXP-CMB-008 · S6 Batch Size 200/128/64 실측 비교

> **결론은 개발용 추천이며 최종 HP 적용/모델 배포는 하지 않음.** Early Stopping=False는 유지. 3 Seed × 4 held-out Writer × 3 Batch Size = **36개 새 학습**.

## 목표 및 고정 조건

- 강의자료 09B 13번 슬라이드의 작은 Batch Size 비교 제안 검증. 바뀌는 항목은 **batch_size** 뿐이다.
- 전처리 Size30% / Stroke28% / CoM, MLP (256,), ReLU, Adam, alpha=1e-4, lr_init=.001, early_stopping=False, max_iter=200. Official Train 76,800 + 다른 Writer 1,920장으로 학습, 보류 Writer 640장과 OfficialVal 19,200장 평가.
- Original Test, 교수님 Test, 신규 480장 사용하지 않음. 원본 SHA·평가 파이프라인 동일.

## 정량 결과

| Batch Size | Writer OOF | OfficialVal | D→0 | B→8 | 8→B | 0→D | 가드레일 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 200 | 91.810% | 96.980% | 77 | 46 | 46 | 10 | 통과 |
| 128 | 92.409% | 97.067% | 72 | 50 | 43 | 15 | 실패: class_0_8_B_D_error_nonincrease, B_to_8_nonincrease, 0_to_D_nonincrease |
| 64 | 92.240% | 96.993% | 73 | 44 | 47 | 20 | 실패: class_0_8_B_D_error_nonincrease, 8_to_B_nonincrease, 0_to_D_nonincrease |

### Writer별 보류 OOF 및 학습 비용

| Batch | W41 | W42 | W43 | W44 | 평균 학습 Epoch | 평균 Fit Time (초) | 수렴 경고 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 88.646% | 92.292% | 95.156% | 91.146% | 63.4 | 58.7 | 0 |
| 128 | 87.917% | 93.125% | 96.042% | 92.552% | 53.8 | 68.2 | 0 |
| 64 | 88.021% | 92.656% | 95.781% | 92.500% | 48.5 | 96.8 | 0 |

### 주요 클래스별 전체 오분류율

| Batch | 0 | 8 | B | D | 최악 Writer OOF | 클래스별 최대 OOF Flip |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3.750% | 13.333% | 15.625% | 19.792% | 88.646% | 22.083% |
| 128 | 4.583% | 12.292% | 15.833% | 18.333% | 87.917% | 22.917% |
| 64 | 5.208% | 13.542% | 16.458% | 18.542% | 88.021% | 20.833% |

## 선정 규칙과 판단

- **1순위:** D→0 / B→8 / 8→B / 0→D 4방향 각각 <5% (480개에서 23건 이하), 동시에 모든 가드레일 통과.
- **미달 시:** Writer OOF가 기준선보다 낮지 않고, D→0 등 네 방향과 class 0/8/B/D 오류가 증가하지 않으며, Writer별 OOF 최소 1%p/OfficialVal 최대 0.5%p 손실 제한 및 최대 클래스 가족 Flip 증가 5%p 이내의 후보만 선정.
- **가드레일 통과 후보 중** OOF 최고 정확도와 0.2%p 이내인 후보는 D→0을 우선하고 나머지 세 방향 합계로 순서를 매긴다.
- **검증 결과:** 4방향 엄격 목표 통과 후보: 없음. 안전 가드 통과 후보: [200]. **개발용 추천 Batch Size: 200**.
- 후보의 OOF가 높더라도 한 방향·취약 클래스가 악화되면 전체 정확도만으로 변경하지 않는다.

## 기준선 재실행 감사

- 원본 S5 ES=False OOF: **92.214%**. 이번 새 Batch=200 OOF: **91.810%**. 원본과 전체 정확도 일치 여부: **False**.
- 4방향 오류 완전 일치 여부: **False**. 기존 연구의 분할·환경·실행 차이에 따른 재현성 차이가 나타나면 수치를 숨기지 않는다.
- **이번 S6 후보 비교·선정은 새 36개 실행에 포함된 동일 Batch=200 기준선에 상대 비교했다.** 원본 S5와 수치가 다르더라도 이를 원본 S5를 덮어쓴 결과로 취급하지 않는다.

## 동일 7,680건 예측에서 교정과 신규 오류

| 후보 vs Batch=200 | 기존 오답 교정 | 기존 정답이 오답으로 변함 | 순개선 |
|---:|---:|---:|---:|
| 128 | 193 | 147 | +46 |
| 64 | 233 | 200 | +33 |

## 후속 결정

- **추천 조건: Batch Size 200** (개발용 분석 결과). **현재 기준선 200의 HP와 모델을 자동으로 바꾸지 않는다.**
- 기존 Early Stopping=False 확정 상태와 무관하게, 이번 실험은 batch_size만 변경했다.
- 동일 보류 Writer 4명을 반복 활용한 개발용 OOF 선택 편향 가능성은 남아 있으며, 독립 Test나 최종 배포 전에 별도 승인 및 검증이 필요하다.

## 원본 추적

- [정량 CSV](../../results/Hyperparameter_Optimization/S6_Batch_Size/S6_batch_size_comparison.csv)
- [종합·가드레일 JSON](../../results/Hyperparameter_Optimization/S6_Batch_Size/S6_results.json)
- [오답 교정/유발 JSON](../../results/Hyperparameter_Optimization/S6_Batch_Size/S6_paired_corrections.json)
- [S6 학습 코드](../../scripts/exp_cmb008_s6_batch_size.py)
- [S6 집계 코드](../../scripts/exp_cmb008_s6_batch_aggregate.py)
