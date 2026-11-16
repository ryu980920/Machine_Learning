# EXP016 — OfficialVal 크기 개입 교차 진단

78개 학습 후 기존 사전 Gate를 통과한 9개 후보와 동결된 CoM 원본을 **고정 가중치**로 재평가하였다.
기존 OfficialVal 19,200장의 원본 이미지를 0.8·0.9·1.0·1.1·1.2배 변형한 후 CoM 정렬.
**Hand-made에 대해 사전 등록한 Gate/순위는 변경하지 않는다.**
OfficialVal은 이미 반복 사용된 개발 Validation이지 새로운 독립 Test가 아니다.

## 숫자 ×1.2 및 문자 ×0.8의 Family 판단 전환율

| 모델 | OfficialVal 숫자 1.2x flip | OfficialVal 문자 0.8x flip | 원본 OfficialVal Accuracy |
|---|---:|---:|---:|
| baseline_com | 75.67% | 49.42% | 97.052% |
| size30_stroke28 | 5.00% | 5.23% | 96.181% |
| size40_stroke28 | 3.96% | 5.59% | 95.936% |
| size50_stroke28 | 3.71% | 4.63% | 95.628% |
| size70_stroke00 | 3.07% | 4.21% | 95.278% |
| size70_stroke14 | 2.80% | 4.77% | 94.863% |
| size80_stroke14 | 2.89% | 4.49% | 94.811% |
| size90_stroke00 | 2.71% | 4.70% | 94.562% |
| size90_stroke14 | 2.65% | 5.17% | 94.382% |
| size90_stroke42 | 2.78% | 5.38% | 94.142% |

## 집중 혼동쌍 B·8·D·0·F·5의 Family Flip

| 모델 | 문자/숫자 | 0.8x | 1.2x |
|---|---|---:|---:|
| baseline_com | 8 | 1.31% | 98.26% |
| baseline_com | B | 83.53% | 0.69% |
| baseline_com | 0 | 1.80% | 98.37% |
| baseline_com | D | 65.08% | 1.11% |
| baseline_com | 5 | 0.58% | 81.55% |
| baseline_com | F | 60.00% | 0.56% |
| size30_stroke28 | 8 | 8.92% | 12.17% |
| size30_stroke28 | B | 10.19% | 12.42% |
| size30_stroke28 | 0 | 10.86% | 11.62% |
| size30_stroke28 | D | 9.53% | 18.25% |
| size30_stroke28 | 5 | 3.75% | 6.00% |
| size30_stroke28 | F | 3.75% | 3.33% |
| size40_stroke28 | 8 | 7.52% | 9.03% |
| size40_stroke28 | B | 10.33% | 16.44% |
| size40_stroke28 | 0 | 8.38% | 9.79% |
| size40_stroke28 | D | 10.28% | 25.08% |
| size40_stroke28 | 5 | 3.08% | 4.40% |
| size40_stroke28 | F | 3.58% | 4.36% |
| size50_stroke28 | 8 | 7.58% | 9.32% |
| size50_stroke28 | B | 8.47% | 20.39% |
| size50_stroke28 | 0 | 9.45% | 8.78% |
| size50_stroke28 | D | 8.78% | 23.22% |
| size50_stroke28 | 5 | 3.78% | 4.52% |
| size50_stroke28 | F | 2.69% | 4.78% |
| size70_stroke00 | 8 | 6.64% | 7.66% |
| size70_stroke00 | B | 7.22% | 12.28% |
| size70_stroke00 | 0 | 9.17% | 6.44% |
| size70_stroke00 | D | 7.89% | 17.03% |
| size70_stroke00 | 5 | 3.94% | 3.69% |
| size70_stroke00 | F | 3.03% | 5.78% |
| size70_stroke14 | 8 | 7.52% | 6.89% |
| size70_stroke14 | B | 7.89% | 20.89% |
| size70_stroke14 | 0 | 9.76% | 4.81% |
| size70_stroke14 | D | 9.31% | 28.03% |
| size70_stroke14 | 5 | 3.66% | 3.23% |
| size70_stroke14 | F | 3.39% | 7.17% |
| size80_stroke14 | 8 | 7.24% | 7.55% |
| size80_stroke14 | B | 7.61% | 24.47% |
| size80_stroke14 | 0 | 8.89% | 5.99% |
| size80_stroke14 | D | 8.53% | 25.56% |
| size80_stroke14 | 5 | 3.69% | 3.38% |
| size80_stroke14 | F | 3.56% | 8.81% |
| size90_stroke00 | 8 | 7.07% | 6.44% |
| size90_stroke00 | B | 7.58% | 28.11% |
| size90_stroke00 | 0 | 8.72% | 5.68% |
| size90_stroke00 | D | 8.11% | 34.00% |
| size90_stroke00 | 5 | 4.31% | 3.57% |
| size90_stroke00 | F | 3.83% | 12.17% |
| size90_stroke14 | 8 | 7.61% | 6.18% |
| size90_stroke14 | B | 8.92% | 32.86% |
| size90_stroke14 | 0 | 9.14% | 6.02% |
| size90_stroke14 | D | 9.14% | 35.50% |
| size90_stroke14 | 5 | 2.89% | 2.21% |
| size90_stroke14 | F | 4.44% | 13.61% |
| size90_stroke42 | 8 | 7.98% | 6.98% |
| size90_stroke42 | B | 9.19% | 37.11% |
| size90_stroke42 | 0 | 7.59% | 6.19% |
| size90_stroke42 | D | 10.36% | 32.14% |
| size90_stroke42 | 5 | 3.38% | 2.92% |
| size90_stroke42 | F | 3.58% | 11.53% |

### 한계 및 결론
- 이미지 배율 조절은 크기뿐 아니라 획 질량, 두께, 보간·topology도 함께 바꾼다.
- Hand-made에서 Size Shortcut이 완화된 후보라도 OfficialVal에서 동일한 강건성을 보이지 않을 수 있다.
- B/8/D/0/F/5 등 특정 문자 쌍에서 높은 Flip이 남는다면 Shortcut 완전 제거를 주장할 수 없다.
- 어떤 모델도 Test를 평가하거나 final 모델에 교체하지 않았다.
