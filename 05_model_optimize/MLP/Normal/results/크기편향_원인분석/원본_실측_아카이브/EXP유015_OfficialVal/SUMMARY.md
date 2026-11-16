# EXP015 — OfficialVal 크기 개입 교차 진단

78개 학습 후 기존 사전 Gate를 통과한 9개 후보와 동결된 CoM 원본을 **고정 가중치**로 재평가하였다.
기존 OfficialVal 19,200장의 원본 이미지를 0.8·0.9·1.0·1.1·1.2배 변형한 후 CoM 정렬.
**Hand-made에 대해 사전 등록한 Gate/순위는 변경하지 않는다.**
OfficialVal은 이미 반복 사용된 개발 Validation이지 새로운 독립 Test가 아니다.

## 숫자 ×1.2 및 문자 ×0.8의 Family 판단 전환율

| 모델 | OfficialVal 숫자 1.2x flip | OfficialVal 문자 0.8x flip | 원본 OfficialVal Accuracy |
|---|---:|---:|---:|
| baseline_com | 75.67% | 49.42% | 97.052% |
| size30_stroke28 | 5.00% | 5.23% | 96.181% |
| size30_stroke42 | 4.57% | 5.73% | 96.134% |
| size35_stroke35 | 4.56% | 5.05% | 96.012% |
| size35_stroke42 | 4.69% | 5.34% | 95.682% |
| size40_stroke14 | 4.46% | 5.10% | 95.880% |
| size40_stroke28 | 3.96% | 5.59% | 95.936% |
| size50_stroke00 | 4.03% | 4.13% | 95.639% |
| size50_stroke14 | 3.65% | 4.78% | 95.576% |
| size50_stroke28 | 3.71% | 4.63% | 95.628% |

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
| size30_stroke42 | 8 | 8.63% | 11.45% |
| size30_stroke42 | B | 10.69% | 17.58% |
| size30_stroke42 | 0 | 9.85% | 10.38% |
| size30_stroke42 | D | 11.33% | 20.08% |
| size30_stroke42 | 5 | 3.75% | 5.84% |
| size30_stroke42 | F | 3.33% | 3.83% |
| size35_stroke35 | 8 | 9.83% | 10.37% |
| size35_stroke35 | B | 8.53% | 15.61% |
| size35_stroke35 | 0 | 8.10% | 10.80% |
| size35_stroke35 | D | 10.08% | 22.64% |
| size35_stroke35 | 5 | 4.43% | 5.32% |
| size35_stroke35 | F | 3.28% | 4.11% |
| size35_stroke42 | 8 | 11.28% | 10.85% |
| size35_stroke42 | B | 10.28% | 20.06% |
| size35_stroke42 | 0 | 15.39% | 11.53% |
| size35_stroke42 | D | 10.25% | 23.50% |
| size35_stroke42 | 5 | 4.18% | 5.26% |
| size35_stroke42 | F | 3.06% | 4.36% |
| size40_stroke14 | 8 | 7.72% | 10.37% |
| size40_stroke14 | B | 9.06% | 12.28% |
| size40_stroke14 | 0 | 10.32% | 11.81% |
| size40_stroke14 | D | 9.19% | 15.22% |
| size40_stroke14 | 5 | 3.87% | 5.63% |
| size40_stroke14 | F | 3.61% | 4.44% |
| size40_stroke28 | 8 | 7.52% | 9.03% |
| size40_stroke28 | B | 10.33% | 16.44% |
| size40_stroke28 | 0 | 8.38% | 9.79% |
| size40_stroke28 | D | 10.28% | 25.08% |
| size40_stroke28 | 5 | 3.08% | 4.40% |
| size40_stroke28 | F | 3.58% | 4.36% |
| size50_stroke00 | 8 | 9.29% | 9.49% |
| size50_stroke00 | B | 6.58% | 8.33% |
| size50_stroke00 | 0 | 11.14% | 9.48% |
| size50_stroke00 | D | 6.97% | 13.08% |
| size50_stroke00 | 5 | 3.47% | 4.80% |
| size50_stroke00 | F | 3.50% | 4.64% |
| size50_stroke14 | 8 | 7.55% | 9.97% |
| size50_stroke14 | B | 7.67% | 14.33% |
| size50_stroke14 | 0 | 10.13% | 7.65% |
| size50_stroke14 | D | 9.92% | 19.17% |
| size50_stroke14 | 5 | 3.47% | 3.60% |
| size50_stroke14 | F | 3.00% | 4.78% |
| size50_stroke28 | 8 | 7.58% | 9.32% |
| size50_stroke28 | B | 8.47% | 20.39% |
| size50_stroke28 | 0 | 9.45% | 8.78% |
| size50_stroke28 | D | 8.78% | 23.22% |
| size50_stroke28 | 5 | 3.78% | 4.52% |
| size50_stroke28 | F | 2.69% | 4.78% |

### 한계 및 결론
- 이미지 배율 조절은 크기뿐 아니라 획 질량, 두께, 보간·topology도 함께 바꾼다.
- Hand-made에서 Size Shortcut이 완화된 후보라도 OfficialVal에서 동일한 강건성을 보이지 않을 수 있다.
- B/8/D/0/F/5 등 특정 문자 쌍에서 높은 Flip이 남는다면 Shortcut 완전 제거를 주장할 수 없다.
- 어떤 모델도 Test를 평가하거나 final 모델에 교체하지 않았다.
