# EXP-CMB-008 · 07 S4 Learning Rate 최적화 사전등록

> **S3 결과 확인 후, S4 학습 이전 작성.** 실측 결과가 나온 뒤 탐색 범위·조건을 바꾸지 않는다.

## S3 → S4로 넘어간 이유

S3에서 더 높은 L2 규제(alpha=0.001/0.01)는 OOF 정확도 일부 개선에도 불구하고 **D→0이 67→80/86건**으로 증가했다. 따라서 **alpha=0.0001과 단층 (256,) 유지**. 규제의 영향과 최적화 속도 영향을 구별하기 위해 Adam의 초기 Learning Rate 하나만 변경한다.

| learning_rate_init | 실험 이유 |
|---:|---|
| 0.0005 | 현 기준의 절반. 더 느린 가중치 업데이트로 안정적 수렴과 D/0 판별 개선 여부 확인 |
| **0.001** | 현재 S1/S2/S3 기준선. S1 원본 12개 모델 재사용 |
| 0.002 | 현 기준의 두 배. 빠른 수렴의 정확도 이득과 진동·작성자별 악화 가능성 확인 |

고정 변수: **Official Train Size30%/Stroke28% 증강 + CoM, hidden_layer_sizes=(256,), alpha=0.0001, ReLU+Adam, max_iter=200, early_stopping=False**, Seed 42·43·44, Writer 41·42·43·44. 신규 후보 2 × 3 Seed × 4 Writer = **24개 모델**. 같은 Training 원본 SHA와 augmentation Seed 20261116을 유지한다.

## 결과 확인 전 채택 규칙

- 1차 목표: D→0, B→8, 8→B, 0→D의 *실제 클래스 기준* OOF 방향별 오류율이 **각각 5% 미만**(480회 중 각 23건 이하).
- 목표 달성 조건 중 기존 LR=0.001 기준 OOF 정확도 **92.213542% 이상**이고 Writer·OfficialVal·Family Flip 등 안전 규칙을 지킨 후보에서 정확도를 우선.
- 목표 불가능 시: OOF 정확도 비하락, 네 방향 각각 오류 비증가, 실제 클래스 0/8/B/D 전체 오분류율 비증가, Writer별 Accuracy −1%p·OfficialVal Accuracy −0.5%p·Family Flip +5%p 가드레일 통과 후보에서 정확도 최우선. OOF 정확도 0.20%p 이내이면 D→0 오류가 적은 조건 우선. 후보 없으면 **LR=0.001 유지**.
- Writer별 OOF 정확도, Seed 변동, 각 Fold Loss Curve·n_iter·수렴 경고·fit 시간, 동일 이미지 교정/신규 오답을 함께 보관한다. 반복 탐색에 의한 OOF 선택 과적합 가능성을 명시한다.

**S4 종료 즉시 S5 연결:** 실제 선정된 LR을 그대로 고정해 Training-only Early Stopping(False/True)을 실측 비교한다. S4 결과 이전에 S5 LR값을 임의 결정하거나 재탐색하지 않는다. Original Test·교수님 Test·신규 Hand-made 480장은 두 단계 모두 사용하지 않는다.

[S3 원본 결과](EXP-CMB-008_06_Stage3_Alpha_Results.md) · [S4 실행 코드](../../scripts/exp_cmb008_s4_learning_rate.py) · [S4 집계 코드](../../scripts/exp_cmb008_s4_s5_aggregate.py).
