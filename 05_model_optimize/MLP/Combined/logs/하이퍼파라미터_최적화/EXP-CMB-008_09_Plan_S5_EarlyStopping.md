# EXP-CMB-008 · 09 S5 Early Stopping 비교 사전등록

> **S4 실제 승자 Learning Rate가 정해지기 전 작성된 후속 실험 규칙.** S5는 새로운 LR 탐색이 아니라, S4에서 선택된 LR **하나만 고정한 뒤 Early Stopping의 효과를 검사**한다. Learning Rate는 S4 원본 결과 JSON에서 자동으로 읽으며 사전 결정하지 않는다.

## 왜 Early Stopping을 마지막에 비교하는가?

S1~S4에서 Size30/Stroke28/CoM, 은닉층 용량, L2 규제, LR을 순차 동결했다. 현재까지 S1/S2/S3 모델은 `early_stopping=False`로 학습했다. S5는 학습 데이터를 일부 보류하고 자동 종료하면 과적합이 완화되는지, 또는 학습 샘플이 줄어 미학습·D→0 등 악화를 초래하는지 직접 검증한다.

| 후보 | 구체적 학습 방식 | 검증 이유 |
|---|---|---|
| **False (S4 기준선)** | max_iter=200, 내부 조기 종료용 Validation 미분리. S4 또는 S1 원본 12개 모델 그대로 재사용 | 기준 정확도·오류 유지 |
| **True** | max_iter=200, **Training Fold 내부만 10% 분리**(`validation_fraction=0.10`), `n_iter_no_change=10`. 신규 3Seed×4Writer = **12개 모델** | 내부 검증 기반 종료가 Writer 일반화/오분류/학습 손실에 미치는 영향 |

- 고정: S4 실제 선정 LR, 단층 (256,), alpha=0.0001, ReLU/Adam, Size30%/Stroke28% 학습 증강, CoM, 4 Writer 분리, 3 Seed, 원본 데이터 SHA.
- **데이터 누수 방지:** 조기 종료용 10%는 그 Fold **Training 데이터 안에서만** 추출한다. held-out Writer, OfficialVal, Original Test 및 교수님 Test는 조기 종료·피팅에 입력하지 않는다.
- 동일 원본 손글씨 이미지 예측에 대해 ES=False/True를 일대일 비교하고 오분류가 교정된 개수/새로 생성된 개수를 보고한다.

## 사전 선정 기준 및 과적합 관리

1. D→0, B→8, 8→B, 0→D의 방향별 오분류율 **모두 5% 미만** + OOF 정확도 이전 LR 기준 이상이 우선 목표.
2. 목표 달성이 없으면 S4 선정 ES=False 기준보다 OOF 정확도가 낮아지지 않고, 네 방향과 클래스 0/8/B/D 전체 오류율이 악화되지 않으며, OfficialVal(−0.5%p), Writer별 OOF(−1.0%p), Family Flip(+5%p) 가드레일을 통과할 때만 True 채택. 아니면 False 유지.
3. 차이 0.20%p 이내면 D→0 오류가 적은 설정을 우선하고, 이후 3Seed 안정성·OfficialVal·모델 단순성을 비교한다.
4. 내부 조기 종료 검증 정확도는 독립 Test가 아니다. `n_iter_`, 평균 처음/마지막 Train Loss, Writer별 OOF 및 공식 Val 정확도·수렴 경고를 기록해 과소적합/과적합 가능성을 평가한다.

## 완료 기준

S4 결정 → S5의 동일 LR 두 후보 실측 → 4방향 오류/Seed/Writer/Train Loss 비교 → Early Stopping 잠정 동결 → **S1~S5 결정 이유와 탈락 사유를 최종 연구 기록으로 정리**. 독립 Test는 여전히 미사용한다.

[S4 사전등록](EXP-CMB-008_07_Plan_S4_LearningRate.md) · [S5 재현 코드](../../scripts/exp_cmb008_s5_early_stopping.py) · [S4/S5 공통 분석](../../scripts/exp_cmb008_s4_s5_aggregate.py).
