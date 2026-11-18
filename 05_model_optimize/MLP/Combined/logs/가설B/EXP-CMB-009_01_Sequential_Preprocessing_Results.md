# EXP-CMB-009 — Combined MLP 추가 전처리 순차 비교 실측 보고

- **실험:** Local Warp → Rotation(±3°/±5°) → Shear(±0.05/±0.10) → 유효 후보만 조합
- **실행:** GitHub Actions 44246719485, `completed/success`
- **브랜치:** `exp/cmb009-sequential-augmentation-20261118`에서 실험 실행 후 코드·기록을 `main`에 반영
- **학습:** 조건당 Seed 42·43·44 × 보류 Writer 41·42·43·44 = 12개, 6조건 총 **72개 모델**
- **모델 고정:** Combined Size30% / Stroke28% / CoM, MLP `(256,)`, ReLU, Adam, alpha=0.0001, LR=0.001, Early Stopping=False, Batch=128, max_iter=200
- **입력 변형:** 공식 학습 76,800장 중 클래스 균형 **20%에만** 추가 변형. 보류 Writer와 OfficialVal에는 추가 변형 없음.
- **변형 보존:** 28×28 보간·빈 공간 0, 변형 전후 threshold128의 연결요소·Euler 위상 불변, 전체 픽셀 강도 비율 0.7~1.3, CoM 재정렬.
- **평가 경계:** Writer OOF 2,560장 × 3 Seed = 7,680 예측, OfficialVal 19,200장 × 12모델 평균. 신규 438장, 원본 Test, 교수님 Test는 학습·후보 선정에 사용하지 않음.

## 조건별 실제 결과

| 조건 | Writer OOF | OfficialVal | Macro F1(OOF) | 7→9 | 9→7 | 0→D | B→8 | 기준선 대비 OOF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Batch128 / Size30–Stroke28–CoM 기준선 | 92.2786% | 97.0894% | 0.92266 | 26 | 4 | 18 | 57 | +0.0000%p |
| Local Warp: 상단/하단 ±1px | 92.2266% | 97.1853% | 0.92231 | 25 | 4 | 14 | 54 | -0.0521%p |
| Rotation ±3° | 92.3568% | 97.1003% | 0.92366 | 31 | 7 | 17 | 41 | +0.0781%p |
| Rotation ±5° | 92.3698% | 97.1115% | 0.92377 | 23 | 7 | 19 | 52 | +0.0911%p |
| Shear ±0.05 | 92.2266% | 97.0890% | 0.92236 | 23 | 1 | 15 | 48 | -0.0521%p |
| Shear ±0.10 | 92.1224% | 97.0122% | 0.92123 | 24 | 5 | 22 | 52 | -0.1563%p |

방향별 7→9·9→7 오답 건수는 실제 해당 클래스의 **480회 예측(40장×4 Writer×3 Seed)** 중 발생한 총건수다. 438장의 해당 오류 건수가 아니다.

## 단계별 관찰과 조합 판단

### 1단계 — Local Warp
- OOF 92.2266%(기준 대비 -0.0521%p), OfficialVal 97.1853%(+0.0959%p), 7→9 26→25건.
- 보류 Writer 43 하락이 -1.1979%p로 크며 OOF 개선 기준 미충족. 단독 조합 채택 불가.

### 2단계 — Rotation
- ±3°: OOF +0.0781%p, 7→9 26→31건 및 9→7 4→7건으로 악화.
- ±5°: OOF +0.0911%p, OfficialVal +0.0221%p, 7→9 26→23건. 그러나 9→7 4→7건 증가, 사전 기준 +0.1%p 미달.
- 작은 정확도 개선은 관찰됐으나, 원래 7↔9 목표의 양방향 비악화 조건을 충족하지 못함.

### 3단계 — Shear
- ±0.05: OOF -0.0521%p, OfficialVal 약 -0.0004%p, 7→9 26→23건, 9→7 4→1건. 기존 B→8 57→48건.
- ±0.10: OOF -0.1563%p, OfficialVal -0.0773%p. 단독 적용 불채택.

### 4단계 — 개선 조건만 결합
- 사전 정의한 6개 탐색 가드레일을 **모두 통과한 단독 조건이 0개**다.
- 따라서 **최소 2개 이상의 검증된 개선 조건을 결합한다는 조건을 충족하지 못해 조합 재학습은 실행하지 않았다.**
- 여러 개를 무조건 합치는 실험으로 바꾸지 않았다. 기존 Batch128 기준선도 대체·배포하지 않았다.

## 단독 조건 선정 가드레일

후속 실험 계획을 실행 전에 고정했으며, 438장 결과에 맞추어 수정하지 않았다.

- Writer OOF 기준선 대비 **+0.10%p 이상** 개선
- OfficialVal 감소 **0.30%p 이내**
- Writer 최악 성능 하락 **1.00%p 이내**
- 7→9 **비증가**, 9→7 **비증가**
- 기존 0↔D·8↔B 네 방향 오류 합계 증가 **6건 이하**

| 조건 | 기존 오답 교정 | 신규 오답 | 순변화 | 통과 | 실패한 조건 |
|---|---:|---:|---:|---|---|
| Local Warp: 상단/하단 ±1px | 181 | 185 | -4 | 미통과 | `oof_gain_at_least_0p1pp, writer_worst_drop_max_1pp` |
| Rotation ±3° | 172 | 166 | +6 | 미통과 | `oof_gain_at_least_0p1pp, seven_to_nine_no_increase, nine_to_seven_no_increase` |
| Rotation ±5° | 198 | 191 | +7 | 미통과 | `oof_gain_at_least_0p1pp, nine_to_seven_no_increase` |
| Shear ±0.05 | 181 | 185 | -4 | 미통과 | `oof_gain_at_least_0p1pp` |
| Shear ±0.10 | 173 | 185 | -12 | 미통과 | `oof_gain_at_least_0p1pp, nine_to_seven_no_increase` |

## 결론과 후속 선택

**전체적으로 우세한 신규 전처리는 이번 탐색 범위에서 확인되지 않았다.** 정확도만으로는 Rotation ±5°가 가장 높았으나 개선 폭이 작고 9→7 악화가 동반됐다. Shear ±0.05는 7↔9 양방향 오류를 줄였지만 전체 OOF가 낮아졌다. 그러므로 현재는 **기존 Batch 128 / Size30%·Stroke28%·CoM을 기준 후보로 유지**한다.

이번 연구의 7→9 개선은 기존 네 작성자를 반복 평가한 개발용 결과에 한정된다. 특히 기존 438장은 이미 QA·오류 분석 및 Batch Size 선정에 활용된 데이터이므로 독립 Test로 사용하지 않는다.

## 원본 추적

- [GitHub Actions 완료 기록](https://github.com/ryu980920/Machine_Learning/actions/runs/44246719485)
- [실험 브랜치](https://github.com/ryu980920/Machine_Learning/tree/exp/cmb009-sequential-augmentation-20261118)
- [학습 코드](https://github.com/ryu980920/Machine_Learning/blob/exp/cmb009-sequential-augmentation-20261118/05_model_optimize/MLP/Combined/scripts/exp_cmb009_sequential_aug.py)
- [집계 코드](https://github.com/ryu980920/Machine_Learning/blob/exp/cmb009-sequential-augmentation-20261118/05_model_optimize/MLP/Combined/scripts/exp_cmb009_aggregate.py)
- GitHub Actions 아티팩트: `cmb009-standalone-summary-selection` 및 72개 Fold별 아티팩트(가중치·Manifest·OOF CSV; 30일 보존).
- **게시 상태:** 실험 코드·후속 연구 기록을 `main`에 반영. 학습에 사용되는 기존 실험 원본과 실제 배포 모델·체크포인트는 변경하지 않음.