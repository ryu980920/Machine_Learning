# Combined 가설 B EXP유002 / EXP-CMB-005 — 저비율 Size × Stroke 전처리 최종 실측 및 기전 판정

> **실험 목적 달성 (원인 설명 기준) / 모델 채택 보류.** 하이퍼파라미터는 실험 전체에서 고정. 기존 Normal EXP유017 HP 조합은 사용하지 않았다. 실험은 Official Original Test·교수님 Final Test·Normal 신규 480장을 사용하지 않았고, `main` 또는 배포 모델을 변경하지 않았다.

## 1. 이전 실패 실행과 성공한 복원

- **1차 0%/10% 성공:** [#43857354379](https://github.com/ryu980920/Machine_Learning/actions/runs/43857354379). 4조건×3 Seed×4 held-out writer = 48개 신규 학습.
- **2차 20%/30% 18개 학습·2개 집계 모두 성공:** [#43871619789](https://github.com/ryu980920/Machine_Learning/actions/runs/43871619789). 6조건×3Seed×4Fold = 72개 신규 학습. 다만 **결과 집계 후 Git push가 stale branch로 거절**, 원본 run은 failure 표시. 학습 실패로 간주하지 않는다.
- **복구 성공:** [#43876627107](https://github.com/ryu980920/Machine_Learning/actions/runs/43876627107). 원본에서 성공적으로 업로드된 `EXP-CMB-005-full-10-arm-3seed-4writer-audit` Artifact를 **재학습 없이** 복원했다. 각 `SUMMARY.md`/집계 CSV, expected_fits=120, 10×240 오분류 방향, 10×16 클래스 오류율, no_hp_search=true, no_test_used=true를 검증하고 실험 브랜치에 push했다.
- **전체 10조건 × 3Seed × 4Fold = 총 120개 MLP 재학습 완료**, 동일 CV·학습 데이터 원본 SHA·MLP 기본 HP·학습 개수 비교. 학습당 Official Training 76,800 + 다른 작성자 손글씨 1,920, 평가 OOF 총 2,560 및 OfficialVal 19,200. Fold가 겹치므로 독립 120회 인구추정이 아니라 seed/작성자에 대한 반복 측정이다.
- 사전등록: [EXP유002 설계](EXP유002_Balanced_Scale_Stroke_Augmentation.md) · [1차 일지](EXP유002_Balanced_Scale_Stroke_Stage1_Results.md).

## 2. 같은 학습 설정에서 비교한 전체 결과 (3Seed 평균)

| Size 증강 | Stroke 강도 | OOF Accuracy | 숫자 Accuracy | 문자 Accuracy | OfficialVal Accuracy | 숫자 ×1.2 Family Flip | 문자 ×0.8 Family Flip |
|---|---|---:|---:|---:|---:|---:|---:|
| 0% | 0% | 84.883% | 87.750% | 80.104% | **96.733%** | 69.77% | 35.59% |
| 10% | 0% | 89.935% | 91.375% | 87.535% | 96.489% | 26.77% | 11.28% |
| 10% | 14% | 88.971% | 90.583% | 86.285% | 96.294% | 23.12% | 11.91% |
| 10% | 28% | 89.661% | **91.563%** | 86.493% | 96.565% | 18.71% | 11.88% |
| 20% | 0% | 90.195% | 91.021% | 88.819% | 96.387% | 15.90% | 9.38% |
| 20% | 14% | 89.792% | 90.771% | 88.160% | 96.301% | 13.81% | 10.21% |
| 20% | 28% | 89.961% | 91.021% | 88.194% | 96.330% | 12.00% | 10.35% |
| 30% | 0% | 90.573% | 90.542% | **90.625%** | 96.181% | 12.23% | **7.15%** |
| 30% | 14% | 90.690% | 90.958% | 90.243% | 96.134% | 10.60% | 7.95% |
| 30% | 28% | **90.716%** | 91.104% | 90.069% | 96.117% | **9.19%** | 8.19% |

- **0→10% Size 변화:** 공식 Train을 10%만 클래스 균형 크기로 대체해도 OOF가 84.883→89.935%, 숫자×1.2 Family Flip이 69.77→26.77%로 변화. 이 비교는 동일 CoM·기본 MLP/Seed/Fold이므로 Size 증강을 통한 학습 분포 변화와 관련된 결과로 해석한다.
- **10→20→30% Size 증가:** Stroke0 고정에서 OOF 89.935→90.195→90.573%, 문자 87.535→88.819→90.625%, 숫자×1.2 Flip 26.77→15.90→12.23%. 증가할수록 가족 간 판단 민감도는 낮아지지만 공식 Val 96.489→96.387→96.181%로 감소.
- **획 강도 효과:** 같은 Size에서 Stroke를 추가해도 정상 OOF Accuracy가 항상 높아지지 않는다. Size10은 Stroke0이 최고 OOF, Size30은 Stroke28이 최고 OOF이나 +0.143%p에 불과. Size30 Stroke28은 숫자×1.2 Flip이 9.19%로 최저, 반면 문자×0.8 Flip은 Size30 Stroke0의 7.15%가 최저다. 단일 승자 선언 불가.
- **OfficialVal 상충:** 30%-28%는 CoM-only 대비 OOF **+5.833%p**, OfficialVal **−0.616%p**. 공식 분포 성능 유지 문제를 해결하지는 못했다.

## 3. 정확도만으로 감춰질 수 있는 16개 클래스·240개 오류 방향

기존 EXP-CMB-001 역사적 원본 Baseline은 동일 OOF 2,560장 Seed42·CoM 미적용(81.25%)으로, 아래 3Seed CoM-only 기준과 **동일 전처리 대조가 아니다**. **인과적 효과 크기는 CoM-only 3Seed를 직접 대조**해야 한다. 아래 오류 건수는 OOF 2,560장 기준 3Seed 평균.

| 실제→예측 | CoM-only 0% | Size10 Stroke0 | Size30 Stroke0 | Size30 Stroke28 | 판정 |
|---|---:|---:|---:|---:|---|
| 0→D | 27.0 | 9.7 | 8.3 | 6.3 | 대폭 감소 |
| D→0 | 35.0 | 27.0 | 24.7 | 23.7 | 감소하나 여전히 잔존 |
| 8→B | 21.3 | 10.7 | 16.3 | 14.3 | 10% 대비 일부 재증가 |
| B→8 | 35.3 | 25.7 | 16.7 | 18.0 | 30%에서 개선 |
| 5→F | 12.0 | 7.0 | 4.3 | 3.0 | 감소 |
| 7→9 | 12.0 | 12.3 | 13.3 | 12.3 | 개선 확인 안 됨 |
| 9→7 | 3.0 | 2.3 | 1.3 | 2.3 | 소폭 감소 |
| 5→8 | 4.3 | 5.0 | 4.3 | 5.0 | 개선 확인 안 됨 |

- CoM-only 대비 Size30-Stroke28의 B 클래스 **전체 오분류율 37.29→17.92%**, D **26.88→17.71%**, 숫자 0 **19.58→5.42%**, C **14.79→6.46%**, 숫자 9 **9.38→5.00%**.
- 한편 숫자 2 전체 오류 **7.29→9.79%**, 숫자 7 **9.58→10.83%**로 악화했다. 최초 EXP-CMB-001(전처리·단일Seed 상이) 원본의 C 오류 **3.13%**보다 Size30-Stroke28 C 오류 **6.46%**가 높다는 반례도 남았다.
- **동일 크기 증강에서 Stroke 강도별 trade-off:** Size30-Stroke0 vs Stroke28은 전체 OOF 90.573→90.716%만 상승했지만 문자 OOF 90.625→90.069%로 하락, 숫자×1.2 Flip 12.23→9.19%로 감소. B→8은 16.7→18.0건으로 소폭 악화.
- Writer 41은 CoM-only 78.28→Size30-Stroke28 85.99%, Writer43은 83.07→94.79%, Writer44는 90.10→90.42%. 작성자별로 개선 폭이 다르다.
- 각 클래스는 OOF 160장/Seed. [16개 전체 오류율](../../results/Balanced_Scale_Stroke/full_comparison/all_16_class_error_changes.csv) · [240개 오분류 방향](../../results/Balanced_Scale_Stroke/full_comparison/all_240_direction_changes.csv) · [숫자/문자 계열 내부·교차 오류](../../results/Balanced_Scale_Stroke/full_comparison/family_error_direction_comparison.csv) · [해당 작성자·Seed별 원본](../../results/Balanced_Scale_Stroke/full_comparison/per_fold_group_metrics.csv).

## 4. 기전 설명, 반례 및 연구 목적 달성 판정

1. **가설과 일치하는 사실:** Normal에서 발견한 공식 Training의 숫자/문자 크기 통계 차이가 Combined에 Hand-made 1,920장을 추가해도 남을 수 있다는 문제 제기에서, 공식 Training 일부에 클래스 균형 목표 크기(16/18/20/22/24px)를 적용하여 재학습하면 크기 변경에 따른 숫자·문자 Family Flip이 감소하고 정상 Hand-made 성능도 함께 향상됨을 **동일 모델·Fold·Seed 비교**로 확인했다.
2. **Size 자체 vs 동반 변화:** BBox 크기 조정은 획 두께·픽셀 질량·보간 효과도 바꾼다. 따라서 “모델이 BBox 긴 변 하나만 보고 숫자/문자를 판정한다” 또는 인과적으로 크기 외의 단서가 0%라고 단정할 수는 없다. 남는 Writer/클래스별 최악 Family Flip(30%-28% Writer-Seed B/8/D 최대 **52.5%**, OfficialVal Fold-Seed 최대 **25.58%**)이 반례다.
3. **Stroke 추가 효과는 부분적:** Stage1~2 비교에서 Stroke가 평균 Flip을 추가로 줄이기도 하지만 정상 Accuracy/문자 성능에 부작용을 낸다. 원인이 Size만은 아니라는 증거와 동시에, Stroke 강도 28%가 정답이라는 주장은 불충분하다.
4. **정량 통과선이 아닌 설명 중심 목표는 충족:** Size 비율과 Stroke 강도를 독립적으로 변경하는 동일 HP의 10조건·3Seed·4Writer Fold, 5배율 16클래스/240방향 오류, 정상 OOF·OfficialVal, 작성자별 반례를 함께 비교하여 *어떤 개입이 어떤 오류와 크기 민감도를 변화시키는지* 입증 가능한 수준의 근거가 확보되었다.
5. **최종 모델 채택은 보류:** OfficialVal의 일관된 손실, 특정 클래스/새 혼동의 악화, 이미 개발에 반복 사용된 작성자 4명이라는 일반화 한계, 새로운 독립 Writer 검증 부재. Original/Professor Test 미사용. 30%-28%는 0~30% **탐색된 조건 내 최고 OOF일 뿐** 최종 추천 모델 아님.

**결론:** Combined의 Hand-made 학습만으로는 크기 관련 분류 Shortcut 문제가 자연히 없어지지 않았다. Official Training의 클래스 균형 Size 증강은 해당 Shortcut을 크게 낮추며 전 클래스 오류 구조를 바꾸지만, 데이터 출처 간 Accuracy 상충과 개별 클래스의 잔존 혼동을 함께 관리해야 한다.

## 5. 원본 산출물

- [10조건×3Seed 실측 집계](../../results/Balanced_Scale_Stroke/full_comparison/SUMMARY.md) · [10조건 비교 CSV](../../results/Balanced_Scale_Stroke/full_comparison/comparison.csv) · [120개 MLP 실행 manifest](../../results/Balanced_Scale_Stroke/full_comparison/manifest.json).
- Stage1·Stage2 각각의 `raw/` 원본 CSV·loss·이미지·Manifest; 개별 모델 체크포인트는 각각의 GitHub Actions Artifact에서 30일 보존. 이번 정상 branch 기록에 모델 joblib을 포함하지 않음.
- **원본 집계 계산 완료 후 Git push 실패를 복구**했으므로, 2차 원본 run Conclusion=failure는 **결과 집계 실패가 아닌 저장 충돌**이며 복구 성공 run으로 감사 추적해야 한다.
- [Draft PR #16](https://github.com/ryu980920/Machine_Learning/pull/16). 사용자 승인 전 `main` 병합·최종 모델/HP 변경 없음.
