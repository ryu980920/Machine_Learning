# EXP-CMB-005 · 01 계획 — Size 0~30% × Stroke

> **아카이브 안내:** 이 문서는 당시 계획·중간 진행 상태를 그대로 기록한 자료다. 본문에 남은 ‘미실시/진행 중’ 표기는 **작성 당시 상태**이며, 이후 완료된 실측은 [최종 보고](EXP-CMB-005_03_Final_Size00_30.md)를 참조한다.

> 상태: 사전등록 v2 (실행 전 사용자가 실험 순서 변경 요청). **통과 기준은 특정 Accuracy/F1 숫자가 아니라 두 방법의 작용·부작용을 실측으로 설명할 수 있는가**이다. 실행 후 이 조건을 바꾸지 않는다.

## 근거
Normal EXP유013~017은 Official Training에서 숫자·문자의 BBox 크기 단서가 크고, 클래스별 균형 크기 증강(16/18/20/22/24px)+약한 Stroke+CoM으로 크기 개입 Family Flip을 완화했음을 확인했다. Normal의 70-0과 80-14는 서로 다른 데이터 출처에서 B·8·D 최악 Flip이 달랐다. Normal EXP유017 모델의 256×128/alpha=0.03 구조 효과와 전처리 효과는 혼합하지 않는다.

Combined는 공식 Training 76,800 + 타 작성자 Hand-made 1,920을 학습하므로 Normal 결과를 그대로 적용해 성공했다고 주장하지 않는다. 기존 Combined EXP-CMB-004의 CoM-only 85.8594%, Size+CoM 88.9453%는 1-Seed 4-Fold의 역사적 수치다.

## 실험 실행 전 변경 기록 — 2026-11-16

초기 계획은 Normal의 70-0 / 80-0 / 80-14 비교였으나, **사용자가 실제 실행 전에 낮은 Size 비율부터 단계적으로 Size × Stroke 증강을 비교하도록 변경**했다. 기존 브랜치에 기록했던 70/80 계획은 실험 결과가 아니며 실행하지 않았다. 이번 사전등록을 새 기준으로 사용한다. **모델 Hyperparameter 탐색은 일절 수행하지 않는다.**

## 사전에 고정한 1차 저비율 전처리 그리드

| Official Training Size 증강 비율 | Stroke 0% | Stroke 14% | Stroke 28% |
|---|---|---|---|
| 0% | A: CoM-only 기준 | — | — |
| 10% | B10-0 | B10-14 | B10-28 |
| 20% | B20-0 | B20-14 | B20-28 |
| 30% | B30-0 | B30-14 | B30-28 |

**총 10조합.** 0%는 원본 이미지에 CoM 정렬만 수행. Stroke 강도는 **증강 선택된 이미지에만** 적용하며, Normal의 cross-kernel grayscale erode/dilate 혼합 및 연결요소/Euler topology guard와 동일. 0% Size에 Stroke만 적용하는 조건은 이번 비교에서 제외하여, Stroke를 Size 증강의 추가 효과로 해석한다.

1차는 **0%와 10% 조건부터 시작**, 다음 20%, 30%로 상승한다. 동일 Seed·Fold에서 Size 비율만 달리하거나 같은 비율에서 Stroke 0/14/28을 대조한다. 40~50% 이상은 **현재 실행 범위에 포함하지 않으며**, 저비율 결과와 연구 질문 달성 여부를 확인한 뒤 별도 사전등록·승인으로 진행한다. 낮은 비율에서 효과가 작더라도 목표 정확도를 맞추려 임의로 기준이나 변수 범위를 변경하지 않는다.

- 공식 Training **76,800장** 중 각 원본 소스 행을 정확히 하나만 유지하면서, 선정된 비율에 대해 BBox 긴 변 목표 **16·18·20·22·24px**로 균형 할당한다. 10/20/30% subset은 Normal EXP유015의 `nested_subset`과 동일한 클래스·목표 크기별 nested sampling(증강 seed 20261116).
- Official 대상의 Scale→Stroke(선택 강도)→CoM, 나머지 Official 원본은 CoM. 학습 중 입력 표본 수는 동일한 76,800장이다.
- **Hand-made 학습 작성자 3명 1,920장은 CoM만** 적용해 Official 증강의 영향부터 분리한다. 보류 작성자 640장과 공식 개발 Validation 19,200장도 기본 평가는 CoM만 적용하고, 별도 5배율 개입에만 Size 변경한다.
- 전 조건 동일 모델: `Pipeline(PixelScaler255(), MLPClassifier(random_state=seed))` (hidden=100, ReLU, Adam, alpha=0.0001, max_iter=200), **Seed=42·43·44**, 4-Fold 작성자 보류. **10조건 × 3 Seed × 4-Fold = 120개 신규 모델**. Normal EXP유017의 256×128·alpha0.03 최적화는 절대 포함하지 않는다.
- SHA, source-group split, 보류 작성자 데이터 누출, 0.8~1.2배 동일 이미지 평가 검증, 학습 시간과 오류 전환 및 모델 SHA를 기록한다.
- **Original Test, 교수님 Test, Normal EXP유018 신규 480장 미사용.** Baseline 모델/로그 및 Normal 기존 자료 불변.

## 연구 종료 / 통과 기준 (Accuracy 하한 없음)

1. **전처리 조작 확인:** 학습에 실제 대체된 이미지 수, 5개 목표 크기×16클래스 빈도, Size·Width/Height/Ink Mass의 집단 분포, Stroke topology-guard 거절 수 기록.
2. **OOF·공식 Val 대응 비교:** 각 Seed×Fold 동일 이미지의 Accuracy·Macro F1, 숫자/문자·작성자·클래스별 결과, 오답→정답·정답→오답 및 주요 혼동 0↔D·8↔B·5↔F 방향별 전환. 학습시간·loss/iteration 저장.
3. **크기 Shortcut 직접 개입:** 학습한 Fold 모델 고정, 보류 Handmade 및 OfficialVal 각 이미지를 0.8/0.9/1.0/1.1/1.2배 확대·축소 후 CoM 정렬하여 16개 실제 클래스의 family flip, P(letter) 변화와 정상 Accuracy를 분해. 숫자 1.2배, 문자 0.8배, B/8/D 최악을 전체 평균과 별도로 보고.
4. **요인 분리:** 같은 Stroke 강도에서 0/10/20/30%의 비율별 효과, 같은 Size 비율에서 Stroke 0/14/28%의 추가 효과를 같은 Seed/Fold로 비교. 특히 10-0 vs 10-14/28, 20-0 vs 20-14/28, 30-0 vs 30-14/28의 대응 대조로 요인별 효과 검증.
5. **해석·반례:** Shortcut이 감소했지만 정확도가 떨어지는 경우도 성공으로 포장하지 않고, 작성자/클래스 역전 및 OfficialVal 손실 설명. 효과가 상충하거나 기전이 불명확하면 판정 보류 후 추가 통제 분석.
6. **외부 일반화 경계:** 기존 4명 OOF와 OfficialVal은 반복 개발 데이터. Normal EXP유018 신규 480장 결과를 Combined 최적화에 섞지 않는다. 독립 작성자 평가 없이 Shortcut 완전 제거 또는 최종 배포 성공을 주장하지 않는다.

**이번 실험은 낮은 증강 비율부터의 **전처리만** 비교한다.** Normal 최종 256×128/alpha0.03 HP 적용, Hand-made 학습 비중 변화, 새로운 480장 검증은 별도 연구 질문이다.

결과 예정 위치: Combined/results/Balanced_Scale_Stroke/ 와 Combined/scripts/exp_cmb005_balanced_scale_stroke.py. 사용자 승인 전 main 병합/배포 금지.
