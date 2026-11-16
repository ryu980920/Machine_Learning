# Combined 가설 B · EXP유003 / EXP-CMB-006 — 40~50% Size × Stroke 전처리 후속 실험

> **상태: 사전등록 / 40·50% 실측 결과는 아직 없음.** [직전 EXP-CMB-005 실제 결과](EXP유002_Balanced_Scale_Stroke_Final_Results.md)와 [원본 10조건 집계](../../results/Balanced_Scale_Stroke/full_comparison/SUMMARY.md)를 기준으로 한다.

## 연구 질문

직전 EXP-CMB-005에서 30%까지 Official Training의 클래스 균형 크기 증강을 늘리자 Hand-made OOF 정확도 및 크기 변화 시 숫자·문자 계열 판단 안정성이 개선되었다. 그러나 공식 Validation 정확도는 감소하고, 특정 작성자·클래스에서 남은 혼동이 있었다. **40%, 50%에서도 동일한 추세가 지속·포화·역전되는지, Stroke 혼합이 추가 도움이 되는지**를 비교한다.

**성공 기준은 특정 Accuracy/F1 숫자가 아니다.** Size 비율/Stroke 강도/작성자/클래스별 결과와 공식 Validation·크기 개입·새로 발생한 오류를 근거로 결과 기전을 설명할 수 있는지 판단한다.

## 실행 전 확정한 여섯 조건

| Condition | Official Training Size 증강 | Stroke 강도 | 나머지 |
|---|---:|---:|---|
| scale40_stroke00 | 40% | 0% | CoM |
| scale40_stroke14 | 40% | 14% | CoM |
| scale40_stroke28 | 40% | 28% | CoM |
| scale50_stroke00 | 50% | 0% | CoM |
| scale50_stroke14 | 50% | 14% | CoM |
| scale50_stroke28 | 50% | 28% | CoM |

Normal EXP유015의 nested_subset(40%,50%) 사용, 각 클래스 균형 목표 BBox 긴 변 16/18/20/22/24px. Official Training은 원본 76,800장을 1:1 대체하므로 전체 학습 이미지 수 불변. Scale→Stroke(같은 대상만, 3×3 십자형 morphology+topology guard)→CoM을 적용. 학습 Hand-made 1,920장은 CoM-only, 보류 작성자 및 공식 Validation은 CoM-only 평가. 배율 0.8~1.2는 모델 고정 진단에서만 변경.

**MLP HP 변경 금지:** PixelScaler255 + 기본 MLPClassifier(random_state=Seed 42·43·44), hidden(100,), ReLU, Adam, alpha=1e-4, learning_rate_init=0.001, max_iter=200 등 EXP-CMB-005와 완전 동일. Normal EXP유017의 HP 튜닝 및 기타 데이터 증강·학습 표본 수 변경 없음.

작성자 41~44 Leave-One-Writer-Out 4-Fold × Seed 42/43/44 × 6조건 = **72 신규 MLP fits**. EXP-CMB-005의 0~30% 10조건은 재학습하지 않고 기존 모델 예측·요약·SHA 원본을 대응 비교에 사용한다.

## 사전 정의 분석

1. 각 조건의 Hand-made OOF 전체/숫자/문자 Accuracy, Macro F1, 작성자별 정확도와 OfficialVal 19,200장 정확도·클래스별 변화.
2. 고정 모델에서 OfficialVal과 OOF의 동일 이미지를 0.8/0.9/1.0/1.1/1.2배 조작·CoM 재정렬: 숫자×1.2 Family Flip, 문자×0.8 Family Flip, B·8·D 최악 작성자/Seed 및 전체 16클래스.
3. 16개 클래스 실제 기준 오류율(OOF 각 160장/Seed), 가능한 **240개 오분류 방향** 및 숫자→문자, 문자→숫자, 동일 계열 혼동. 특히 0↔D, 8↔B, 5→F, 7→9, 2→1 증가·감소 여부.
4. 동일 Seed·작성자·이미지 순서로 과거 30%-0, 30%-14, 30%-28을 새 여섯 조건과 대응시켜 **교정된 오류와 새로 틀린 오류**를 함께 세고 10~50% 추세를 비교.
5. OfficialVal 하락, 초기 데이터와 신규 작성자 사이 일반화, 클래스/작성자별 반례, 보간으로 인한 획·잉크 변화, topology 거절 수까지 보고한다.

## 재현성·반영 원칙

- Official Train 76,800 / OfficialVal 19,200 / Fold당 Hand-made Train 1,920 / Held-out 640. 원본 SHA·레이블·Fold 정답 순서 검증. 3 Seeds 반복. Original Test·교수님 Test·Normal 신규 480장 사용하지 않는다.
- 기존 EXP-CMB-005 코드를 수정하지 않고 실험6 래퍼로 동결된 이미지 변형·재학습·5배율·오류 분석을 재사용한다. 결과 및 체크포인트는 실험6 브랜치/Artifacts에서 추적.
- 연구 목적 달성 ≠ 새 모델 최종 채택. 독립 신규 작성자 Test 없이 Shortcut 완전 제거 및 최종 성능을 주장하지 않는다.
- **EXP-CMB-006은 별도 브랜치/PR에 기록. 사용자 별도 승인 전 main에 병합하거나 최종 배포 모델을 변경하지 않는다.**

## 실험 전 기준 수치(새 실험 결과 아님)

EXP-CMB-005 Size30/Stroke28: Hand-made OOF 90.716%, 공식 Validation 96.117%, 숫자×1.2 Family Flip 9.19%. Size0 CoM-only: OOF 84.883%, 공식 Validation 96.733%, 숫자×1.2 Family Flip 69.77%. 본 실험에서 이 수치보다 크거나 작다는 이유만으로 연구 성공/실패를 결정하지 않는다.

## 실제 진행·결과

실험 코드를 실행하기 전에는 40/50% 조건의 실제 성능이 **확인되지 않았다**. 성공 실행 로그·3-Seed 집계·16개/240 방향 분석을 받은 뒤 실측을 별도 결과 문서로 기록한다.