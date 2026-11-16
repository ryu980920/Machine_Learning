# Combined 가설 B — EXP유004 / EXP-CMB-007: Size 60~100% × Stroke 실험 사전등록

> 상태: **실행 전 사전등록**, 새 성능은 아직 미측정. [EXP-CMB-005 0~30%](EXP유002_Balanced_Scale_Stroke_Final_Results.md), [EXP-CMB-006 40~50%](EXP유003_Size40_50_Stroke_Extension.md)의 병합된 결과를 동결된 역사적 대조로 사용한다.

## 연구 문제

Combined의 0~50% 전체 16조건에서는 Hand-made OOF Accuracy는 Size30% Stroke28%에서 최고(90.716%)였으나, Size50% Stroke28%에서 숫자×1.2 Family Flip은 6.10%까지 낮아졌다. Normal EXP유016에서 Size70%-14% 및 Size70%-0%는 16클래스/공식 Validation B·8·D 최악 Flip을 낮추는 후보였다. **Combined도 50% 초과에서 (A) 전체/문자 인식 성능, (B) 전체·개별 클래스 Flip, (C) OfficialVal 사이의 Pareto trade-off를 보이는지** 확인한다. 최고 OOF가 특정 비율 이하라고 상위 구간을 사전에 배제하지 않는다.

## 고정 실험 설계

| 순서 | Size 증강 비율 | Stroke 강도 | Seed | Held-out Writer | 학습 개수 |
|---|---|---|---|---|---:|
| 1차 | 60%,70%,80% | 0%,14%,28% | 42,43,44 | 41,42,43,44 | 108 |
| 2차 | 90%,100% | 0%,14%,28% | 42,43,44 | 41,42,43,44 | 72 |
| **합계** | 5종 | 3종 | 3종 | 4-Fold | **180** |

- 공식 Train 76,800장, 공식 Validation 19,200장, Fold별 다른 손글씨 작성자 1,920장 학습 + 보류 작성자 640장 평가(OOF 합계 2,560장/Seed).
- Normal EXP유016의 **50% 이하 역사적 IDs/target 보존 + 미선택 이미지 추가** nested_subset(y,ratio)를 정확히 재사용. Size 목표 BBox long side=16·18·20·22·24px, 각 클래스·target 별 증강 대상 균형 유지. 원본 76,800장 행 수 유지, 증강 대상 이미지는 1:1 교체.
- 선택 대상만 Normal EXP유016과 동일한 morphology topology guard Stroke 0·14·28% 적용. 원본 Official Train의 CoM 처리와 같은 방식으로 CoM. 손글씨 학습·OOF 평가 및 OfficialVal에는 CoM-only, Size 0.8/0.9/1.0/1.1/1.2 고정 모델 개입은 평가 시에만 수행.
- EXP-CMB-005/006 **기존 MLP와 동일한** Pipeline(PixelScaler255, MLPClassifier(random_state=Seed)) 기본 HP, (100,) single hidden layer, alpha=0.0001, ReLU/Adam, max_iter=200. Normal EXP유017 HP 재사용·변경, 데이터 분할 변경, 증강 이미지 추가하여 학습 행 수 변경 **금지**.
- Seed42/43/44과 각 Writer fold의 원본 SHA256, labels, 동일 OOF instance key(writer,index)를 검증. 이전 16조건은 **새로 학습하지 않고 main 원본 CSV 재사용**.
- Original Test, 교수님 Test, Normal에서 보유한 신규 480 Hand-made 세트 사용 금지. OfficialVal과 4 Writer OOF는 반복 개발용이지 독립 신규 평가가 아니다.

## 사전 판정·원인 질문 — Accuracy 합격선 없음

1. **성능 Pareto:** 0~100% 전체 31조건의 Hand-made OOF 전체/숫자/문자 Accuracy, macro-F1 및 Seed별 편차와 OfficialVal 정확도·차이. 30/50/70/80%의 일반 인식과 안정성의 상충 확인.
2. **크기 안정성:** 숫자 ×1.2 Family Flip, 문자 ×0.8 Family Flip 및 두 조건 이외의 모든 배율(0.8/0.9/1.1/1.2)의 16개 클래스 개별 Flip. Writer×Seed 최악과 OfficialVal Fold×Seed B/8/D 최악도 구분해서 기록.
3. **240개 방향·16클래스 오분류:** 0↔D,8↔B,5→F,7→9뿐 아니라 새로 늘어난 2→1·7→4 등 전체 방향 분석. 각 실제 클래스 n=160/Seed의 OOF 오분류율, 숫자→문자와 문자→숫자 및 내부 계열 오류율. 기존 30%-28%, 50%-0%, 50%-28%를 참고점으로 고정.
4. **새 오답/교정:** OOF 동일 Writer·Index·Seed 이미지에서 기존 30%-0/28, 50%-0/28와 새로운 15개 전처리 조건 간 corrected/induced/net과 클래스별·작성자별 변화를 계산. 평균 정확도 하나만으로 후보 선언 금지.
5. **엄격한 안정성 검토:** Normal EXP유016의 개별 클래스 위험 통제(B·8·D, 16클래스 최대), OfficialVal에서의 B·D 확대 오류 및 새 혼동을 함께 보고. 특정 B8D 기준을 뒤늦게 임의로 바꾸어 '통과' 후보로 만들지 않는다.
6. **100% 증강의 부작용:** 50%와 비교하여 OfficialVal 성능이 감소하거나 특정 클래스의 flip이 증가해 전체 가족 flip 감소와 역방향이 되는지 여부를 원본 실측으로 검증.

## 실행과 안전장치

- 코드는 기존 CMB-005 원본 함수에 Normal EXP유016의 >50% nested_subset 함수를 주입하는 얇은 실행 래퍼. 기존 CMB-005/006 코드와 역사적 결과를 수정하지 않는다. 1차 108개/2차 72개 학습, 원본 joblib·prediction·5-scale·이미지/seed/label·16클래스/240방향 CSV를 각 GitHub Actions Artifact에 보관한다.
- 신규 15조건 집계 및 기존 16조건과 교차 집계를 분리. 각 원본 Artifact를 검증하고 31조건의 paired error CSV, class flip·혼동방향 CSV를 병합 후 **별도 실험 브랜치에만 저장**한다.
- User 승인 전 **main 병합 금지**, 기존 최종 배포 체크포인트 및 Hyperparameter 변경 금지. 성공한 Actions 기록으로만 새로운 수치를 보고한다.

## 사전 등록 시점의 기준 관측

- Combined Size30 Stroke28: OOF 90.716%, OfficialVal 96.117%, digit×1.2 Flip 9.19%. Size50 Stroke28: OOF 89.883%, OfficialVal 95.622%, digit×1.2 Flip 6.10%.
- Normal EXP유016에는 60~100% 데이터가 존재하나 모델별 학습 조건과 Hand-made 학습 포함 여부가 다르므로 Combined 신규 결과로 간주하지 않는다.

## 실제 결과

현재 실행 전. 학습 성공/실패·정확도·전체 31조건의 최적점은 실제 GitHub Actions 종료와 CSV 검증 뒤 기록한다.