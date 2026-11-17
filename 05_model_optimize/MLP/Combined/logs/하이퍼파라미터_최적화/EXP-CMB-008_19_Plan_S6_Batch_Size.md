# EXP-CMB-008 · 19 S6 실험 계획 — Batch Size 200 / 128 / 64

> **사전등록 / 결과 전 작성.** 09B 강의자료의 13번 슬라이드에서 제안한 Batch Size 효과를 실험한다. Early Stopping 최적화는 종료했으며 **early_stopping=False 유지**한다.

## 가설
- 학습 데이터의 미니배치 크기를 기본 200에서 128 또는 64로 줄이면 Adam 업데이트와 확률적 학습 경로가 달라져 손글씨 작성자 일반화 성능이나 D→0, B→8 등 주요 혼동이 개선될 수 있다.
- 반대로 안정성이 줄거나, 데이터에 과하게 맞추거나, 학습 시간이 증가하면서 정확도가 하락할 수도 있다. **개선은 가설일 뿐 결론이 아님.**

## 모든 조건에서 동결
- 데이터: Official Train 76,800장 중 Size30%-Stroke28% 균형 증강 적용, CoM 중심, 학습에 포함된 수기 Writer 3인×640=1,920장 CoM만 적용. 총 78,720장/Fold.
- 검증: 나머지 Writer 1인 640장 보류, OfficialVal 19,200장 보조 지표. Original Test, 교수님 Test, 새 Hand-made 480장은 사용하지 않는다.
- 모델: sklearn 1.8.0 MLPClassifier, 단층 `(256,)`, ReLU, Adam, alpha=0.0001, initial LR=0.001, **early_stopping=False**, max_iter=200, tol=0.0001, n_iter_no_change=10.
- 변경 변수: **batch_size 단 하나** 200(현재 auto의 실제 값) / 128 / 64.
- 고정 Seeds 42/43/44 × LO Writer 41/42/43/44 = 각 조건 12모델, 총 **36 신규 Fits**. 재현성 감사를 위해 200도 재학습하고 기존 S5 문서 수치와 비교. 단, **서로 다른 실행을 억지로 동일한 모델의 예측으로 간주하지 않음**.

## 평가지표
1. **Primary** Writer-held-out OOF Accuracy(7,680예측), Writer별 Accuracy 및 Seed 분산.
2. **Critical confusions** D→0/B→8/8→B/0→D (각 정답 클래스 분모 480). 목표는 각 5% 미만(최대 23건).
3. 보호 클래스 0/8/B/D 총 오분류율 비증가, Writer별 정확도 최대 하락 1%p, OfficialVal 최대 하락 0.5%p, 클래스별 worst family flip 증가 5%p 제한.
4. Epoch, Training Loss/경고/시간, 기존 오답 교정과 새 오답.

## 사전등록된 선택 규칙
1. **strict:** 네 방향 각각 <5% + 모든 Non-Regression Gate 만족하는 후보가 있으면 우선한다.
2. strict 없으면 **fallback:** 200 기준선 대비 OOF 비하락, 주요 네 방향·0/8/B/D 클래스 오분류 비증가, Writer·OfficialVal·Flip 가드레일 통과 후보만 선정한다.
3. 통과 조건 중 OOF가 최고와 0.2%p 이내면 **D→0 오류 최소** → 다른 3방향 오류 합계 최소 → OOF 최고 → Seed/OfficialVal 순으로 비교.
4. 모두 악화되면 **batch_size=200 유지**. Batch=200 새 재실행과 기존 S5가 일치하지 않는 경우 원인 미확정 재현성 차이를 기록하고 새 실험의 동일 조건 200 기준으로 신규 후보를 비교한다.
5. 실험 후 결정은 **연구용 추천**이다. 기존 선택값이나 최종 학습 코드/모델 배포는 사용자의 보고·승인 전 수정하지 않는다.

## 산출물
- 36개 Fit별 `manifest.json`, 640 OOF 예측 CSV, 클래스별 scale family flip CSV.
- 종합 3조건 정량 CSV, 가드레일과 per-Writer 수치를 포함한 JSON, 오답 교정/유발 비교, **20 결과보고서**.
- 기존 S1~S5 로그는 덮어쓰지 않으며 09B의 Batch Size 실험으로 기록한다.
