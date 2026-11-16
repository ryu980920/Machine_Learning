# Normal MLP · 가설 A EXP유004 — 전처리 모델 재학습 실험 (브랜치 검토용)

## 연구 질문
공식 Training에서 추정한 획 굵기/글자 가로폭을 기준으로 미지의 입력을 레이블 없이 정규화하면, CoM을 유지하면서 문자 정확도를 높일 수 있는가?

- **대상 모델:** MLPClassifier만 사용. 앙상블·CNN 사용하지 않음.
- **분할:** 공식 Training 76,800장 / Official Validation 19,200장. 공식 Original Test·교수님 Test 미사용.
- **Hand-made:** 기존 작성자 41–44의 2,560장. 전처리 피팅·학습에는 사용하지 않고 개발 평가에만 사용.
- **정규화:** Training X만 사용해 획 굵기 skeleton/distance transform과 가로폭 분위수 q20/q80 추정. 새 입력은 레이블·작성자 없이 계산. 형태 연결 보존 검사 및 최종 CoM 적용.
- **후보:** C0 CoM, C1 stroke 0.15, C2 stroke 0.30, C3 horizontal width 0.15, C4 stroke+width 0.15.
- **학습 반복:** 조건마다 seed 42·43·44, MLP 은닉층 100 ReLU, Adam, max_iter=200; 다른 하이퍼파라미터 고정.
- **성능 1차 기준:** 작성자 기존 Hand-made 문자≥80%, 숫자≥85%, 공식 Validation 전체≥97.30% **동시** 충족. 방법별 3-seed 평균이 모두 기준 이상이고 개별 2/3 seed 이상 동시 충족 시 개발 통과.
- **최종 일반화 검증:** 별도 미관찰 작성자 및 최종 Test로만 확정. Test 결과를 보고 정규화 기준 재피팅 금지.
- **출력:** 매 seed 분류 결과·방향별 혼동·모델·실제 손글씨 이미지·전처리 기준값과 해시. Actions Artifact + 실험 브랜치 검토 메트릭.
- **배포 정책:** 이 브랜치는 실험 전용이며 Draft PR은 검토용. 사용자 승인 전 main 변경/병합 금지.

학습 작업은 이 브랜치의 GitHub Actions 'Normal Hypothesis A EXP004 Stroke and Width MLP Training'에서 실행한다. 성공 여부와 실제 성능은 해당 Run의 Artifact 및 SUMMARY.md로 판단한다. 코드 등록만으로 통과라고 기록하지 않는다.
