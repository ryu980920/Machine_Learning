# 가설 A EXP유004 — 추가 조건 검증 (동일 실험 내 Stage2)

**이유:** Stage1에서 C2 획 정규화(강도 .30)가 3-seed 문자 평균 73.611%, 숫자 87.812%, 공식 Val 97.043%로 세 지표 중 숫자만 통과. B·D Recall이 여전히 낮다. 다음 실험 번호로 넘기지 않고 같은 EXP유004 안에서 전처리 강도와 형태를 확장.

- **C5:** 획 정규화 강도 .45 + 최종 CoM.
- **C6:** 획 정규화 강도 .60 + 최종 CoM.
- **C7:** 레이블을 모르는 새 입력의 Bounding Box 가로폭을 Official Train 중앙값 쪽으로 .35 완화 정규화 + CoM (C3의 20~80% 허용 구간과 비교).
- **C8:** 획 정규화 .30 → 가로폭 중앙값 정규화 .35 → CoM.
- **C9:** 공식 Train Bounding Box 긴 변 길이 중앙값으로 등방성 크기 정규화 → CoM.
- **공통:** Official Train 76,800장만 fit, Validation 19,200장과 Hand-made 2,560장 평가. Official Test·교수님 Test 미사용. MLPClassifier(100, ReLU, Adam), seed 42·43·44, 200 iter, Training 노출량 동일.
- **통과:** 3-seed 평균 문자≥80%, 숫자≥85%, 공식 Validation≥97.30% 및 최소 2/3 seed 동시 충족.
- **주의:** C7/C8/C9는 Stage1 Hand-made 결과에서 유도한 탐색 후보이므로 독립 새 작성자 일반화 성능이 아니다. 최종 Test를 보고 정규화 규칙을 조정하지 않는다.
- **보관:** 원본 테스트·이미지 근거와 모델은 Actions Artifact, 집계 지표와 실험에서 나온 실제 손글씨 그림만 실험 브랜치에 저장한다. 사용자 승인 전 main 병합 금지.
