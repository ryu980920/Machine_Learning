# MLPClassifier / Normal — 실험 로그

실험마다 `EXP[작성자코드][번호]_[실험주제].md` 형태의 **Markdown 파일**을 여기에 저장한다.

- **공통 양식:** [05 최적화 전용 양식](../../../EXPERIMENT_TEMPLATE.md) · [기존 상세 양식](../../../../03_experiments/EXPERIMENT_TEMPLATE.md)
- 기록 순서: **오류 분석 → 원인 가설 → 강의 내용에 근거한 개선 → 실행 조건 → 실측 결과 → 다음 실험**.
- 작성자·실행일·데이터 버전·모델 설정·비교 지표·실제 코드/그림 경로를 명시한다.
- **[EXP유001 — Pixcel_Attribution (1·2단계)](EXP유001_Pixcel_Attribution.md)**: 기존 Normal MLP의 검증된 분류 결과를 기준으로 98개 실제 이미지의 픽셀 기여도를 계산한 원인 진단 기록. **3단계 입력 변화 실험·최적화 미실시**.

- **[EXP유002 — Input_Sensitivity (3단계)](EXP유002_Input_Sensitivity.md)**: 공식 19,200장/손글씨 2,560장의 위치·밝기·굵기·크기 변화 민감도를 정량 검증. **20조건, 456,960개 예측**. 모델 전처리 도입 및 최적화는 아직 미실시.
