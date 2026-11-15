# MLPClassifier / Normal — 실험 로그

실험마다 `EXP[작성자코드][번호]_[실험주제].md` 형태의 **Markdown 파일**을 여기에 저장한다.

- **공통 양식:** [05 최적화 전용 양식](../../../EXPERIMENT_TEMPLATE.md) · [기존 상세 양식](../../../../03_experiments/EXPERIMENT_TEMPLATE.md)
- 기록 순서: **오류 분석 → 원인 가설 → 강의 내용에 근거한 개선 → 실행 조건 → 실측 결과 → 다음 실험**.
- 작성자·실행일·데이터 버전·모델 설정·비교 지표·실제 코드/그림 경로를 명시한다.
- **[EXP유001 — Pixcel_Attribution (1·2단계)](EXP유001_Pixcel_Attribution.md)**: 기존 Normal MLP의 검증된 분류 결과를 기준으로 98개 실제 이미지의 픽셀 기여도를 계산한 원인 진단 기록. **3단계 입력 변화 실험·최적화 미실시**.

- **[EXP유002 — Input_Sensitivity (3단계)](EXP유002_Input_Sensitivity.md)**: 공식 19,200장/손글씨 2,560장의 위치·밝기·굵기·크기 변화 민감도를 정량 검증. **20조건, 456,960개 예측**. 모델 전처리 도입 및 최적화는 아직 미실시.

- **[EXP유003 — Spatial Distribution & MLP Response](EXP유003_Spatial_Response.md)**: 공식 Training/Validation과 개발용 Hand-made의 중심·크기 분포 차이 및 예측·은닉층 반응을 고정 모델로 분석. **98,560장 특징·130,560건 대응 분석**, 자동 중심 정렬 도입/학습 최적화 미실시.


- **[EXP유004 — Automatic Centering 및 CoM 판단근거 통합 실험](EXP유004_Automatic_Centering.md)**: 같은 EXP유004에서 Center of Mass/Bounding Box MLP 학습, 공식 Validation/Hand-made 평가와 **기존 고정 모델의 CoM 이동 전후**, **같은 CoM 이미지에서 재학습 가중치 비교**, 5/F·B/8 Signed IG를 모두 기록. CoM MLP Hand-made **62.85→77.11%**; 이동만으로 숫자 **+25.69%p**, 문자 **−21.35%p**. **단일 로그·단일 결과 폴더**, 최종 Pipeline 채택 전 추가 검증 필요.
