# MLPClassifier / Normal — 실측 결과

실제 실행에서 산출한 Accuracy·Precision·Recall·F1 CSV, Confusion Matrix, Learning Curve, 학습/전처리/추론 시간과 오류 이미지 등을 저장한다.

각 파일은 [logs/](../logs/README.md)의 해당 실험일지와 연결해 결과의 출처를 명확히 한다. 원본·검증·Test 데이터가 무엇인지 파일 또는 로그에 표기한다.

- **[Pixcel_Attribution/](Pixcel_Attribution/README.md)**: EXP유001(1·2단계)의 실제 선택 표본 CSV, 2,560장 예측 검증 CSV, 픽셀/IG NPZ, 이미지 19장(대표 2장 + 상세 17장), 실행 메타데이터. 기존 모델을 변경하지 않은 분석 결과이며 성능 개선 수치를 포함하지 않는다.

- **[Input_Sensitivity/](Input_Sensitivity/README.md)**: EXP유002(3단계)의 456,960행 예측 CSV.gz, 조건별 정량표, 106장 집중 사례, 원본/변형 PNG, 학습 모델 체크포인트. *진단 목적이며 개선 모델의 Test 결과가 아니다.*

- **[Spatial_Response/](Spatial_Response/README.md)**: EXP유003의 공식/손글씨 **98,560장 특징 CSV.gz**, **130,560행 대응 입력/은닉층 반응 CSV.gz**, 클래스·작성자별 공간 분포, **72개 IG 지도**, 그림 및 원본 메타데이터. *자동 전처리 개발/모델 성능 개선 실험 아님.*


- **[Automatic_Centering/](Automatic_Centering/README.md)**: **EXP유004 전체 단일 결과 폴더**. 공식 Val **19,200장·Hand-made 2,560장** 3개 모델 정확도/CM/정렬 오차, CoM·Bounding Box 후보 모델 2개, 고정 MLP의 **원본 대 CoM 이동** 및 동일 CoM 입력의 재학습 효과, Hand-made **2,560장 × 4조건** 전수 예측, 5/F·B/8 IG. 본 실험과 픽셀 판단 분석을 **19개 원본 파일**로 통합. *최종 Pipeline 채택은 보류.*
