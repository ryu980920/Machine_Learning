# MLPClassifier / Normal — 실측 결과

실제 실행에서 산출한 Accuracy·Precision·Recall·F1 CSV, Confusion Matrix, Learning Curve, 학습/전처리/추론 시간과 오류 이미지 등을 저장한다.

각 파일은 [logs/](../logs/README.md)의 해당 실험일지와 연결해 결과의 출처를 명확히 한다. 원본·검증·Test 데이터가 무엇인지 파일 또는 로그에 표기한다.

- **[Pixcel_Attribution/](Pixcel_Attribution/README.md)**: EXP유001(1·2단계)의 실제 선택 표본 CSV, 2,560장 예측 검증 CSV, 픽셀/IG NPZ, 이미지 19장(대표 2장 + 상세 17장), 실행 메타데이터. 기존 모델을 변경하지 않은 분석 결과이며 성능 개선 수치를 포함하지 않는다.

- **[Input_Sensitivity/](Input_Sensitivity/README.md)**: EXP유002(3단계)의 456,960행 예측 CSV.gz, 조건별 정량표, 106장 집중 사례, 원본/변형 PNG, 학습 모델 체크포인트. *진단 목적이며 개선 모델의 Test 결과가 아니다.*
