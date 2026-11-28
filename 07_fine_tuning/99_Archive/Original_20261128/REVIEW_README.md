# SVC + KNN Fine-Tuning 검토 패키지

## 열람 순서
1. `04_final_comparison/FINAL_REVIEW.md` — 현재 실제 실행 결과와 미완료/탈락 범위
2. `00_baseline/STEP00_Baseline.md` — 원본 무결성, 1,280개 예측 일치
3. `01_error_analysis/STEP01_Error_Analysis.md` — 115개 고유 오류 분석
4. `02_SVC/EXP-FT-SVC-001.md` ~ `004.md` — SVC별 가설·DOE·실측·판정
5. `03_KNN/EXP-FT-KNN-001.md` ~ `005.md` — KNN별 가설·DOE·실측·판정
6. `06_reports/` — 모델별 종합 보고서
7. `Fine_Tuning_Reproduction.ipynb`, `Final_Inference.ipynb`, `extract_inference.py`, `run_svc*.py` — 코드 및 재현 자료

## 재현 및 데이터
`INPUT_MANIFEST.json`의 입력 ZIP 해시가 일치해야 한다. 개인 검토용 입력 ZIP은 `inputs/`에 보존한다. `requirements.txt`의 원래 고정 버전을 사용한다. Notebook은 실행 셀의 재구성본으로 새 환경에서 end-to-end 독립 검증하지 않았다. 전체 학습은 시간과 메모리가 충분한 로컬 환경에서 개별 스크립트로 실행한다. 원본 파일은 수정하지 말고 새로운 출력 폴더를 사용한다.

모든 DOE의 예측 CSV는 원본 평가 이미지 순서에 맞춰져 있으며 `01_error_analysis/paired_predictions.csv`의 collector/index/true를 통해 이미지와 연결된다. Pilot과 Full의 학습 행 수, 평가 규모, 공식 평가 방식 및 시간 측정 방식은 다르므로 함께 비교할 때 해당 범위를 확인한다.

## 배포 금지
현재는 사용자 검토용이다. 이 ZIP에는 비공개 평가 이미지 및 기존 체크포인트가 포함될 수 있으므로 전체 ZIP을 공개 저장소에 올리지 않는다. 사용자 승인 전 GitHub 변경 없음. 공식 Original Test 및 교수님 Test는 사용하지 않음.
