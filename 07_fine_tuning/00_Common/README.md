# 공통 데이터 및 실행 환경

Python 3.13 및 [requirements.txt](requirements.txt) 설치: `python -m pip install -r requirements.txt`. 기존 기록은 sklearn 1.8.0, seed 42, 2 threads. 같은 장비의 Batch1 1,280장×5회 비교 권장.

비공개 입력은 팀의 승인된 채널로 전달받아 저장소 밖에 보관하고 Notebook의 INPUT_DIR을 지정한다. 개인별 이미지, 예측, 바이너리는 공개 저장소에 올리지 않는다. 신뢰된 SHA-256 일치 joblib만 로드.

| 입력 | 목적 / 확보 방법 |
|---|---|
| Final_Validation_Private_Evaluation_Kit_20261126.zip | 팀 자료: final_benchmark 및 1,280장 평가 데이터 |
| final-validation-development-sources.zip | [Actions 원본](https://github.com/ryu980920/Machine_Learning/actions/runs/45775698001/artifacts/15095279779), Train 및 공식 Validation만 |
| Final_10_Model_Validation_Results.zip | 팀 자료: 역사적 1,280장 예측 일치 확인 (해당 모델 재현 시 필요) |
| final-fit-SVC_Combined.zip | [역사적 SVC C=1](https://github.com/ryu980920/Machine_Learning/actions/runs/45777648203/artifacts/15097273430), 현재 선정 모델이 아님 |
| final-fit-KNN_Combined.zip | [현재 KNN](https://github.com/ryu980920/Machine_Learning/actions/runs/45777648203/artifacts/15097432745) |
| C2_G2 model.joblib | 팀 리더가 별도로 전달할 현재 SVC 모델; SHA256은 선정 Manifest 참조. 기존 재현 검증에서 동일 SHA가 재생성되었으나 공개 저장소에는 없음. |

[입력 SHA-256](INPUT_MANIFEST.json) · [역사적 프로토콜](Historical_Protocol.json) · [후속 평가 기준](Followup_Protocol.json). Actions artifacts 만료 가능. 개발 1,280장 반복 사용; 독립 Test 아님. Original Test 및 교수님 Test 금지.

`prepare.py`는 입력 해시 확인·기존 전처리·모델별 baseline 재현을 담당한다. 역사적 재현과 후속 실험은 별도 output directory 사용. Notebook 전체 재실행은 이번 폴더 정리에서 수행하지 않았다.
