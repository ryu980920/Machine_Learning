# 숫자·문자 정확도 역전 원인분석 — EXP유010·011·012

CoM-only, Size-only, Size→CoM에서 숫자와 문자 정확도가 역전되는 원인을 **세 가설에 대한 하나의 연구 시리즈**로 관리한다. 이 폴더는 `logs/가설A/`, `logs/가설B/`와 **동일한 깊이**에 위치한다. 연구 번호 010·011·012는 그대로 유지한다.

## 연구 목록

| 실험 | 검증 가설 | 원본 실측의 핵심 | 연구 기록 / 수치 |
|---|---|---|---|
| **EXP유010** | Training과 Hand-made의 크기 분포 차이 | 공식 B/D/F 긴 변 24px, 손글씨 20px 확인 | [실험일지](EXP유010_크기분포_분석.md) · [summary.csv](EXP유010_summary.csv) |
| **EXP유011** | Size-only의 숫자 붕괴는 CoM 및 재학습과 관련되는가 | 같은 Size 모델에서 평가 입력을 CoM으로 바꾸자 숫자 36.46%→73.81% | [실험일지](EXP유011_Size_CoM_교차분석.md) · [summary.csv](EXP유011_summary.csv) |
| **EXP유012** | MLP가 크기 관련 특징을 숫자·문자 구분 단서로 쓰는가 | 동일 이미지 0.8~1.2배 개입 시 숫자/문자 판단 경향 역전 | [실험일지](EXP유012_크기분류단서.md) · [summary.csv](EXP유012_summary.csv) |

각 실험일지는 **사전 계획, 실험 절차, 실제 결과, 해석 및 한계, 재현 방법**을 한 문서에 통합한다. 각 CSV는 이전 GitHub 실험 폴더의 파일 내용을 **변경 없이 옮긴 핵심 지표 요약**이다.

## 재현성 및 데이터 경계

- **실제 완료된 공통 GitHub Actions:** [EXP유010~012 원본 분석 #43801194796](https://github.com/ryu980920/Machine_Learning/actions/runs/43801194796)
- **재사용된 EXP유006의 9개 모델:** [원본 학습 #43785390739](https://github.com/ryu980920/Machine_Learning/actions/runs/43785390739)
- **공통 원본 코드:** [EXP010 크기 분포](../../scripts/exp010_geometry_distribution.py) · [EXP011/012 모델 고정·크기 개입](../../scripts/exp011_012_causal_probes.py) · [종합 집계](../../scripts/exp010_012_aggregate.py)
- **전체 전수 결과:** 위 GitHub Actions의 Artifacts (보존 기간 제한). 요약 CSV는 영구 GitHub 기록이나 전체 예측 결과를 대체하지 않는다.
- **분석 조건:** Official Train 76,800장, Official Validation 19,200장, 기존 작성자 4명의 Hand-made 2,560장; Seed 42·43·44.
- **안전 경계:** 기존 체크포인트를 고정해 평가했으며 새 학습, 공식 Original Test·교수님 Test 접근, 최종 Pipeline 변경은 없다.

[Normal 메인 안내](../../README.md) · [가설 A](../가설A/README.md) · [가설 B](../가설B/README.md)
