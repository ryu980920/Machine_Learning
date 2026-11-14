# EXP유003 — Spatial Distribution & MLP Response (Normal MLP)

**실제 성공 실행의 원본 산출물:** [GitHub Actions #43634701305](https://github.com/ryu980920/Machine_Learning/actions/runs/43634701305), Artifact \`normal-mlp-exp003-spatial-response-review\`. 현재 폴더의 수치는 성공한 실행 산출물에 근거한다. EXP유001·002의 기존 결과는 유지한다.

- [실험 로그 — EXP유003_Spatial_Response.md](../../logs/EXP유003_Spatial_Response.md)
- [분석 코드 — Spatial_Response_Analysis.py](../../scripts/Spatial_Response_Analysis.py)
- [재현용 GitHub Actions 수동 실행](../../../../.github/workflows/mlp_spatial_response_stage4.yml)

## 실행 요약

| 항목 | 성공 실행에서 검증된 값 |
|---|---|
| 모델 | EXP유002에서 저장한 **고정 Normal MLP 체크포인트**, 재학습 없음 |
| 공식 Training | **76,800장** (원본 72,521 / 증강 4,279) |
| 공식 Validation | **19,200장** |
| Hand-made | **2,560장** (작성자 41~44) |
| 공간 특징 측정 | **98,560장** |
| 대응 입력 분석 | **130,560행** (6조건 × 21,760장) |
| 대표 은닉층 반응 | **636행** (106장 × 6조건) |
| 대표 IG 지도 | **72개** (18장 × 4조건) |
| 기존 예측 검증 | 공식 CM 차이 합 **0**, Hand-made 개별 예측 차이 **0** |
| 공식/교수님 Test | **미사용** |
| 자동 전처리 도입 / 모델 성능 최적화 | **미실시** |

## 파일 안내

### 원본 계측·비교 테이블

| 파일 | 설명 |
|---|---|
| [all_image_geometry.csv.gz](all_image_geometry.csv.gz) | 공식 Train/Validation + Hand-made **98,560장**의 중심 X·Y, Bounding Box, 여백, 획 밀도, 임계값별 특징 — gzip CSV |
| [geometry_distribution_summary.csv](geometry_distribution_summary.csv) | 출처/작성자/클래스별 분포 중앙값·평균·분위수 **153행** |
| [per_class_geometry_differences.csv](per_class_geometry_differences.csv) | 공식 Train 대비 16개 클래스/작성자별 중앙값 및 Wasserstein distance **96행** |
| [official_class_references.csv](official_class_references.csv) | 공식 Train 기반 클래스별 중심·너비/높이 중앙값과 IQR 스케일 |
| [paired_geometry_model_response.csv.gz](paired_geometry_model_response.csv.gz) | 동일 모델의 원본/변형 입력·공식 참조 거리·은닉층·출력 margin·오류 결과 **130,560행** |
| [spatial_response_summary.csv](spatial_response_summary.csv) | 공식/Hand-made별 6조건 대응 반응 및 정확도 **12행** |
| [spatial_response_groups.csv](spatial_response_groups.csv) | 숫자/문자 및 손글씨 작성자별 대응 요약 |
| [geometry_proximity_vs_outcome.csv](geometry_proximity_vs_outcome.csv) | 공식 학습 분포 접근 여부와 교정/신규 오류 연관성 집계 |
| [geometry_proximity_by_writer_class.csv](geometry_proximity_by_writer_class.csv) | 위 관계를 **클래스·작성자·기존 정오답·접근 여부**로 구분한 **1,777행** |
| [focus_hidden_responses.csv](focus_hidden_responses.csv) | 실제 106개 핵심 이미지에서 6조건별 은닉층 반응 **636행** |
| [focus_ig_completeness.csv](focus_ig_completeness.csv) | Integrated Gradients 근사량과 completeness 잔차 **72행** |
| [focus_ig_maps.npz](focus_ig_maps.npz) | **72개** 입력 및 signed IG 배열 |
| [run_metadata.json](run_metadata.json) | 모델 SHA-256, 데이터 해시, 사용 조건·개수·실행 환경 및 Test 미사용 증빙 |
| [SUMMARY.md](SUMMARY.md) | 성공 실행이 자동 생성한 요약 |

### 시각화 자료

| 파일 | 설명 |
|---|---|
| [center_scatter.png](center_scatter.png) | 공식 학습과 손글씨 중심 좌표 비교 |
| [bbox_width_distributions.png](bbox_width_distributions.png) | 주요 클래스 Bounding Box 너비 분포 |
| [per_class_spatial_difference.png](per_class_spatial_difference.png) | 16개 클래스별 중심 X 및 가로 너비 중앙값 차이 |
| [hidden_response_by_transform.png](hidden_response_by_transform.png) | 변형별 은닉층 100차원 표현 변화 |
| [focus_ig_grids/](focus_ig_grids/) | 5개 그림: 실제 손글씨/변형 입력과 고정 경쟁 클래스 IG 지도 |

## 연구적 해석과 주의

- **정답 클래스별 공식 학습 분포 거리**는 정답을 알고 계산한 연구용 지표다. 이 값을 정답을 모르는 실제 Test 전처리에 그대로 사용하면 정보 누출에 해당할 수 있다.
- Hand-made는 **4명의 작성자 2,560장**으로 이미 여러 실험에 사용했다. 현재 Hand-made에서 성능이 좋은 특정 이동·축소값을 고르는 것은 **개발 데이터 선택 과적합 위험**이 있다.
- 중심·너비 분포 차이와 오류 교정의 연관성은 확인했지만 **모든 클래스/작성자에서 성립하지 않았고, 95% 축소는 공식 분포와의 평균 거리가 커지면서도 정확도가 개선되는 반례**가 있다.
- 원본 Test를 사용한 적이 없으며, 이번 연구에서 **새 중심 정렬 Pipeline, 전처리 채택 또는 최적 모델 개발은 하지 않았다**.
- 모델 파일은 EXP유002 결과 [normal_mlp_refit_stage3.joblib](../Input_Sensitivity/normal_mlp_refit_stage3.joblib)에서 관리한다. 외부에서 받은 임의의 joblib 파일을 역직렬화하지 않는다.

**재현:** Python 3.13, NumPy 2.3.5, scikit-learn 1.8.0, SciPy 1.17.0, matplotlib 3.10.7.

\`\`\`bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
python 05_model_optimize/MLP/Normal/scripts/Spatial_Response_Analysis.py \
  --out 05_model_optimize/MLP/Normal/results/Spatial_Response --batch 1024
\`\`\`
