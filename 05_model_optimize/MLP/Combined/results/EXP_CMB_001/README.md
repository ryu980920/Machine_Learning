# EXP-CMB-001 — MLP Combined 4-Fold CV 결과

이 폴더는 **2026-11-15에 실제 실행 완료된** 16-class MLP Normal/Combined 통제 비교의 재현 산출물입니다.

- **작성자 보류 OOF**: Normal Accuracy **62.5781%**, Macro F1 **0.623046**; Combined Accuracy **81.25%**, Macro F1 **0.812043** (총 2,560장).
- Combined는 서로 다른 작성자를 제외하고 학습한 모델 4개의 OOF 예측을 합한 **개발 평가**입니다. 단일 최종 Combined 모델 또는 외부 Test 결과가 아닙니다.
- 공식 v3 학습 데이터: 96,000장 중 Group-aware Train 76,800장 / Development Validation 19,200장. Fold별 Combined Train은 여기에 Hand-made 학습 1,920장을 더한 **78,720장**, 작성자 보류 Validation 640장입니다.
- 공식 Original Test와 교수님 Test는 사용하지 않았습니다.

| 파일 | 사용 목적 |
|---|---|
| `metrics_summary.csv` | 전체·작성자별·공식 개발 Validation 지표와 학습 시간·반복 수를 15개 행으로 통합 (행마다 데이터셋·평가 방식·모델·보류 작성자 명시) |
| `OOF_predictions.csv` | 모든 2,560장의 정답, 작성자, 원본 행 인덱스, Normal/Combined 예측 및 정오 여부. CM·클래스별 F1·오분류 목록 재생성의 근거 |
| `loss_curves.csv` | Normal + Combined 4-Fold의 iteration별 Training Loss. **Validation Loss가 아님** |
| `experiment_manifest.json` | 데이터 해시·분할·Fold 목록·환경·모델 설정·실행 URL·시각화 조건·검사 결과 통합 |
| `official_split_indices.npz` | 공식 데이터 Train/Validation 원본 행 인덱스 및 출처 그룹 ID. 정확한 분할 복원 |
| `trained_models.zip` | 실제 학습·보존한 Normal 1개, Combined 4개 `.joblib` 파일 (압축 내부 이름 유지; 해시는 manifest 참고) |
| `figures/` | 최종 Viridis Count CM 2개 / Row-normalized CM 2개 / MLP Training Loss PNG 1개 |

## 재현 방법

실행 코드는 [`../../scripts/`](../../scripts/)에서 확인할 수 있습니다. 새로운 학습 없이 CM만 재생성하려면 이 저장소 루트에서 다음 명령을 실행합니다.

```bash
python 05_model_optimize/MLP/Combined/scripts/render_cm_viridis.py
```

`OOF_predictions.csv`의 `writer`, `index_in_writer`는 기존 원본 `02_data/Hand_made/ml26_{writer}.npz`의 **0 기반 순서**입니다. 행별 예측에서 Accuracy, Macro F1 및 Confusion Matrix를 다시 계산할 수 있습니다.

**주의:** 모델 역직렬화(`joblib.load`)는 신뢰할 수 있는 파일에 대해서만 수행합니다. 모델 5개는 CV 평가 및 검증용이지, 별도 최종 모델 선정·재학습을 의미하지 않습니다.

GitHub Actions 원본 기록: [Run #43622998640](https://github.com/ryu980920/Machine_Learning/actions/runs/43622998640).
