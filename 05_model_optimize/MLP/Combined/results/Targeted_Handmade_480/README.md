# EXP-CMB-010 — 신규 손글씨 480장 추가 Training 실측 원본

[품질 검사·원본](../../logs/학습데이터_추가/EXP-CMB-010_01_Input_QA_and_Train_Plan.md) → [**24개 학습 실측 보고**](../../logs/학습데이터_추가/EXP-CMB-010_02_Added_480_Training_Results.md) · [GitHub Actions](https://github.com/ryu980920/Machine_Learning/actions/runs/44274109988)

## 측정 결과

| 지표 | 기존 78,720장 | 480장 추가 79,200장 |
|---|---:|---:|
| Writer-held-out OOF Accuracy | **92.4089%** | 92.2917% |
| Writer OOF Macro F1 | **0.924018** | 0.923033 |
| OfficialVal Accuracy | **97.0972%** | 97.0677% |

- 기존 오답 186건 교정, 기존 정답에서 신규 오답 195건, 순 9건 손실.
- **7→9: 22→24**, **9→7: 3→10** 악화. B→8: 44→40, 8→B: 45→43, D→0: 77→46 개선. 0→D: 20→29 악화.
- **최종 결정:** 데이터 원본은 별도 Training 후보로 유지하되 **480장 전체 추가 조건을 영구 모델에 채택하지 않음**. 기존 main·배포 모델 불변.

## 실제 원본

- [`summary.json`](summary.json): 24개 Fold 원본 Manifest를 검사한 후 집계한 Writer, Seed, 모든 16클래스 및 주요 혼동쌍·대응 오답 교정/유발·선정 기준.
- [`summary.csv`](summary.csv): 두 조건 요약.
- [`class_error_comparison.csv`](class_error_comparison.csv): 전체 16개 클래스의 오류수·해당 클래스 오분류율·추가 전후 차이.
- GitHub Actions 아티팩트: Fold별 `cmb010-base-s*-w*`, `cmb010-add480-s*-w*` (총 24개, 가중치·예측 CSV·manifest) 및 `cmb010-add480-final-audit`; 보존 기간은 30일.

신규 손글씨 작성자 ID가 기록되지 않았고 기존 개발용 Writer OOF가 반복 사용됐다는 점 때문에 결과는 **개발용 비교**이다. Original Test·교수님 Test·기존 438장 평가 데이터를 이번 학습/선정에 사용하지 않았다.
