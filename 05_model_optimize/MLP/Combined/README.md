# MLPClassifier — Combined 모델 최적화

**진행 상태:** EXP-CMB-001 통제 Baseline 완료 (2026-11-15), 후속 최적화 미실시.

## 등록된 실험

- **[EXP유001 — Combined MLP 4-Fold CV Baseline](logs/EXP유001_Combined_MLP_4Fold_CV_Baseline.md)** — Normal vs Combined, 손글씨 작성자 4명 OOF, 공식 Validation 보존 점검
- **[EXP-CMB-001 결과 폴더](results/EXP_CMB_001/README.md)** — 통합 지표·전체 OOF 예측·분할 인덱스·모델 ZIP·최종 Viridis CM
- **[재현·시각화 실행 코드](scripts/README.md)** — 학습, 분할 검증, 4개 CM 재생성

## Baseline 결과 (Hand-made OOF 2,560장)

| 지표 | Normal | Combined |
|---|---:|---:|
| Accuracy | 62.5781% | **81.2500%** |
| Macro F1 | 0.623046 | **0.812043** |

Combined는 **76,800장 공식 Train + 각 Fold의 Hand-made 1,920장**으로 다시 학습하여, 제외한 작성자 640장씩을 검증했다. 위 OOF 결과는 서로 다른 4개의 모델을 합친 **개발 평가**이고, 독립 Test 또는 단일 최종 모델 성능이 아니다. 공식 Original Test/교수님 Test는 사용하지 않았다.

## 기록 및 재현 원칙

- `logs/`: 문제 → 가설 → 수행 방법 → 실제 지표 → 판단 → 다음 실험
- `scripts/`: 재실행 스크립트, 픽셀 스케일러, 검증 코드, 버전 고정 패키지
- `results/EXP_CMB_001/`: **핵심 결과 11개** (지표·원본 예측·Loss·manifest·분할·모델 ZIP 6개 + Viridis CM 4개 및 Loss PNG 5개)
- 새 최적화 실험에서는 Training과 Validation에 같은 필체 작성자나 이미지를 중복 사용하지 않는다.
- [05 전체 규칙](../../README.md) · [실험일지 양식](../../EXPERIMENT_TEMPLATE.md) · [연구 흐름도](../../../01_project/RESEARCH_FLOW.md)

**최종 시각화:** Viridis Count 2개와 Normalized 2개. 최초 생성된 Blues CM, 비교 패널, SVG 및 재생성 가능한 중복 CM CSV는 결과 저장소에 남기지 않는다.
