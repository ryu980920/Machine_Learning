# EXP-CMB-009 — 추가 전처리 단독 검증 결과

[실측 보고서](../../logs/가설B/EXP-CMB-009_01_Sequential_Preprocessing_Results.md) · [학습 코드](../../scripts/exp_cmb009_sequential_aug.py) · [집계 코드](../../scripts/exp_cmb009_aggregate.py) · [GitHub Actions 44246719485](https://github.com/ryu980920/Machine_Learning/actions/runs/44246719485)

- **실행 완료:** 기준선·Local Warp·Rotation ±3°/±5°·Shear ±0.05/±0.10, 6조건 × 3 Seed × 4 Writer = 72모델. 모든 Fold 학습 성공.
- **원본 숫자:** [standalone_table.csv](standalone_table.csv). 해당 파일은 GitHub Actions 산출 CSV 원본의 수치와 열을 유지한다.
- **후속 조합 선택:** [selection.json](selection.json)에서 `selected=[]`, `combo_allowed=false`. 사전 기준 충족 후보가 없어 조합 모델은 학습하지 않았다.
- **후속 개발 기준:** 기존 **Batch 128 / Size30%-Stroke28% + CoM**. 다른 전처리를 최종 모델에 자동 적용하지 않는다.
- **아티팩트:** 개별 72개 `cmb009-*` (가중치·Fold별 원본 OOF·Manifest) 및 `cmb009-standalone-summary-selection`에 상세 JSON·원본 CSV가 저장되어 있다. Actions retention 30일이므로 필요 시 코드를 재실행해 복원한다.
- **보호:** Original Test, 교수님 Test, 438장은 본 실험 학습·후보 선정에 사용하지 않았다. 새 438장 이미지 NPZ도 이 저장소에 게시하지 않는다.
