# EXP-CMB-004 — 학습 효과/입력 변형 효과 분리 대조

**A** 기존 입력+기존 MLP / **B** 전처리 입력+동일한 기존 MLP
**C** B와 정확히 같은 전처리 입력+해당 방법으로 새로 학습한 MLP / **D** 원본 입력+새 MLP(훈련 분포 밖 대조)

A→B는 가중치를 고정한 입력 조작의 효과, B→C는 변형 입력을 고정한 재학습 효과를 분리하여 비교한다.
D는 배포용 Pipeline이 아니며 OOD diagnostics로만 읽어야 한다.

| Method | A old/raw | B old/transformed | C new/transformed | D new/raw (OOD) | Input only B−A | Model adaptation C−B |
|---|---:|---:|---:|---:|---:|---:|
| bbox_center | 81.250% | 81.211% | 80.312% | 80.234% | -0.039pp | -0.898pp |
| bbox_size | 81.250% | 83.906% | 80.781% | 75.742% | +2.656pp | -3.125pp |
| com_center | 81.250% | 77.891% | 85.859% | 72.656% | -3.359pp | +7.969pp |
| size_then_com | 81.250% | 77.852% | 88.945% | 70.625% | -3.398pp | +11.094pp |

### 그룹별 대비와 부작용

| Method | Group | A old/raw | B old/trans | C new/trans | B−A | C−B |
|---|---|---:|---:|---:|---:|---:|
| bbox_center | digits | 79.562% | 79.688% | 80.688% | +0.125pp | +1.000pp |
| bbox_center | letters | 84.062% | 83.750% | 79.688% | -0.312pp | -4.062pp |
| bbox_size | digits | 79.562% | 85.125% | 72.500% | +5.562pp | -12.625pp |
| bbox_size | letters | 84.062% | 81.875% | 94.583% | -2.188pp | +12.708pp |
| com_center | digits | 79.562% | 87.000% | 89.312% | +7.438pp | +2.312pp |
| com_center | letters | 84.062% | 62.708% | 80.104% | -21.354pp | +17.396pp |
| size_then_com | digits | 79.562% | 90.750% | 86.688% | +11.188pp | -4.062pp |
| size_then_com | letters | 84.062% | 56.354% | 92.708% | -27.708pp | +36.354pp |

**Interpretation constraint:** Even B→C compares independently refitted models, not a direct causal identification of one particular weight. OOF authors were repeatedly examined in earlier studies; extrapolations require new writers. Final project Test was not used.
