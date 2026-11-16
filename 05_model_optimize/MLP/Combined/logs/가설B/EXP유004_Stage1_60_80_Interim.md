# EXP-CMB-007 1차 Size 60~80% × Stroke 0·14·28% — 중간 학습 기록

> **상태: 1차 완료 · 전체 실험 미완료.** 성공한 GitHub Actions 개별 학습 로그에서 3 Seed × 4 heldout Writer Fold의 OOF/OfficialVal 평균을 확인했다. 아래 값은 개별 Job 로그의 6자리 반올림 값을 다시 평균 낸 **중간 집계**이므로 2차 완료 후 생성되는 `stage_high_60to100/comparison.csv`가 최종 원본 수치다.

- 성공 실행: [GitHub Actions #43891689090](https://github.com/ryu980920/Machine_Learning/actions/runs/43891689090)
- 1차 9조건 × Seed42·43·44 = **27개 Job 전부 성공**, Job당 4 Writer Fold = **108개 신규 MLP 학습 성공**. HP는 `PixelScaler255+MLPClassifier(random_state=seed)` 그대로 유지.
- 최초 실행 [#43891460569](https://github.com/ryu980920/Machine_Learning/actions/runs/43891460569)는 실행 YAML의 Python 래퍼 경로 오타로 학습 시작 전에 실패했다. 새 실행에서 경로 수정·학습 진행, 최초 실패를 학습 결과로 간주하지 않았다.

| Size | Stroke | Hand-made OOF (3Seed 평균) | OfficialVal (Seed×4Fold 평균) |
|---|---|---:|---:|
| 60% | 0% | 90.195% | 95.558% |
| 60% | 14% | **90.286%** | 95.435% |
| 60% | 28% | 90.156% | 95.423% |
| 70% | 0% | 89.674% | 95.030% |
| 70% | 14% | 89.570% | 95.138% |
| 70% | 28% | 90.130% | 95.020% |
| 80% | 0% | 89.987% | 94.970% |
| 80% | 14% | 89.870% | 94.813% |
| 80% | 28% | 90.221% | 94.678% |

**사전 대조:** 이전 EXP-CMB-005 Size30%-Stroke28%의 OOF 90.716%, OfficialVal 96.117%. 이번 1차 9조건의 정상 OOF·공식 Validation은 모두 이 기준보다 낮다.

**최종 판정 보류:** 숫자/문자 크기 배율 개입에 따른 Family Flip, 16클래스 개별 최악값·240오분류 방향·작성자별 새 오류/교정 수치를 아직 공식 집계하지 않았다. 크기 편향 안정성이 개선될 수 있으므로 위 정상 정확도만으로 60~80%의 실험 가치를 폐기하지 않는다.

**다음 단계:** 2차 90·100% × Stroke 0/14/28 × 3Seed × 4Fold의 72개 학습, 신규 15조건/역사적 16조건 합산 31조건 집계, `OfficialVal`과 Hand-made OOF 개별 Class Worst Flip 검증. 모두 완료하기 전에는 최종 모델을 변경하지 않는다.
