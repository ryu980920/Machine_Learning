# Combined — 가설 B: 공간·중심·크기·명암

## 검증 목적

이미지 중심·Bounding Box·잉크 비율과 공간적 밝기 분포가 Combined 작성자 보류 OOF 오류에 미치는 영향을 **단일 입력 조작 및 이후 필요 시 4-Fold 재학습**으로 검증한다.

## 현재 판정 — 후속 검증 우선

[초기분석 EXP유002](../초기분석/EXP유002_Error_Analysis.md)에서 전체 0/D/8/B 640장에 동일한 하나의 크기·중심 방향이 유효하다는 결과는 없었으나, **특정 작성자·클래스 내부의 오류-특징 연관성**을 발견했다.

- 작성자 **43번 8→B**: 잘못 분류된 8은 정답 8보다 더 넓고 Ink Ratio가 높음. 32/64/128 threshold 방향 유지.
- 작성자 **41번 D→0**: 잘못 분류된 D의 x 중심 위치가 정답 D보다 오른쪽.
- 작성자 **43번 0→D**: 잘못 분류된 0의 y 중심 위치가 더 위쪽. 단, 정답은 7장으로 표본이 작고 분석이 사후 지정됨.

**따라서 가설 B 우선 진행**. 그러나 이 관찰 결과는 **입력 변환이 정답률을 높인다는 인과적 증명은 아니다**.

## EXP유003 실제 결과와 후속 판단

[EXP유003 — Input Sensitivity](../초기분석/EXP유003_Input_Sensitivity.md)는 Normal 1개 + Combined 4개 저장 모델을 고정하고 위치·Size·획·밝기 10개 단일 변형을 2,560장 및 공식 개발 Validation에 실제 적용해 완료했다.

- **크기 95%:** Combined OOF 81.2500%→83.8672%(+2.6172%p), 하지만 순개선 67장 모두 43번에서만 발생. 공식 Validation 네 Fold 모두 하락, 문자 Accuracy도 하락. **일률적 Size 보정 채택 불가**.
- **획 얇게25%:** 82.3047%(+1.0547%p), 50장 교정/23장 신규 오류, 작성자 3/4 비악화, 공식 Validation 하락폭 −0.0521~−0.2031%p. **사전 후속 후보 선별 기준 통과**.
- **센터 ±1px 일괄 이동:** 네 방향 모두 OOF 전체 Accuracy 하락. 한 작성자·오류쌍의 교정이 다른 클래스의 오답을 증가시킨다.

**EXP유003 당시 연구 판단:** 획 얇게25%가 후속 재학습 후보였고, 95% 축소는 작성자 의존성이 컸다. 이후 **서로 다른 두 정규화 방법의 차이를 설명하기 위한 EXP-CMB-004를 별도로 먼저 실시**했다. EXP유003의 Accuracy를 EXP-CMB-004 재학습 수치와 직접 혼용하지 않는다.

## EXP유001 — Geometry Normalization (실제 재학습 완료)

- **[실험 전 사전등록](EXP유001_Geometry_Normalization.md):** Bounding Box Size 정규화 vs CoM 중심 정렬의 효과와 이유를 검증할 때, 정량 Accuracy·F1 하한 대신 작성자/클래스·입력 기하 변화·오류 전환·반례를 통해 설명 가능한지로 통과 판정.
- **[실행 결과 및 연구 판정](EXP유001_Geometry_Normalization_Results.md):** Size-only 80.7813%, CoM-only 85.8594%, Size+CoM 88.9453% (4-Fold OOF), 입력만 변경 vs 재학습 4-way 대조와 공식 Validation 손실 보고.
- **[실측 자료](../../results/Geometry_Normalization/README.md):** 16개 학습 모델·실제 CM/이미지·CSV·Loss·manifest.
- **결론:** Size는 문자에 유리하지만 숫자 악화, CoM은 숫자에 유리하지만 문자 악화, 두 방법 결합은 각 집단의 상충을 완화. 공식 개발 Validation 하락으로 최종 Pipeline 채택은 보류.

## EXP유002 / EXP-CMB-005 — 저비율 Size × Stroke 전처리 설명 검증 완료

- [실험 전 사전등록과 통제](EXP유002_Balanced_Scale_Stroke_Augmentation.md): 기존 MLP HP 고정, Size 0/10/20/30% × Stroke 0/14/28%, 3Seeds·4 Writer-held-out CV(10조건×3×4=120모델), 설명 중심 목표(정확도 절대 합격선 없음).
- [Stage1(0/10%) 결과](EXP유002_Balanced_Scale_Stroke_Stage1_Results.md) · **[전체 Stage2(20/30%) 및 최종 기전 분석](EXP유002_Balanced_Scale_Stroke_Final_Results.md)**.
- [전체 10조건×3Seed OOF·OfficialVal·Family Flip CSV/16클래스 전체/240오류 방향 원본](../../results/Balanced_Scale_Stroke/full_comparison/SUMMARY.md). 최고 탐색 OOF **30%-28% 90.716%**, 공식 Val **96.117%**. CoM-only 84.883%/96.733%. 30%-28% 숫자×1.2 Family Flip **69.77→9.19%**지만 특정 클래스·작성자 반례 잔존.
- [Stage2 실제 학습·집계](https://github.com/ryu980920/Machine_Learning/actions/runs/43871619789)는 마지막 git push 충돌로 run 결론 'failure'였으나, **[아티팩트 복원·검증 성공](https://github.com/ryu980920/Machine_Learning/actions/runs/43876627107)**으로 원본 학습을 다시 하지 않고 전 조건 결과를 실험 브랜치에 보존.
- **원인 설명 중심 연구 목적 달성 / 최종 모델 채택 보류** (공식 Validation 감소 및 독립 Writer 검증 부재). [Draft PR #16](https://github.com/ryu980920/Machine_Learning/pull/16)에서만 진행, 사용자 승인 전 `main`·배포 모델 변경 없음.

## EXP유003 / EXP-CMB-006 — 40~50% Size × Stroke 후속 실험

- **[실험 전 동결한 연구 질문·6조건·원인 설명 중심 판정](EXP유003_Size40_50_Stroke_Extension.md)**: 40/50% Size × Stroke 0/14/28%, 기본 MLP 하이퍼파라미터·3 Seeds·Writer-held-out 4-Fold 고정 (**72개 신규 모델**).
- 기존 [EXP-CMB-005 0~30% 원본](../../results/Balanced_Scale_Stroke/full_comparison/SUMMARY.md)을 재학습하지 않고 읽어 전체 **16조건·16클래스·240오분류 방향**과 30%-0/30%-28 대비 같은 이미지에서 새 오답·교정을 비교한다.
- [GitHub Actions EXP-CMB-006 실행](https://github.com/ryu980920/Machine_Learning/actions/runs/43881769100) / [검토용 Draft PR #18](https://github.com/ryu980920/Machine_Learning/pull/18).
- 실측 미완료 단계에서는 성능·성공 판정을 기재하지 않는다. 전체 결과는 실험 브랜치만 기록하고 사용자 승인 전 main/최종 모델 병합하지 않음.

참고: [Normal 가설 B](../../../Normal/logs/가설B/README.md). Normal 실험 결과를 Combined의 개선치로 간주하지 않는다.
