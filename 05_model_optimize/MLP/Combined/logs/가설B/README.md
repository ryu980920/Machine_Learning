# Combined 가설 B — CMB-004~007 및 CMB-009 기하·전처리 연구

[전체 연구 흐름](../README.md) · [초기분석 CMB-002~003](../초기분석/README.md)

## 연구 질문과 진행 이유

Hand-made 숫자/문자의 글자 크기·위치·획 차이가 0↔D, 8↔B의 오분류에 영향을 주는지 탐색한다. **정규화의 효과를 먼저 분리(CMB-004)**한 다음, **훈련 증강 비율을 단계적으로 확장(CMB-005 → 006 → 007)**하여 크기 편향 감소와 다른 클래스 오답 증가 사이의 상충을 검증했다.

| 실험 | 바꾼 조건 | 가설과 결론 | 연구 문서 |
|---|---|---|---|
| **CMB-004** | Size-only / CoM-only / Size+CoM, 4-Fold 비교 | Size·CoM 기여 및 부작용 확인, 최종 모델 채택 보류 | [01 계획](EXP-CMB-004_01_Plan_Geometry_Normalization.md) → [02 최종](EXP-CMB-004_02_Final_Geometry_Normalization.md) |
| **CMB-005** | Size **0~30%** × Stroke 0/14/28%, 10조건·120모델 | 기본 MLP HP 고정, 입력 크기 편향 감소·클래스별 상충 관찰 | [01 계획](EXP-CMB-005_01_Plan_Size00_30_Stroke.md) → [02 중간](EXP-CMB-005_02_Interim_Size00_10.md) → [03 최종](EXP-CMB-005_03_Final_Size00_30.md) |
| **CMB-006** | Size **40~50%** × Stroke 0/14/28%, 6조건·72모델 | CMB-005와 합산 16조건 비교. 30%-28% OOF 최고치 미갱신 | [01 사전 계획](EXP-CMB-006_01_Plan_Size40_50.md) → [16조건 결과](../../results/Balanced_Scale_Stroke_Extension/full_0to50_comparison/SUMMARY.md) |
| **CMB-007** | Size **60~100%** × Stroke 0/14/28%, 15조건·180모델 | 총 31조건·16클래스·240방향 비교, 평균 Family Flip 개선과 8/B/D의 비대칭 오류·공식 Val 악화 확인 | [01 계획](EXP-CMB-007_01_Plan_Size60_100.md) → [02 중간](EXP-CMB-007_02_Interim_Size60_80.md) → [03 최종](EXP-CMB-007_03_Final_Size60_100.md) → [04 오분류 후속](EXP-CMB-007_04_Review_Top3_OOF_Error.md) |

## 실측 핵심 결론

- **CMB-004:** OOF Size-only 80.7813%, CoM-only 85.8594%, Size+CoM 88.9453%. 네 Fold OfficialVal 손실로 최종 Pipeline 미채택.
- **CMB-005:** Hand-made OOF 최고 **30%-28% = 90.716%**, OfficialVal 96.117%. 평균 크기 Family Flip은 크게 낮아졌으나 특정 클래스 오류 남음.
- **CMB-006:** 40~50%를 확장한 **16조건** 비교. 고비율이 모든 지표를 동시에 개선하지 못함.
- **CMB-007:** Size 60~100%에서 신규 180모델 학습 완료. 비교한 31조건의 OOF 최고치는 여전히 30%-28%이며, **8→B 증가 / B→8 감소**라는 오분류 재분배와 극단 100% 조건의 B/D 확대 취약성 확인.
- **후속 오분류율(OOF 상위 3개):** 30%-28% **9.284%**, 30%-14% **9.310%**, 30%-0% **9.427%**. [16클래스·방향별 상세](EXP-CMB-007_04_Review_Top3_OOF_Error.md).

## 결과 원본 및 데이터 경계

[CMB-004 정규화 결과](../../results/Geometry_Normalization/README.md) · [CMB-005 10조건 결과](../../results/Balanced_Scale_Stroke/full_comparison/SUMMARY.md) · [CMB-006 16조건 결과](../../results/Balanced_Scale_Stroke_Extension/full_0to50_comparison/SUMMARY.md) · [CMB-007 31조건 최종 CSV](../../results/Balanced_Scale_Stroke_HighFraction/full_0to100_comparison/SUMMARY.md).

- 모든 CMB-005~007 모델은 기본 MLP 하이퍼파라미터를 고정하고 Writer-held-out OOF를 비교했다.
- OfficialVal은 개발 데이터이고, 기존 Writer 손글씨 4명도 반복 관측했으므로 **최종 독립 Test 성능으로 해석하지 않는다**.
- Original Test·교수님 Test·Normal 신규 손글씨 480장은 CMB-007에서 사용하지 않았다.
- **PR #19를 포함한 과거 실험 PR은 `main` 병합 완료, 최종 배포 모델은 미교체**. 파일명·목차 정리는 실험 자체와 별도 문서 정리다.

## CMB-007의 후속 결정 및 CMB-008 연결

- **학습 증강 조합 선정:** 정상 입력 Hand-made OOF 정확도 우선 기준으로 **Size30%-Stroke28% + CoM**을 다음 Combined MLP HP 탐색의 고정 설정으로 사용한다. 이는 최종 테스트 일반화·모든 클래스 최고 성능 증명이 아니다.
- **하이퍼파라미터 연구 이관:** [EXP-CMB-008 사전등록](../하이퍼파라미터_최적화/EXP-CMB-008_01_Plan_30_28_Frozen_HP.md). 너비·깊이·L2·LR·종료 방식을 순차 평가한다. **신규 실험 결과가 아직 없으므로 어떤 HP도 추가 확정하지 않는다.**

## EXP-CMB-009 — 기존 전처리에 추가 변형을 적용한 재학습 검증 (2026-11-18)

- 기준: 기존 Size30%-Stroke28%-CoM, MLP (256,), Batch 128, 3 Seed × 4 Writer.
- 신규 train-only 20% 증강: Local Warp 상·하단 ±1px / Rotation ±3°·±5° / Shear ±0.05·±0.10. 기준선 포함 총 72모델.
- [**CMB-009 실측 결론**](EXP-CMB-009_01_Sequential_Preprocessing_Results.md): Rotation ±5° OOF **92.3698%**(기준 92.2786%)이나 9→7 악화 및 사전 가드레일 미통과. Shear ±0.05는 7↔9 감소하나 전체 OOF 하락. **조합에 사용할 검증 완료 조건이 없어 조합 재학습은 생략**했다.
- [**조건별 원본 CSV**](../../results/Sequential_Preprocessing/standalone_table.csv) · [GitHub Actions 72모델](https://github.com/ryu980920/Machine_Learning/actions/runs/44246719485). 438장·독립 Test·교수님 Test는 CMB-009 학습과 선택에 사용하지 않았다.
