# EXP유010~016 — 크기 Shortcut 분석 결과 CSV

이 폴더에는 **실측 CSV·요약 파일·3-Seed 집계 결과**를 보관합니다. [실험일지](../../logs/크기편향_원인분석/README.md)와 [원본 실행 코드](../../scripts/README.md)는 별도 위치에 있으며 여기서 링크로 연결합니다.

| 실험 | CSV 결과 | 실험일지 |
|---|---|---|
| EXP유010 | [크기분포](EXP유010_summary.csv) | [원인 분석](../../logs/크기편향_원인분석/EXP유010_크기분포_분석.md) |
| EXP유011 | [Size/CoM 교차](EXP유011_summary.csv) | [교차 분석](../../logs/크기편향_원인분석/EXP유011_Size_CoM_교차분석.md) |
| EXP유012 | [크기 개입](EXP유012_summary.csv) | [크기 분류 단서](../../logs/크기편향_원인분석/EXP유012_크기분류단서.md) |
| EXP유013 | [기하특징 4개](EXP유013_binary_4feature_summary.csv) | [기하 특징 이진 분류](../../logs/크기편향_원인분석/EXP유013_기하특징_이진분류.md) |
| EXP유014 | [정확도](EXP유014_accuracy_3seed_summary.csv), [크기 민감도](EXP유014_scale_flip_3seed_summary.csv) | [크기·획 증강 학습](../../logs/크기편향_원인분석/EXP유014_Shortcut_완화_학습.md) |
| EXP유015 | [26조건 전체 판정](EXP유015_26조건_전체판정.csv), [B·D·8 개입](EXP유015_26조건_B8D_전환율_전체.csv), [9개 Gate 합격](EXP유015_shortcut_gate_통과9조건_요약.csv), [OfficialVal 요약](EXP유015_OfficialVal_크기변형_요약.csv) | [78개 MLP 최적화](../../logs/크기편향_원인분석/EXP유015_증강비율_획강도_최적화.md) |
| EXP유016 | [23조건 결과](원본_실측_아카이브/EXP유016_60모델/all_23_conditions_ranked_gates.csv), [16개 클래스 배율별](원본_실측_아카이브/EXP유016_60모델/all_16class_size_flip_3seed.csv), [OfficialVal](원본_실측_아카이브/EXP유016_OfficialVal/officialval_scale_3seed_summary.csv) | [60~100% 실험일지](../../logs/크기편향_원인분석/EXP유016_고비율_크기증강_한계확인.md), [80-14/70-0 보존 결정](../../logs/크기편향_원인분석/EXP유016_80-14_70-0_두_후보_병행보존.md) |

## 원본 실측 아카이브

- [EXP유015 78모델 3-Seed 전체 기록](원본_실측_아카이브/EXP유015_78모델/SUMMARY.md)
- [EXP유015 OfficialVal 0.8~1.2배 검증](원본_실측_아카이브/EXP유015_OfficialVal/SUMMARY.md)
- [EXP유016 60개 신규 모델 3-Seed 집계](원본_실측_아카이브/EXP유016_60모델/SUMMARY.md)
- [EXP유016 OfficialVal 독립 데이터 출처 확인](원본_실측_아카이브/EXP유016_OfficialVal/SUMMARY.md)

실측 값·원본 Git blob SHA는 수정하지 않고 이동합니다. **80-14와 70-0 모델 가중치(joblib)는 EXP유016 실험 브랜치에 별도 보관하며 최종 모델에 적용하지 않았습니다.**
