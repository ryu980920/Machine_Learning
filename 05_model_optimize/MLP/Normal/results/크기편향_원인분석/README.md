# EXP유010~018 — 크기 Shortcut 분석 결과 CSV

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


| EXP유017 | [OfficialVal 5배율·3 Seed](EXP유017_OfficialVal_5배율_3Seed_요약.csv), [동결 모델 SHA-256](EXP유017_동결모델_SHA256_Manifest.json) | [탐색 설계·선정 근거 통합 로그](../../logs/크기편향_원인분석/EXP유017_80-14_MLP_하이퍼파라미터_최적화.md) |
| EXP유018 | [455/480장·3-Seed 그룹별](EXP유018_3Seed_그룹별성능.csv), [16클래스 Precision·Recall·F1](EXP유018_16클래스_3Seed_지표.csv), [3개 혼동쌍 이전/신규 비교](EXP유018_이전CM_신규CM_주요쌍비교.csv), [원본 그룹별](EXP유018_주요쌍_원본그룹별.csv), [판독 제외 25장 인덱스/사유](EXP유018_제외25장_목록.csv) | [신규 손글씨 분석 실험일지](../../logs/크기편향_원인분석/EXP유018_신규손글씨480_판독필터_혼동쌍_진단.md) |

## 원본 실측 아카이브

- [EXP유015 78모델 3-Seed 전체 기록](원본_실측_아카이브/EXP유015_78모델/SUMMARY.md)
- [EXP유015 OfficialVal 0.8~1.2배 검증](원본_실측_아카이브/EXP유015_OfficialVal/SUMMARY.md)
- [EXP유016 60개 신규 모델 3-Seed 집계](원본_실측_아카이브/EXP유016_60모델/SUMMARY.md)
- [EXP유016 OfficialVal 독립 데이터 출처 확인](원본_실측_아카이브/EXP유016_OfficialVal/SUMMARY.md)

실측 값·원본 Git blob SHA는 수정하지 않고 이동합니다. **80-14와 70-0 모델 가중치(joblib)는 EXP유016 실험 브랜치에 별도 보관하며 최종 모델에 적용하지 않았습니다.**
