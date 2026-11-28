# EXP-FT-SVC-002 — Hand-made Sample Weight

## 배경/목적/가설
Official weight=1, Hand-made weight=1,1.5,2,3,4. 모든 C,gamma 원래 값 고정. SW4가 Pilot guardrail 통과 후보 중 최고 정확도와 F1. SW2/3은 공식 Pilot 정답 2장 하락으로 Pilot guardrail 실패. SW4는 전체 학습 검증 대상으로 승격.
## 고정 조건
CoM→Deskew→Pixel:HOG20:80, RBF, seed42. 축소 Pilot 기준 8,528장: 공식 8,000장 및 Hand-made 원본/증강 512장, D/B 집중 증강 16장. 공식 개발 640장 및 Hand-made 개발 1,280장 동일 평가. 증강량 변경 시 학습 행 수만 변경. Pilot의 절대 수치는 full baseline 93.6719%와 직접 비교하지 않는다.
## 사전 통과 기준
동일 Pilot baseline 대비 정답 증가, Precision 유지, F1 증가, 공식 Pilot 정답 하락 1장 이내. Full 적용 시 공식 19,200장 최대 19장 하락, B/D/C pooled recall 개선, 0/8/6 각 2장 이하 하락, 기존 full 정확도/Precision/F1 유지·개선. 속도는 별도 판정.
## 실측 조건별 결과
| condition | train_rows | correct | Accuracy(%) | Precision(%) | F1(%) | official_pilot_correct | delta_correct_vs_pilot | fit_s | pilot_guardrail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_G1 | 8528 | 1153 | 90.0781 | 90.3571 | 90.0642 | 622 | 0 | 2.3599 | False |
| SW1.5 | 8528 | 1164 | 90.9375 | 91.1768 | 90.9125 | 621 | 11 | 2.3449 | True |
| SW2 | 8528 | 1166 | 91.0938 | 91.2853 | 91.0666 | 620 | 13 | 2.2713 | False |
| SW3 | 8528 | 1166 | 91.0938 | 91.2914 | 91.0737 | 620 | 13 | 2.3235 | False |
| SW4 | 8528 | 1167 | 91.1719 | 91.3744 | 91.1537 | 621 | 14 | 2.3912 | True |
## 분석
Baseline C1_G1을 재사용. `pilot_all_conditions.csv`에 전체 원본 수치와 두 비교 기준별 교정/퇴행 기록 보존. True는 Pilot 선별 기준 통과일 뿐 최종 채택 아님. Full 학습 미완료 조건은 최종 결과로 사용하지 않는다.

## 전체 학습 후속 실측
| condition | training_rows | correct | Accuracy(%) | Precision(%) | F1(%) | official_correct | corrected | regressed | accuracy_guardrails_pass | fit_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SW1.5 | 82080 | 1203 | 93.9844 | 94.2186 | 93.9653 | 18962 | 5 | 1 | True | 230.2562 |
| SW4 | 82080 | 1213 | 94.7656 | 94.9403 | 94.7613 | 18945 | 17 | 3 | False | 194.5528 |

SW4는 신규 1213장으로 최고지만 공식 18945장, -22장으로 사전 최대 -19장 기준 실패. 따라서 다른 Pilot 통과 후보 SW1.5를 추가 full 평가해 1203/1280 및 공식 18962로 보호 기준 통과 확인. SW2/3의 full은 Pilot 탈락으로 생략. SW4 공식 native 전체 예측은 계산 시간 절약을 위해 중단하고, RBF OVO 전체 19200과 native 640 subset의 정확 일치로 cross-check. SW1.5도 같은 검증된 직접 계산 경로 사용.
