# EXP-FT-SVC-003 — 취약 클래스 조정

## 배경/목적/가설
B,D,C의 Hand-made weight만 1.5/2.0배; 별도 기존 D/B 집중 증강을 절반으로 줄이거나 B/D/C +240 (full intended, pilot +24). 학습용 원본에서만 증강. TA 증감 모두 Pilot 개선 실패. TW2를 full 후보로 승격.
## 고정 조건
CoM→Deskew→Pixel:HOG20:80, RBF, seed42. 축소 Pilot 기준 8,528장: 공식 8,000장 및 Hand-made 원본/증강 512장, D/B 집중 증강 16장. 공식 개발 640장 및 Hand-made 개발 1,280장 동일 평가. 증강량 변경 시 학습 행 수만 변경. Pilot의 절대 수치는 full baseline 93.6719%와 직접 비교하지 않는다.
## 사전 통과 기준
동일 Pilot baseline 대비 정답 증가, Precision 유지, F1 증가, 공식 Pilot 정답 하락 1장 이내. Full 적용 시 공식 19,200장 최대 19장 하락, B/D/C pooled recall 개선, 0/8/6 각 2장 이하 하락, 기존 full 정확도/Precision/F1 유지·개선. 속도는 별도 판정.
## 실측 조건별 결과
| condition | train_rows | correct | Accuracy(%) | Precision(%) | F1(%) | official_pilot_correct | delta_correct_vs_pilot | fit_s | pilot_guardrail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_G1 | 8528 | 1153 | 90.0781 | 90.3571 | 90.0642 | 622 | 0 | 2.3599 | False |
| TW1.5 | 8528 | 1158 | 90.4688 | 90.6976 | 90.4444 | 621 | 5 | 2.3776 | True |
| TW2 | 8528 | 1161 | 90.7031 | 90.9186 | 90.6820 | 621 | 8 | 2.3721 | True |
| TA_reduced | 8520 | 1153 | 90.0781 | 90.3633 | 90.0638 | 622 | 0 | 2.6927 | False |
| TA_increased | 8552 | 1153 | 90.0781 | 90.3754 | 90.0597 | 621 | 0 | 2.3480 | False |
## 분석
Baseline C1_G1을 재사용. `pilot_all_conditions.csv`에 전체 원본 수치와 두 비교 기준별 교정/퇴행 기록 보존. True는 Pilot 선별 기준 통과일 뿐 최종 채택 아님. Full 학습 미완료 조건은 최종 결과로 사용하지 않는다.

## 전체 학습 후속 실측
| condition | training_rows | correct | Accuracy(%) | Precision(%) | F1(%) | official_correct | corrected | regressed | accuracy_guardrails_pass | fit_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TW2 | 82080 | 1205 | 94.1406 | 94.3337 | 94.1229 | 18962 | 6 | 0 | True | 225.9014 |

TW2는 1205장, 교정 6/신규오답 0, 공식 18962, target/protected 기준 통과. TW1.5는 pilot-only. 증강량 감소/증가 조건은 Pilot에서 정답 개선 없어 full 승격 안 함. 공식 19200 RBF 직접식, native 640 및 전체 1280 개발 예측 일치 확인.
