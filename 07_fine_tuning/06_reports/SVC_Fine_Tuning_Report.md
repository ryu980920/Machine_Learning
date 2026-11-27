# SVC Combined Fine-Tuning 종합 보고서

## 1. 기준 및 목표
기존 SVC RBF C=1, gamma=scale, CoM→Deskew→Pixel:HOG=20:80, 82,080 학습 행. 기존 개발 93.671875% (1199/1280), 공식 개발 18967/19200. 실제 오류 81장 중 B 20장, D 8장, C 7장. 정답이 OVR score 2위인 사례 52장. 따라서 결정 경계, Hand-made 가중치, B/D/C 대상 가중치 및 증강을 비교.

## 2. 실험 설계
승인한 full DOE 범위를 계산 제약에 맞춰 8,528장 Pilot 선별 → 전체 82,080장 후보 승격으로 조정했다. 모든 Pilot 조건은 동일 1,280장 개발 및 공식 개발 640장 평가. 공식 원본 학습/Hand-made 학습만 fit에 사용. 사전 기준은 `EXPERIMENT_PROTOCOL.json`. 이 변경으로 모든 DOE가 full 조건에서 실행된 것은 아니다.

- [C·gamma](../02_SVC/EXP-FT-SVC-001.md): 9개 Pilot 및 C2_G2 full.
- [Hand-made Weight](../02_SVC/EXP-FT-SVC-002.md): 5개 Pilot(기존 대조군 재사용), SW4 full 및 SW1.5 follow-up full.
- [취약 클래스](../02_SVC/EXP-FT-SVC-003.md): 5개 Pilot(기존 대조군 재사용), TW2 full. 증강 증감은 Pilot 미개선.
- [조합](../02_SVC/EXP-FT-SVC-004.md): 4개 Pilot 조합, 단독 최고보다 개선된 조합 없음으로 full 미승격.

## 3. Full 실측 비교
| condition | training_rows | correct | Accuracy(%) | Precision(%) | F1(%) | official_correct | corrected | regressed | accuracy_guardrails_pass | fit_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C2_G2 | 82080 | 1209 | 94.4531 | 94.6050 | 94.4137 | 19003 | 15 | 5 | True | 268.6191 |
| SW1.5 | 82080 | 1203 | 93.9844 | 94.2186 | 93.9653 | 18962 | 5 | 1 | True | 230.2562 |
| SW4 | 82080 | 1213 | 94.7656 | 94.9403 | 94.7613 | 18945 | 17 | 3 | False | 194.5528 |
| TW2 | 82080 | 1205 | 94.1406 | 94.3337 | 94.1229 | 18962 | 6 | 0 | True | 225.9014 |

SW4의 공식 정확도 하락이 22장으로 최대 허용 19장을 초과한다. 전체 1280장 정확도가 높아도 선택하지 않는다. C2_G2는 정확도 보호 기준을 통과한 후보 중 최고로, 기존보다 순정답 10장, 공식 개발 정답 +36장, B/D/C 합계 +6장. TW2는 교정 6, 신규 오답 0으로 보수적인 대안. SW1.5도 보호 기준 통과.

## 4. 시간 비교
C2_G2와 기존 SVC의 동일 1280장×5회 Batch1 재측정은 각각 12.712982 및 10.994163 ms/image. 상대 증가 15.634%. 별도 CPU affinity에 동시 학습이 존재해 공유 메모리 등 영향 가능. 권장 10%를 초과. 다른 full 후보는 1280×5 시간 미측정. Fit 시간은 별도로 실측, 학습 준비 공통 12.523029초는 이전 준비 실행 측정값이며 순수 fit에 합쳐 이전 전체 학습 시간과 공정 비교하지 않음.

## 5. 공식 보호 기준 검증
C2_G2는 원래 sklearn predict 19200 전량 및 RBF OVO 직접 계산 일치. SW4는 native 19200 반복 계산을 중단하고 직접 계산 19200 + native 640 확인, TW2/SW1.5도 직접 계산 19200 + native 640 및 dev 1280 전량 일치. 공식 method 및 native completion 여부는 개별 `review_manifest.json`에 기록. 모든 모델 체크포인트 SHA256 재검증.

## 6. 결론
정확도 우선 검토 후보는 C2_G2 (94.453125%). 95% 목표에 7장 미달. 추론 시간 권장치 미충족 및 독립 신규 Test 미실시로 기존 모델 교체/최종 배포는 사용자 검토 전 보류. SW4 공식 보호 기준 실패, TW2/SW1.5의 공식 직접 계산 및 시간 검증 범위 차이, Pilot-only 조건을 숨기지 않는다. 통계검정은 이미 동일 개발 데이터로 선택한 후보의 기술적 paired analysis이며 독립 검정으로 주장하지 않는다.
