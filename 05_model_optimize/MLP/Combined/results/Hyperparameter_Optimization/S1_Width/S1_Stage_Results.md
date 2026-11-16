# EXP-CMB-008 · 02 S1 은닉층 너비 실측

> **엄격 통과 여부:** NOT_PASSED / 완료된 신규 모델 36개, 과거 기준선 12개 재사용.

## 변경 근거
기본 은닉층 100의 표현 용량이 부족한지 검증하기 위해 64/128/256을 시험했다. 다른 하이퍼파라미터와 Size30/Stroke28/CoM, Seeds 및 Writer 분할은 동일하다.

| Width | Hand-made OOF | OfficialVal | 0→D 건수 | 8→B 건수 | D→0 | B→8 | Gate 위반 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 87.734% | 95.027% | 27 | 49 | 73 | 73 | official_validation_no_drop_over_0p5pp;writer_no_drop_over_1pp;B8D_error_no_increase_over_3pp;zero_0_to_D;zero_8_to_B;reverse_D_to_0_no_increase;reverse_B_to_8_no_increase |
| 100 | 90.716% | 96.117% | 19 | 43 | 71 | 54 | zero_0_to_D;zero_8_to_B |
| 128 | 90.964% | 96.562% | 26 | 38 | 64 | 61 | zero_0_to_D;zero_8_to_B;reverse_B_to_8_no_increase |
| 256 | 92.214% | 97.044% | 14 | 38 | 67 | 43 | zero_0_to_D;zero_8_to_B |

## 결론 및 다음 실험
0→D와 8→B 두 방향을 세 Seed 전체에서 모두 **0건**으로 만들고 다른 혼동의 악화를 방지하는 것이 최종 통과 조건이다.
**S1 최종 통과 Width:** 없음 (최종 승인 불가)
**S2를 위한 잠정 Width:** 256
잠정 선택은 최종 통과가 아니며, S2에서도 동일한 Zero-error Gate와 각 작성자/클래스 악화 기준을 다시 적용한다.
모든 Seed/Writer/16클래스 Family Flip과 실패 근거는 S1_details.json 및 S1_width_comparison.csv에 기록했다.
Original Test·교수님 Test·신규 손글씨 480장 미사용.
