# Combined MLP 최종 모델 구성 동결 — 2026-11-19

**최종 채택: 신규 손글씨 480장을 추가하지 않은 기존 Combined `CMB011/base`.**

선정·구성 고정 근거:
- [최종 선정 실험일지](../../logs/EXP-CMB-012_Final_Combined_Model_Freeze_20261119.md)
- [새로 만든 438장에 대한 84모델 실측·최종 결정](../../logs/학습데이터_추가/EXP-CMB-011_03_External438_and_Selection.md)
- [고정 설정·12개 CV 체크포인트 SHA-256](FINAL_COMBINED_FREEZE.json)
- [438장 7조건 정량 비교 CSV](External438_7Arm_Summary.csv)
- [CMB011 84개 학습/Writer OOF 원본 결과](../Targeted_Handmade_480/README.md) · [CMB011 보고서](../../logs/학습데이터_추가/EXP-CMB-011_02_Classwise_MatchedControls_Results.md)
- [CMB011 GitHub Actions 84개 모델](https://github.com/ryu980920/Machine_Learning/actions/runs/44345990347)

## 선택한 구성

`MLPClassifier(hidden_layer_sizes=(256,), alpha=0.0001, learning_rate_init=0.001, batch_size=128, early_stopping=False, max_iter=200, activation="relu", solver="adam")`.

학습: 공식 Training 76,800장(30% Size / Stroke28% 증강 후 CoM) + 기존 손글씨 3명 1,920장(Writer 보류 Fold별). 총 78,720장. 신규 `CMB010_7_8_9_B_480.npz`는 실험 자료로 보존하되 **최종 학습에서 제외**한다.

추론: 28×28 Grayscale → CoM 중심 정렬 → PixelScaler255 → MLP. Size/Stroke 증강은 **추론 시 적용하지 않는다**.

## 제한

- 설정·데이터 레시피·CV 프로토콜의 **최종 선정 및 고정**이며 **새 4명 전체를 모두 학습한 단일 최종 추론 가중치를 생성하거나 GitHub에 배포한 것은 아님**.
- CV 기준선 12개 모델의 가중치는 Actions 산출물이며 원래 30일 보존; 이 JSON의 SHA와 모델별 Manifest를 참고한다.
- 438장은 반복 활용한 개발용 자료이며 Original Test·교수님 Test가 아니다.
- 기존 Batch128의 CMB008 S6 사전 가드레일 미통과 판정은 그대로 보존한다.
- 이후 모델 재선정·새 추론용 체크포인트 생성·평가 Test 실행에는 별도 의사결정이 필요하다.
