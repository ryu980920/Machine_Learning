# SVC Combined — 담당자용 연구 안내

## 현재 선정 모델
C=2, gamma=2×scale, 82,080장. Accuracy 94.453125% (1,209/1,280), Macro Precision 94.604998%, Macro F1 94.413667%, 공식 Validation 19,003/19,200. 95%까지 7장.
CoM → Deskew → Pixel:HOG=20:80. 1,280장은 개발용이며 독립 Test가 아니다.

**처음 시작할 때:** [공통 입력·환경](../00_Common/README.md) → [현재 모델 검증](07_Final_Model/README.md) → [후속 연구 시작](08_Future_Experiments/README.md).

## 1차 연구 흐름
| 단계 | 목적 | 기존 결과 / 판정 | 자료 |
|---|---|---|---|
| 01_Baseline | 1차 이전 기준 | 1199/1280 = 93.671875% | [열기](01_Baseline/README.md) |
| 02_Error_Analysis | 오류 원인 | B 20, D 8, C 7; 정답 OVR 2위 52/81 | [열기](02_Error_Analysis/README.md) |
| 03_C_Gamma | 결정 경계 | 9 Pilot, C2_G2 Full 1209/1280; 채택 | [열기](03_C_Gamma/README.md) |
| 04_Sample_Weight | Hand-made 가중치 | SW1.5 1203; SW4 1213이나 공식 -22장 → 실패 | [열기](04_Sample_Weight/README.md) |
| 05_Targeted_Training | 취약 클래스 가중치/증강 | TW2 Full 1205; 증강량 변경 Pilot 미개선 | [열기](05_Targeted_Training/README.md) |
| 06_Combination | 우수 변수 조합 | 4 Pilot; Full 미실시 | [열기](06_Combination/README.md) |
| 07_Final_Model | 현재 선정 | C2_G2 94.453125%; 속도 권장 기준 초과, 승인 선정 | [열기](07_Final_Model/README.md) |
| 08_Future_Experiments | 후속 연구 시작 | 계획·템플릿, 결과 없음 | [열기](08_Future_Experiments/README.md) |

[역사적 종합 보고서](Historical_Report.md) (과거 보류 문구는 당시 판단; 현재 선정은 위 기준). 기존 보고서 원본: [고정 커밋](https://github.com/ryu980920/Machine_Learning/blob/765b71c2b62e34e95d40d68521bef657635ce2e8/07_fine_tuning/06_reports/SVC_Fine_Tuning_Report.md).

각 단계 README에서 실제 결과와 코드, Pilot/Full 범위를 구분한다. 각 Notebook은 새 커널에서 실행하고 승인된 로컬 데이터 경로를 지정한다. 공개 자료만으로 전체 실험 재현 불가. 신규 결과는 해당 Future_Experiments/Results 아래 집계 자료만 등록; 현재 모델 변경은 별도 승인.
