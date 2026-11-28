# KNN Combined — 담당자용 연구 안내

## 현재 선정 모델
K=3, Euclidean, distance, 113,920장. Accuracy 92.34375% (1,182/1,280), Macro Precision 92.61688%, Macro F1 92.32689%, 공식 Validation 18,856/19,200. 95%까지 34장.
CoM → Deskew → Pixel:HOG=20:80. 1,280장은 개발용이며 독립 Test가 아니다.

**처음 시작할 때:** [공통 입력·환경](../00_Common/README.md) → [현재 모델 검증](08_Final_Model/README.md) → [후속 연구 시작](09_Future_Experiments/README.md).

## 1차 연구 흐름
| 단계 | 목적 | 기존 결과 / 판정 | 자료 |
|---|---|---|---|
| 01_Baseline | 1차 이전/현재 기준 | 1182/1280 = 92.34375% | [열기](01_Baseline/README.md) |
| 02_Error_Analysis | 오류 원인 | B 18, 5 11, D 10, C 9; 정답 Top 9에 60/98 | [열기](02_Error_Analysis/README.md) |
| 03_Distance_Metric | Minkowski p | Pilot 4조건, Euclidean 유지 | [열기](03_Distance_Metric/README.md) |
| 04_K_and_Voting | K 및 투표 가중치 | 13조건, k3 유지 | [열기](04_K_and_Voting/README.md) |
| 05_Sample_Replication | 복제 및 계보 | 5조건, 기존 혼합 복제 유지 | [열기](05_Sample_Replication/README.md) |
| 06_PCA | 차원 축소 | 128·512 정확도 동일, 속도 대안 | [열기](06_PCA/README.md) |
| 07_Combination | PCA×투표 | 26조건, 정확도 개선 없음 | [열기](07_Combination/README.md) |
| 08_Final_Model | 현재 선정 | 기존 KNN 92.34375% 유지 | [열기](08_Final_Model/README.md) |
| 09_Future_Experiments | 후속 연구 시작 | 계획·템플릿, 결과 없음 | [열기](09_Future_Experiments/README.md) |

[역사적 종합 보고서](Historical_Report.md) (과거 보류 문구는 당시 판단; 현재 선정은 위 기준). 기존 보고서 원본: [고정 커밋](https://github.com/ryu980920/Machine_Learning/blob/765b71c2b62e34e95d40d68521bef657635ce2e8/07_fine_tuning/06_reports/KNN_Fine_Tuning_Report.md).

각 단계 README에서 실제 결과와 코드, Pilot/Full 범위를 구분한다. 각 Notebook은 새 커널에서 실행하고 승인된 로컬 데이터 경로를 지정한다. 공개 자료만으로 전체 실험 재현 불가. 신규 결과는 해당 Future_Experiments/Results 아래 집계 자료만 등록; 현재 모델 변경은 별도 승인.
