# 머신러닝 4조 — 담당 모델 확정 (2026-11-22)

## 팀원별 역할

| 팀원 | 담당 모델 | 모델별 실험 기록 |
| --- | --- | --- |
| **유용성** | **MLP (Multi-Layer Perceptron)** | [Normal](../05_model_optimize/MLP/Normal/README.md) · [Combined](../05_model_optimize/MLP/Combined/README.md) |
| **곽지호** | **KNN + SVC** | [KNN](../05_model_optimize/KNN/Normal/README.md) · [SVC](../05_model_optimize/SVC/Normal/README.md) |
| **김다연** | **HistGD (HistGradientBoosting)** | [HGB](../05_model_optimize/HGB/Normal/README.md) |
| **이선형** | **Extra Trees** | [Extra Trees](../05_model_optimize/Extra_Trees/Normal/README.md) |

- **출처:** 2026-11-22 사용자 직접 지정한 역할 분담. 표의 모델별 폴더 링크는 실험 저장 위치로, **담당 배정과 최적화 완료 여부는 별도**이다.
- 각 모델에서 **Normal**(공식 Training만 학습)과 **Combined**(사전에 지정한 Hand-made Training 추가)를 구분한다. 실제 실험 유무와 성능은 해당 모델의 로그·결과로만 판단한다.
- 2026-11-22 현재 첨부받은 **MLP Normal / Combined 최적화 보고서**의 주요 결과는 [Normal 연구요약](../04_reports/MLP_Normal_Optimization_Review_20261122.md) 및 [Combined 연구요약](../04_reports/MLP_Combined_Optimization_Review_20261122.md)에 정리한다.
- **미확인:** 나머지 세 사람의 개별 최적화 완료 상태·최종 모델·정량 성능·향후 개별 마감 날짜. 이를 임의로 '완료' 처리하지 않는다.
- **팀 공통 일정(기존 저장소 기준):** 2026-12-06 최종 결과물 제출, 2026-12-07 최종 발표. 변경 여부는 별도 확인한다.

## 공유·통합 시 확인사항

1. 각 담당자는 본인 모델의 Normal·Combined 실험 경로와 최종 채택 설정·근거를 자신의 README/로그에 연결한다.
2. Accuracy·Macro F1·Confusion Matrix·학습/추론 시간은 사용 데이터와 평가 구분을 함께 기록한다.
3. Original Test 및 교수님 Test는 기존 [Test 정책](../02_data/HANDMADE_SPLIT_PLAN.md)에 따라 관리하며, 개발용 평가값을 독립 Test로 표시하지 않는다.
4. 팀 최종보고서에는 모델별 선정 사유와 실패 조건, 데이터 누출 방지, 개인별 실제 수행 내역을 대응시킨다.
