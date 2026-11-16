# MLP / Combined — 작성자 보류 OOF 최적화 연구

> **현재 상태:** **EXP-CMB-001~007**의 실험과 결과를 기록했다. 실험 번호는 하나의 `EXP-CMB-###` 체계이며, 계획/중간/최종/후속 분석은 **동일 실험의 문서 단계**다. **최종 모델 선택 및 배포 체크포인트 변경은 보류.**

## 연구 흐름 (순서대로 읽기)

```text
CMB-001   Combined Baseline (4명 Writer Leave-One-Out)
   ↓
CMB-002   오분류 원인 분석 (0↔D, 8↔B)
   ↓
CMB-003   기존 모델 고정 입력 민감도 진단
   ↓  가설 B: 크기·중심·획 분포의 영향 검증
CMB-004   Geometry Normalization (Size / CoM / Size+CoM)
   ↓
CMB-005   Size 0~30% × Stroke (10조건)
   ↓
CMB-006   Size 40~50% 추가 (누적 16조건)
   ↓
CMB-007   Size 60~100% 추가 (누적 31조건)
           → 정확도·클래스별 안정성·오분류 방향 분석
```

[**전체 실험 일지 및 단계별 문서 링크**](logs/README.md) · [가설 B 통합 설명](logs/가설B/README.md)

## 연구 단계별 핵심 결과

| 실험 | 수행 내용 | Hand-made OOF 및 판단 | 상세 |
|---|---|---|---|
| **CMB-001** | Baseline, 4-Writer 보류 OOF | Normal 62.578% → Combined **81.250%** | [로그](logs/EXP-CMB-001_Baseline_4Writer_OOF.md) |
| **CMB-002** | 오류·형태/공간 단서 분석 | 가설 B 우선, 가설 A 조건부 보류 | [로그](logs/초기분석/EXP-CMB-002_Error_Analysis.md) |
| **CMB-003** | 고정 MLP 입력 민감도 비교 | 95% 축소의 작성자 편중과 얇게25% 후보 확인 | [로그](logs/초기분석/EXP-CMB-003_Input_Sensitivity.md) |
| **CMB-004** | Size / CoM 정규화 기전 비교 | Size+CoM **88.945%**, 공식 Validation 손실 | [최종 보고](logs/가설B/EXP-CMB-004_02_Final_Geometry_Normalization.md) |
| **CMB-005** | Size 0~30% × Stroke, 120개 MLP | 전체 10조건 중 30%-28% **90.716%** | [최종 보고](logs/가설B/EXP-CMB-005_03_Final_Size00_30.md) |
| **CMB-006** | Size 40~50%, 72개 MLP 추가 | 누적 16조건 비교, 30%-28% 우위 유지 | [계획](logs/가설B/EXP-CMB-006_01_Plan_Size40_50.md) · [16조건 결과](results/Balanced_Scale_Stroke_Extension/full_0to50_comparison/SUMMARY.md) |
| **CMB-007** | Size 60~100%, 180개 MLP 추가 | 누적 **31조건** 비교, 최고 OOF 갱신 실패, 8↔B 비대칭 오류 관측 | [최종 보고](logs/가설B/EXP-CMB-007_03_Final_Size60_100.md) · [오분류율 후속 분석](logs/가설B/EXP-CMB-007_04_Review_Top3_OOF_Error.md) |

## CMB-007까지의 관측과 한계

- 전체 31조건에서 OOF 정확도 상위 세 조합은 **Size30%-Stroke28%, 30%-14%, 30%-0%**이며 오분류율은 **9.284%, 9.310%, 9.427%**.
- 고비율 증강은 평균 Family Flip을 낮추지만 **8→B 오류 증가 / B→8 오류 감소**, OfficialVal 성능 하락, 특정 클래스 최악 반례를 동반했다.
- 사후 순위가 독립 데이터에서의 우월성 또는 최종 모델 선정 근거인 것은 아니다. **Original Test·교수님 Test·Normal 신규 480장은 CMB-007 모델 선택에 사용하지 않았다.**

## 자료 찾기

[실험일지](logs/README.md) · [실측 CSV/그림](results/README.md) · [재현 코드](scripts/README.md) · [가설 A 보류 근거](logs/가설A/README.md) · [실험 공통 양식](../../EXPERIMENT_TEMPLATE.md).

**번호 구분:** Normal 실험의 `EXP유###` 번호는 별개이며 Combined의 `EXP-CMB-###`와 혼용하지 않는다.
