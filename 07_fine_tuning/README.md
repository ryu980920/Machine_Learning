# SVC · KNN Fine-Tuning — 팀원 인수인계

## 담당 모델부터 선택

| 담당 | 현재 최종 모델 | 개발 Accuracy | 시작 링크 |
|---|---|---:|---|
| SVC Combined | C2_G2 (1차 Fine-Tuning 승인) | **94.453125% (1209/1280)** | [SVC 담당자 안내](01_SVC/README.md) · [후속 실험](01_SVC/08_Future_Experiments/README.md) |
| KNN Combined | 기존 KNN 유지 | **92.34375% (1182/1280)** | [KNN 담당자 안내](02_KNN/README.md) · [후속 실험](02_KNN/09_Future_Experiments/README.md) |

## 정리된 연구 폴더

- [00_Common](00_Common/README.md) — 입력 데이터, 환경, 공통 준비 코드, 기존/후속 평가 기준
- [01_SVC](01_SVC/README.md) — 연구 흐름, 실험별 실행 Notebook 및 기존 결과, 현재 모델, 후속 실험
- [02_KNN](02_KNN/README.md) — 동일한 구성
- [03_Final_Comparison](03_Final_Comparison/README.md) — 기존 실측 결과 비교
- [99_Archive](99_Archive/README.md) — 원본 84개 스냅샷, 이전 링크 호환

**실행 순서:** Common → 담당 모델 최종 검증 → 연구 이력 → 후속 계획 → 실험 Notebook. 1,280장은 개발용이며 독립 Test가 아니다. Original Test 및 교수님 Test 금지. 신규 실험 결과 및 모델 변경 없음; 2차 Fine-Tuning 제외.

공개 저장소에는 비공개 학습·평가 원본과 바이너리가 없다. 팀 리더에게 요청하고 SHA-256 확인. [현재 최종 선정 Manifest](05_final_models/CURRENT_FINAL_MODEL_SELECTION.json). SVC 속도 권장 기준 미충족 상태에서 정확도 우선 승인. KNN PCA128은 미채택 속도 대안. [원본-신규 경로 매핑](EXPERIMENT_FILE_MAP.csv).

### 기존 경로 호환
`00_baseline/`, `01_error_analysis/`, `02_SVC/`, `03_KNN/`, `04_final_comparison/`, `05_final_models/`, `06_reports/` 및 기존 root scripts/Notebook은 기존 링크와 통합 Notebook 호환을 위해 유지. **신규 연구자는 위 모델별 안내만 사용.** 이전 README는 [원본 아카이브](99_Archive/Original_20261128/README.md). 분리 Notebook은 원본 코드에서 재구성했으며 전체 학습 재실행 미검증. 
