# EXP유010~019 — 크기 Shortcut 분석 결과 CSV

이 폴더에는 **실측 CSV·요약 파일·3-Seed 집계 결과**를 보관합니다. [실험일지](../../logs/크기편향_원인분석/README.md)와 [원본 실행 코드](../../scripts/README.md)는 별도 위치에 있으며 여기서 링크로 연결합니다.

| 실험 | CSV 결과 | 실험일지 |
|---|---|---|
| EXP유010 | [크기분포](EXP유010_summary.csv) | [원인 분석](../../logs/크기편향_원인분석/EXP유010_크기분포_분석.md) |
| EXP유011 | [Size/CoM 교차](EXP유011_summary.csv) | [교차 분석](../../logs/크기편향_원인분석/EXP유011_Size_CoM_교차분석.md) |
| EXP유012 | [크기 개입](EXP유012_summary.csv) | [크기 분류 단서](../../logs/크기편향_원인분석/EXP유012_크기분류단서.md) |
| EXP유013 | [기하특징 4개](EXP유013_binary_4feature_summary.csv) | [기하 특징 이진 분류](../../logs/크기편향_원인분석/EXP유013_기하특징_이진분류.md) |
| EXP유014 | [정확도](EXP유014_accuracy_3seed_summary.csv), [크기 민감도](EXP유014_scale_flip_3seed_summary.csv) | [크기·획 증강 학습](../../logs/크기편향_원인분석/EXP유014_Shortcut_완화_학습.md) |
| EXP유015 | [26조건 전체 판정](EXP유015_26조건_전체판정.csv), [B·D·8 개입](EXP유015_26조건_B8D_전환율_전체.csv), [9개 Gate 합격](EXP유015_shortcut_gate_통과9조건_요약.csv), [OfficialVal 요약](EXP유015_OfficialVal_크기변형_요약.csv) | [78개 MLP 최적화](../../logs/크기편향_원인분석/EXP유015_증강비율_획강도_최적화.md) |
| EXP유016 | [23조건 결과](원본_실측_아카이브/EXP유016_60모델/all_23_conditions_ranked_gates.csv), [16개 클래스 배율별](원본_실측_아카이브/EXP유016_60모델/all_16class_size_flip_3seed.csv), [OfficialVal](원본_실측_아카이브/EXP유016_OfficialVal/officialval_scale_3seed_summary.csv) | [60~100% 실험일지](../../logs/크기편향_원인분석/EXP유016_고비율_크기증강_한계확인.md), [80-14/70-0 보존 결정](../../logs/크기편향_원인분석/EXP유016_80-14_70-0_두_후보_병행보존.md) |


| EXP유017 | [OfficialVal 5배율·3 Seed](EXP유017_OfficialVal_5배율_3Seed_요약.csv), [동결 모델 SHA-256](EXP유017_동결모델_SHA256_Manifest.json) | [탐색 설계·선정 근거 통합 로그](../../logs/크기편향_원인분석/EXP유017_80-14_MLP_하이퍼파라미터_최적화.md) |
| EXP유018 | [455/480장·3-Seed 그룹별](EXP유018_3Seed_그룹별성능.csv), [16클래스 Precision·Recall·F1](EXP유018_16클래스_3Seed_지표.csv), [455장 평균 Confusion Matrix](EXP유018_455장_ConfusionMatrix_3Seed평균.svg), [평균 CM CSV](EXP유018_455장_CM_3Seed평균.csv), [Seed별 CM CSV](EXP유018_455장_CM_Seed42.csv), [3개 혼동쌍 이전/신규 비교](EXP유018_이전CM_신규CM_주요쌍비교.csv), [원본 그룹별](EXP유018_주요쌍_원본그룹별.csv), [판독 제외 25장 인덱스/사유](EXP유018_제외25장_목록.csv) | [신규 손글씨 분석 실험일지](../../logs/크기편향_원인분석/EXP유018_신규손글씨480_판독필터_혼동쌍_진단.md) |

## EXP유018 후속 판독 438장 및 7→9 원인 분석

- [원본 480장·기존 455장·사용자 판독 438장 Seed별 정확도](EXP유018_480_455_438_Seed별성능.csv)
- [438장 16×16 Confusion Matrix — 3 Seed 평균](EXP유018_438장_CM_3Seed평균.csv)
- [455장과 438장의 주요 오분류쌍 비교](EXP유018_455_438_주요오류비교.csv)
- [추가 제외 17장 원본 인덱스](EXP유018_추가제외17장_사용자판독.csv)
- [원본 그룹2 실제 7의 동결 모델 9장×3 Seed 예측과 확률](EXP유018_438장_그룹2_7_개별예측.csv)
- [기존 80-14·최종 MLP 및 입력 표현별 그룹2 7→9](EXP유018_438장_7to9_모델입력별비교.csv)
- [실험 로그 §8·§9](../../logs/크기편향_원인분석/EXP유018_신규손글씨480_판독필터_혼동쌍_진단.md)

원본 NPZ·손글씨 이미지·학습된 모델 가중치는 본 저장소에 추가하지 않는다. 사용자 육안 판독을 거친 사후 필터 결과이며, 모델을 변경하거나 새로운 독립 Test를 확보한 것은 아니다.

### EXP유018 — 후속 7→9 폭·고리 개입과 원본 학습필체 조사

- [동결 MLP 가로 폭 개입 / 픽셀 강도 합계 보정 대조](EXP유018_7to9_가로폭_개입대조.csv)
- [상단 고리 연결 vs 같은 밝기 증가 대조, 이미지 인덱스별](EXP유018_7to9_상단고리_개입대조.csv)
- [MNIST 원본 숫자 7·9의 폭·강도·HOG 조건별 정확도](EXP유018_7to9_MNIST원본_학습필체조사.csv)
- [MNIST 원본 숫자 7의 그룹2 HOG 유사도 상위 18건](EXP유018_7to9_MNIST원본_유사7_상위18.csv)
- [전체 연구 해석·버전 제약: EXP유018 실험일지 §10~§11](../../logs/크기편향_원인분석/EXP유018_신규손글씨480_판독필터_혼동쌍_진단.md)

CSV는 사후 진단 집계이며 새 모델 또는 독립 테스트 지표가 아니다. MNIST 원본 조사에 이용된 보관본과 최종 v3의 NPZ SHA가 다르므로 정확한 Official Train 분할 편입 개수는 미확인.

### EXP유018 — 픽셀 공간 비교 및 Shortcut 후속 연구 질문 (2026-11-17)

- [그룹2 실제 7의 MNIST 최근접 7/9 코사인 유사도 9장](EXP유018_그룹2_7_픽셀최근접7_9.csv)
- [CoM 후 위·아래 1px 이동에 대한 9장×3 Seed 예측과 P7−P9 마진](EXP유018_그룹2_7_수직1px_Seed별예측.csv)
- [관측 결과와 Shortcut 검증 조건 — EXP유018 실험일지 §12~§13](../../logs/크기편향_원인분석/EXP유018_신규손글씨480_판독필터_혼동쌍_진단.md)

가로 폭·곡선·위치 의존성의 사후 탐색 결과는 **7→9 Shortcut을 식별한 확정적 인과 검증이 아님**. 원본 MNIST 보관 데이터와 GitHub 최종 v3의 파일 버전 차이도 별도 기록함.

### EXP유018 — 7→9 Shortcut 원인 단계별 검증 (2026-11-17)

- [순차 검증 원본 ZIP — 분석 Python 코드·모든 CSV·시각화 포함](EXP유018_7to9_Shortcut_Sequential_Study/EXP018_7to9_Shortcut_Sequential_Study.zip)
- [원본 획 기하 분포](EXP유018_7to9_Shortcut_Sequential_Study/phase1_7vs9_geometry.csv), [동결 MLP 비교](EXP유018_7to9_Shortcut_Sequential_Study/phase1_model7vs9_baselines.csv)
- [기여도(Integrated Gradients)](EXP유018_7to9_Shortcut_Sequential_Study/phase2_attributions_per_image.csv)
- [상·하단 donor 교체 결과](EXP유018_7to9_Shortcut_Sequential_Study/phase3_region_swap_summary.csv)
- [국소 ±1px 위치 개입 결과](EXP유018_7to9_Shortcut_Sequential_Study/phase4_small_shift_summary.csv)
- [전역 shear 비교](EXP유018_7to9_Shortcut_Sequential_Study/phase5_relative_shear_accuracy.csv)
- [강도 총량 보정 대조](EXP유018_7to9_Shortcut_Sequential_Study/phase6_massmatched_summary.csv)
- [고리 부재 규칙의 부작용](EXP유018_7to9_Shortcut_Sequential_Study/phase7_open_loop_heuristic_sanity.csv)
- [이전 80-14 vs 최종 모델 비교](EXP유018_7to9_Shortcut_Sequential_Study/phase8_old_vs_new_model_per_image.csv)
- [전체 판정 및 미실행 개선안](../../logs/크기편향_원인분석/EXP유018_7to9_Shortcut_Sequential_Causal_Study.md)

CSV는 **동결 모델의 사후 원인 진단**이고 ZIP 내부의 개별 CSV와 원본 분석 코드가 전체 근거임. 신규 438장 반복 사용에 따른 독립 Test 제한, 의미 보존/Shortcut 미확정, 향후 학습 개선안 **미실행**을 준수한다.

## 원본 실측 아카이브

- [EXP유015 78모델 3-Seed 전체 기록](원본_실측_아카이브/EXP유015_78모델/SUMMARY.md)
- [EXP유015 OfficialVal 0.8~1.2배 검증](원본_실측_아카이브/EXP유015_OfficialVal/SUMMARY.md)
- [EXP유016 60개 신규 모델 3-Seed 집계](원본_실측_아카이브/EXP유016_60모델/SUMMARY.md)
- [EXP유016 OfficialVal 독립 데이터 출처 확인](원본_실측_아카이브/EXP유016_OfficialVal/SUMMARY.md)

실측 값·원본 Git blob SHA는 수정하지 않고 이동합니다. **80-14와 70-0 모델 가중치(joblib)는 EXP유016 실험 브랜치에 별도 보관하며 최종 모델에 적용하지 않았습니다.**

### EXP유019 — 국소 획 ±1px 증강 학습 B1·B2 실패 분석 (2026-11-17)

- [3 Seed 최종 판정표](EXP유019_국소획위치증강/EXP019_Decision_Matrix.csv)
- [전체·숫자·문자·7·9 정확도](EXP유019_국소획위치증강/all_3seeds_dataset_group_metrics.csv)
- [7→9·9→7 오류](EXP유019_국소획위치증강/all_3seeds_7to9_9to7_pairs.csv), [Seed별 짝지은 정확도 변동](EXP유019_국소획위치증강/all_3seeds_paired_transitions.csv)
- [16클래스 정확도](EXP유019_국소획위치증강/all_3seeds_classwise_metrics.csv), [OfficialVal 배율별 크기 Shortcut Gate](EXP유019_국소획위치증강/officialval_08_12_scale_family_flip_gate.csv)
- [B1 증강 선택·거부 통계](EXP유019_국소획위치증강/B1_top_augment_manifest.json), [B2 증강 통계](EXP유019_국소획위치증강/B2_bottom_augment_manifest.json)
- [6개 새 MLP 체크포인트 SHA256/크기](EXP유019_국소획위치증강/EXP019_model_sha256_manifest.json), [공식 Train 7·9 증강 예시](EXP유019_국소획위치증강/OfficialTrain_7_9_LocalStroke_Augmentation_Examples.png)
- [동결 재현·학습·통과 판정 상세 보고서](../../logs/크기편향_원인분석/EXP유019_국소획위치증강_실측보고서.md) 및 [학습 코드](../../scripts/exp019_train_local_warp.py)

**결과:** B1(상단 ±1px)과 B2(하단 ±1px) 모두 7→9 오류 개선 기준 미달. B2의 전체 정확도 향상은 유효한 관찰 결과이지만 7→9의 성공으로 간주하지 않음. **모델 바이너리 6개는 이 저장소에 포함되지 않고 SHA만 보존**; 기존 EXP유017 동결 체크포인트·원본 v3·Hand-made 데이터를 덮어쓰지 않음. 다음 단계의 7/9 형상 분포 구간 분석 및 타깃 증강은 **아직 미실행**.
