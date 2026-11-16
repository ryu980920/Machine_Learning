# 가설 B EXP유009 — C9 정규화 + Data Augmentation 후속 검증 계획

**상태:** 사전 계획 / 실험 실행 직전. 모든 지표는 실측 후 별도 RESULTS에 기록.

## 출처 / 연구 질문
가설 A EXP유004 실험일지 6절에 사전 제시된 원본+C9 Data Augmentation 후속 계획을 그대로 구현한다.

C9는 이미 **Size(BBox 긴 변을 공식 Train 중앙값 20px로 정규화) → CoM 중심 재정렬**한 결과다. 따라서 C9 뒤에 Size→CoM을 한 번 더 반복하는 것은 Data Augmentation이라기보다 정규화를 중복 적용하는 실험이다. 이번 계획의 비교는 **C9만 학습했을 때와 Training의 원본/C9 두 종류 표현을 함께 제공했을 때**를 구분한다.

## 실험군 (동일한 76800장 원천 Training)

| ID | 학습 이미지 구성 | 학습 행 수 | max_iter | 총 미니배치 업데이트(명목상) | 목적 |
|---|---|---:|---:|---:|---|
| A `c9_only` | 공식 Train C9 처리본 | 76800 | 200 | 76800 | 기존 C9 대조 |
| B `raw_only` | 공식 Train 원본 | 76800 | 200 | 76800 | 원본 대조 |
| C `mix_50_50` | 원본 또는 C9 중 하나로 동일 표본 50:50 치환 | 76800 | 200 | 76800 | 데이터 수 증가 없이 형태 다양성 효과 |
| D `raw_plus_c9` | 모든 공식 Train 원본+C9 처리본 | 153600 | 100 | 76800 | 데이터 양과 형태 다양성 동시 |
| E `raw_twice` | 모든 공식 Train 원본을 2회 반복 | 153600 | 100 | 76800 | D 대비 반복 노출 대조 |

- 두 배 데이터군은 **epochs를 절반으로 조절**해 `batch_size='auto'=200`에서 전체 nominal SGD update 수를 맞춘다. 이는 학습량 교란을 줄이는 통제이며, 배치별 순서·최적화 궤적까지 동일하다는 뜻은 아니다.
- 조건 C는 `StratifiedShuffleSplit(test_size=.5,random_state=20261116)`을 사용, Seed42·43·44에서 동일한 선택을 고정한다.
- 모든 조건 Seed=42·43·44, hidden 100 ReLU, Adam, alpha=1e-4, lr=1e-3, tol=0, early_stopping=False, n_iter_no_change=201. **총 15개 MLP**.
- 공식 Train 76,800 / 공식 Dev Validation 19,200 / Hand-made 2,560(작성자41·42·43·44). 기존 SHA 고정과 원본/증강 동일 출처 그룹이 갈라지지 않은 split 사용.
- C9는 가설 A 원본 `Stage2Preprocessing('C9_BBoxLongside20')` 그대로 호출. 원본 Official Train **만** C9 기준값에 fit. Hand-made나 Validation으로 기준값 추출 금지.
- 모든 모델을 **두 가지 평가 입력**(원본/raw, C9-normalized)으로 각각 Official Validation, Hand-made에 평가하여, 학습의 증강 효과와 평가 시 정규화 효과를 별도 측정한다. 평가 모드는 사전에 고정하고 Hand-made 최고값을 사후 선택하지 않는다.
- 각 Seed별 전체·숫자·문자·클래스별 Recall·작성자별 Accuracy와 B↔8, D↔0, 5↔F 혼동, 같은 필체의 정오답 수정/신규오류 기록.
- 세 목표 **Hand-made digits≥85%, letters≥80%, Official Dev Val≥97.30%** 동시 평가.
- 공식 Original Test와 교수님 Test 미사용, Hand-made fit 미사용. 새 작성자 독립 Test처럼 해석하지 않음.

## 사전 해석 원칙
D `raw_plus_c9` vs E `raw_twice`는 153600행 조건끼리 크기 정규화의 추가 다양성 효과를 비교한다. D vs A는 C9 단독 대비 원본을 섞은 이득을 비교한다. C vs A는 행수 76800으로 고정해 원본/C9 혼합 효과를 본다. 각 조건의 공식 Val+Hand-made 숫자·문자를 동시에 비교한다.

## 안전·파일 관리
`exp/normal-b-exp009-c9-data-augmentation` 실험 브랜치에만 스크립트·계획·실측 결과를 기록한다. GitHub Actions 결과 Artifact(모델·모든 실측 CSV·예측)를 30일 보관한다. 추가 승인 전에 `main` 병합, 기존 모델 교체, 최종 Test 사용은 하지 않는다.
