# Combined 가설 B — EXP유002 / EXP-CMB-005: Normal 크기 편향 완화 전처리의 전이 검증

> 상태: 사전등록. **통과 기준은 특정 Accuracy/F1 숫자가 아니라 두 방법의 작용·부작용을 실측으로 설명할 수 있는가**이다. 실행 후 이 조건을 바꾸지 않는다.

## 근거
Normal EXP유013~017은 Official Training에서 숫자·문자의 BBox 크기 단서가 크고, 클래스별 균형 크기 증강(16/18/20/22/24px)+약한 Stroke+CoM으로 크기 개입 Family Flip을 완화했음을 확인했다. Normal의 70-0과 80-14는 서로 다른 데이터 출처에서 B·8·D 최악 Flip이 달랐다. Normal EXP유017 모델의 256×128/alpha=0.03 구조 효과와 전처리 효과는 혼합하지 않는다.

Combined는 공식 Training 76,800 + 타 작성자 Hand-made 1,920을 학습하므로 Normal 결과를 그대로 적용해 성공했다고 주장하지 않는다. 기존 Combined EXP-CMB-004의 CoM-only 85.8594%, Size+CoM 88.9453%는 1-Seed 4-Fold의 역사적 수치다.

## 사전에 고정한 네 가지 독립 조건

| ID | Official Train 크기 균형 증강 | 증강 대상 Stroke 혼합 | 모든 입력 평가 |
|---|---|---|---|
| A | 0% (CoM-only 대조) | 없음 | CoM 정렬 |
| B | 70% / 16·18·20·22·24px | 0 | CoM 정렬 |
| C | 80% / 동일한 5개 목표 | 0 | CoM 정렬 |
| D | 80% / 동일한 5개 목표 | 14%, topology guard | CoM 정렬 |

- 각 공식 원본 행은 **딱 1개 학습 표현**만 제공한다(대체 증강, 총 76,800개). 목표 크기는 각 실제 정답 클래스에서 균형 있게 배정하되, 학습 이후 레이블은 전처리 선택/추론에 사용하지 않는다.
- Normal의 검증된 nested_subset·scale_base·stroke_jitter를 변경 없이 호출. 70% 선택은 80% 선택의 부분집합이고 선택 Seed는 20261116으로 고정한다.
- 각 Fold 학습은 공식 76,800장 + 나머지 세 작성자 손글씨 1,920장. **학습용 Handmade 세 작성자는 CoM만** 적용하여 Official에 대한 증강의 역할을 분리. 보류 작성자 640장 및 공식 Validation 19,200장은 CoM만 적용.
- 모델 설정을 4조건 모두 동일하게 고정: PixelScaler255 + sklearn MLPClassifier 기본 은닉층 100, alpha=0.0001, ReLU/Adam, random_state는 Seed 42·43·44. Fold별로 가중치 처음부터 학습(4조건 × 4Fold × 3Seed = 48모델).
- Official Train 그룹 분할 SHA·Hand-made 41~44 SHA 원본 및 기존 label 순서 재검증. Official Original Test/교수님 Test 및 Normal EXP유018 신규 480장 미사용. 기존 EXP-CMB-001~004 결과/모델과 Normal 원본 파일 수정 금지.

## 연구 종료 / 통과 기준 (Accuracy 하한 없음)

1. **전처리 조작 확인:** 학습에 실제 대체된 이미지 수, 5개 목표 크기×16클래스 빈도, Size·Width/Height/Ink Mass의 집단 분포, Stroke topology-guard 거절 수 기록.
2. **OOF·공식 Val 대응 비교:** 각 Seed×Fold 동일 이미지의 Accuracy·Macro F1, 숫자/문자·작성자·클래스별 결과, 오답→정답·정답→오답 및 주요 혼동 0↔D·8↔B·5↔F 방향별 전환. 학습시간·loss/iteration 저장.
3. **크기 Shortcut 직접 개입:** 학습한 Fold 모델 고정, 보류 Handmade 및 OfficialVal 각 이미지를 0.8/0.9/1.0/1.1/1.2배 확대·축소 후 CoM 정렬하여 16개 실제 클래스의 family flip, P(letter) 변화와 정상 Accuracy를 분해. 숫자 1.2배, 문자 0.8배, B/8/D 최악을 전체 평균과 별도로 보고.
4. **요인 분리:** B(70-0) vs C(80-0)에서 증강 비율, C(80-0) vs D(80-14)에서 Stroke 추가 효과를 같은 Seed/Fold로 비교. A vs B/C/D에서 균형 Size 증강 자체의 효과 검증.
5. **해석·반례:** Shortcut이 감소했지만 정확도가 떨어지는 경우도 성공으로 포장하지 않고, 작성자/클래스 역전 및 OfficialVal 손실 설명. 효과가 상충하거나 기전이 불명확하면 판정 보류 후 추가 통제 분석.
6. **외부 일반화 경계:** 기존 4명 OOF와 OfficialVal은 반복 개발 데이터. Normal EXP유018 신규 480장 결과를 Combined 최적화에 섞지 않는다. 독립 작성자 평가 없이 Shortcut 완전 제거 또는 최종 배포 성공을 주장하지 않는다.

**이번 실험은 전처리만 비교한다.** Normal 최종 256×128/alpha0.03 HP 적용, Hand-made 학습 비중 변화, 새로운 480장 검증은 별도 연구 질문이다.

결과 예정 위치: Combined/results/Balanced_Scale_Stroke/ 와 Combined/scripts/exp_cmb005_balanced_scale_stroke.py. 사용자 승인 전 main 병합/배포 금지.
