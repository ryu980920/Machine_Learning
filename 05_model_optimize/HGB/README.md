# HistGradientBoosting — 최적화 기록

- [Normal 최적화](Normal/README.md) — 공식 MNIST·EMNIST Training만 학습
- [Combined 최적화](Combined/README.md) — 공식 Training + 다른 작성자 Hand-made 학습, 작성자 보류(Leave-One-Writer-Out) 평가

| 최종 모델 | 구성 | 공식 Validation | 개발용 Hand-made |
|---|---|---|---|
| **Normal N9** | 크기 30%·획 28% 증강 + CoM + 픽셀 784 + HOG 324, `l2_regularization=1.0` | 98.28 ± 0.06% | 91.43 ± 0.25% |
| **Combined C6** | 같은 구성 + 보류 작성자 제외 Hand-made 1,920장 학습 | Fold 평균 98.26% | 93.39 ± 0.47% (OOF) |

수치는 시드 42·43·44 평균이다. Hand-made 2,560장은 개발용 평가 자료이며 독립 Test가 아니다. 공식 Test 16,000장은 사용하지 않았다.

**핵심 발견:** `l2_regularization=0`에서 학습 손실이 0에 가까워지면 잎 값(−G/(H+λ))이 수백~수천으로 폭발해 학습이 붕괴했다. λ=1.0으로 해결했다. 근거는 [Combined/results/diag41_*.json](Combined/results/)과 [그림](Normal/results/figures/fig_leaf.png).

**실행 환경:** N1~N6·C1은 Google Colab(scikit-learn 1.8.0). Colab 무료 CPU에서 학습 1회가 25~40분 걸려, 이후 실험은 같은 코드를 별도 2코어 Linux 환경(scikit-learn 1.9.1)에서 실행했다. 같은 N3 결과 차이는 0.1%p 이내이며 비교는 같은 환경 결과끼리만 했다.

데이터 로드·분할·CoM·크기/획 증강은 `05_model_optimize/MLP/` 아래의 팀 공통 함수(`load_official`, `load_writers`, `prepareraw`, `process`)를 그대로 불러온다.
