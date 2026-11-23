# Normal 스크립트

결과는 `out/`에 저장된다. 공통 부분(데이터·전처리·평가 함수)은 `run_all.py`에 있고 다른 스크립트가 불러온다.

| 명령 | 내용 |
|---|---|
| `python run_all.py n3` | N3 재학습 + D1 분기 gain 히트맵 |
| `python d2.py` | D2 오분류 사례 그림 |
| `python run_all.py n5` | 투영+HOG 특징 추출·저장(`out/Ftr.npy` 등, Combined도 사용) + N5 + 학습곡선 |
| `python more.py hog_42` / `proj_42` / `n5_43` … | 특징 분리, 시드 반복 (N7 = `hog_*`) |
| `python lc7.py` | N7 학습곡선 |
| `python n8.py 42` … | N8 정규화 투영 |
| `python l2run.py n9 42` … | **최종 N9** |
| `python lcl2.py n9` | N9 학습곡선 |
| `python night.py n10 42`, `python night.py n11 42` | N10·N11 |
| `python extra.py n1` | 혼동행렬용 N1 재학습 |
| `cm.py` | 혼동행렬 그림 함수 |

`colab/`: Colab에서 실행한 노트북 (N1~N6, C1, D1~D3). 0번 셀(scikit-learn 1.8.0 설치) 후 1번 셀부터 실행.
