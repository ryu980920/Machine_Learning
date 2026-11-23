# Combined 스크립트

먼저 Normal의 `python run_all.py n5`를 실행해 특징 캐시를 만든다. 결과는 `Normal/scripts/out/`에 함께 저장된다.

| 명령 | 내용 |
|---|---|
| `python run_all.py c2_41` … `c2_44` | C2 Fold별 |
| `python c3.py 41` … `44` | C3 Fold별 (시드 42) |
| `python c4.py c4 41 42` / `python c4.py c3 41 43` … | C4 후보 / C3 시드 반복 |
| `python c5.py 41 43` | C5 (첫 Fold에서 중단) |
| `python diag41.py 0 43 30`, `python diag41.py 1.0 43 10` | 잎 값 폭발 진단 |
| `python l2run.py c6 42 41` … | **최종 C6** (시드 42·43·44 × 작성자 41~44) |
| `python lcc.py`, `python lcl2.py c6` | C3·C6 학습곡선 |
| `python extra.py c1_41` … | 혼동행렬용 C1 재학습 |
| `python f41.py` | C2 작성자 41 Fold 고정 140회 진단 |
| `cm.py` | 혼동행렬 그림 함수 |
