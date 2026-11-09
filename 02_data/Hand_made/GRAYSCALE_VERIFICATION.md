# Hand-made Grayscale 데이터셋 변환·검증 기록

- 변환일: 2026-11-09
- 원본: 팀원 제출 `ML-Project.zip` 내부 `ml26_41.zip` ~ `ml26_44.zip`의 **JPG 2,560장** (원본 ZIP은 저장소에 보관하지 않음)
- 방법: 원본 28×28 JPG를 Pillow `convert("L")`로 읽어 `uint8` 0~255 밝기값 보존. 42·44번만 `255 - gray`로 극성 반전. **임계값 이진화, 크기 변경, 중앙 정렬, 증강 없음**.
- 통일 기준: 어두운 배경(0), 밝은 글자(최대 255), Shape `(640,28,28)`, `uint8`, Key 순서 `img`, `label`, `label_id`.
- 16클래스 `0123456789ABCDEF`, 작성자당 640장(각 클래스 40장). 순서는 `0_01,1_01,...,F_01,0_02,...,F_40`.
- 각 NPZ 저장 직후 원본 배열과 픽셀·레이블·레이블 ID의 완전 일치 검사 통과. 변환 코드를 독립적으로 재실행하여 **4개 NPZ 모두 바이트 단위 재현** 확인.

## 결과

| 파일 | 이미지 수 | 픽셀값 종류 | 1~254 중간 밝기 픽셀 비율 | 극성 반전 | 크기 (byte) | SHA-256 |
|---|---:|---:|---:|---|---:|---|
| `ml26_41.npz` | 640 | 256 | 28.54% | 아니오 | 164,198 | `09e9e472bdb50ee7278513e0e453e09d1b6444d1d11fca872744bf16713d9feb` |
| `ml26_42.npz` | 640 | 4 | 1.76% | 예 | 39,683 | `d6fb990a20cde1fd5d0a396b8e3cec65b39e83571b443a67a387770c0c073c67` |
| `ml26_43.npz` | 640 | 256 | 14.06% | 아니오 | 124,939 | `10db20997c727c1bc1f1299f8e89626fadd3ec9618468d6759c86b0d2ddc4a66` |
| `ml26_44.npz` | 640 | 256 | 14.21% | 예 | 118,750 | `e2ab5637295e2430a90e2bc7032d65eb470d249c2a58e287a3a6e9da811e7e02` |

**주의:** 42번의 원본 JPG에는 중간 밝기 정보가 극히 적어 반전해도 고유 픽셀값은 4종이다. 41·43·44의 중간값에는 JPG 압축으로 인한 밝기 변화도 포함될 수 있다. 따라서 픽셀 범위/극성을 일치시킨 것이지, 필체·선폭·이미지 생성 과정까지 동일하게 만든 것은 아니다.

## 재현 코드

```bash
pip install numpy pillow
python 02_data/notebooks/convert_hand_made.py --source /path/to/ML-Project.zip --output-dir 02_data/Hand_made
# 또는
python 02_data/notebooks/convert_hand_made.py --input-dir /path/to/individual_zips --output-dir 02_data/Hand_made
```

실험 3에서는 **현재 Grayscale NPZ**를 그대로 입력에 사용한다. 이전 0/1 Binary NPZ를 전제로 `*255`를 수행하면 입력이 잘못되므로 금지한다. 기존 scikit-learn Pipeline 안의 `PixelScaler255`는 정상적으로 유지한다. 공식 Test 16,000장은 데이터셋 재구성에 사용하지 않았다.