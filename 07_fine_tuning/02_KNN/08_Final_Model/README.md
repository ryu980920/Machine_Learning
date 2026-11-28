# KNN 현재 선정: 기존 모델 유지

Accuracy 92.34375%, 공식 18,856/19,200. k=3, Euclidean, distance, 113,920장. SHA-256 ff046fffafc27cf19bb0a3e6a997b55c2af42b9494c556d6851c3ea06a5df6d9. PCA128은 속도 후보일 뿐 미채택.

- [현재 선정 Manifest](CURRENT_FINAL_MODEL_SELECTION.json) · [현재 모델 검증 Notebook](Final_Validation.ipynb)
- [최종 학습·추론 Notebook (별도 통합 코드)](../../../08_consolidated_notebooks/06_Final_Models/Individual/KNN_Combined.ipynb)
- [추론 스크립트](extract_inference.py): `python extract_inference.py --kit <trusted-kit.zip> --model <trusted-KNN.joblib> --model-sha256 ff046fffafc27cf19bb0a3e6a997b55c2af42b9494c556d6851c3ea06a5df6d9 --input <input.npz> --out <new.csv>`

입력 확보: [공통 안내](../../00_Common/README.md). 기존 체크포인트를 Actions artifact에서 받아 해시 대조. 원본 Test 금지.
