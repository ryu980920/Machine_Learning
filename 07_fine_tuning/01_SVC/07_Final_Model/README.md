# SVC 현재 선정: C2_G2

Accuracy 94.453125%, 공식 19,003/19,200. C=2, gamma=2×scale, 82,080 학습행, seed42. SHA-256 f88be338da4065315d943ca4150b5a425c24adf63733fada7da212c837ae4398. 기존 시간 10.9942 → 12.7130 ms/image; 권장 기준 미충족을 알고 정확도 우선 승인. 2차 Fine-Tuning 미반영.

- [선정 Manifest](CURRENT_FINAL_MODEL_SELECTION.json) · [현재 모델 검증 Notebook](Final_Validation.ipynb) · [C2_G2 상세 결과](../03_C_Gamma/Details/C2_G2/metrics.json)
- [Full 비교](full_fit_summary.csv) · [동일 환경 시간 비교](C2_G2_matched_full_latency.csv) · [실제 Full 로그](Execution/)
- [최종 학습·추론 Notebook (별도 통합 코드)](../../../08_consolidated_notebooks/06_Final_Models/Individual/SVC_Combined.ipynb)
- [추론 스크립트](extract_inference.py): `python extract_inference.py --kit <trusted-kit.zip> --model <trusted-C2_G2.joblib> --model-sha256 f88be338da4065315d943ca4150b5a425c24adf63733fada7da212c837ae4398 --input <input.npz> --out <new.csv>`

## 역사적 Full 재현 절차
[03_C_Gamma/Experiment.ipynb](../03_C_Gamma/Experiment.ipynb) 실행 시 별도 로컬 결과 폴더에 runtime arrays 및 기존 Full 스크립트 3개가 준비된다. Notebook이 출력한 root에서 실행한다. `python run_svc_full.py` → (실패/중단 여부 및 기존 metrics 점검) `python run_svc_remaining.py` → `python run_svc_sw15_followup.py`. 각 스크립트는 역사적 baseline 대비 gate를 적용하며, 새 연구 기준은 00_Common/Followup_Protocol.json이다. 순차 실행, 중복 실행 금지. Full 실행 코드: [Code/](Code/). 기존 공개 결과/가중치를 덮어쓰지 않는다. 공식 직접 계산 및 native 평가의 차이는 각 후보 review_manifest 참조. 기존 C2_G2 바이너리는 팀 리더에게 별도 요청; GitHub에는 없음.
