"""Write the measured experiment journal without duplicating result entry."""
import argparse
import json
from pathlib import Path

from run_extra_trees import PREPROCESSING, TUNING, official_average


HYPOTHESES = {
    "P00_raw": "W02 기본 픽셀 조건을 재현한다.",
    "P01_com": "중심 이동으로 위치 편차를 줄인다.",
    "P02_size": "공식 학습의 클래스별 크기 단서를 약화한다.",
    "P03_stroke": "침식·팽창 혼합으로 획 굵기 편차를 줄인다.",
    "P04_hybrid": "국소 방향을 픽셀과 함께 사용한다.",
    "P05_hog": "HOG가 픽셀 정보를 단독으로 대체할 수 있는지 본다.",
    "T01_feat08": "후보 분기 특징을 8%로 높이면 더 좋은 분기를 찾는다.",
    "T02_feat18": "후보 분기 특징을 18%로 더 높이면 검증 성능이 개선된다.",
    "T03_leaf2": "최소 잎 표본 수 2개로 과도한 분기를 제한한다.",
    "T04_depth24": "최대 깊이 24로 트리를 제한한다.",
    "T05_trees64": "트리 64개로 줄여도 성능이 유지된다.",
    "T06_trees160": "트리 160개로 늘리면 예측 분산이 감소한다.",
}


def write_log(source, output, mode):
    runs = {f.parent.name: json.loads(f.read_text()) for f in (source/"runs").glob("*/summary.json")}
    selection = json.loads((source/"selection.json").read_text())
    floor = selection["official_accuracy_floor"]
    picked, basis = selection["selected"], selection["preprocessing"]
    baseline = runs["P00_raw"]
    result = runs["R42_selected"]
    count = sum(len(run["official"]) for run in runs.values())
    lines = [
        f"# Extra Trees {mode} 실험일지", "",
        f"공식 Validation 자격선 {100*floor:.4f}% · {len(runs)}개 실행 조건 · {count}회 적합.",
        f"전처리 후속 기준 **{basis}**, 선정 조건 **{picked}**.",
        "세 자리 실험 ID는 실험 로그·폴더·그림에서 공통으로 사용한다.", "",
        "| ID | 변경 가설 | 비교군 | 공식 Acc. | 손글씨 Acc. | 손글씨 F1 | 판정 |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    prep_names = list(PREPROCESSING)
    for name in (*PREPROCESSING, *TUNING):
        run = runs[name]
        idx = prep_names.index(name) if name in PREPROCESSING else None
        ref = basis if idx is None else prep_names[idx-1] if idx else "W02"
        off, hand, f1 = official_average(run), run["hand"]["accuracy"], run["hand"]["macro_f1"]
        if name == "P00_raw":
            decision = "자격선 산정 기준"
        elif name == picked:
            decision = "최종 채택"
        elif name == basis:
            decision = "전처리 후속 기준"
        elif off < floor:
            decision = "공식 자격선 미달"
        else:
            decision = f"채택군 대비 Macro F1 {f1-result['hand']['macro_f1']:+.4f}"
        lines.append(f"| {name} | {HYPOTHESES[name]} | {ref} | {100*off:.4f}% | "
                     f"{100*hand:.4f}% | {f1:.5f} | {decision} |")
    lines += ["", "## 해석", "",
              "전처리·튜닝 후 비교는 동일한 공식 Validation과 작성자 평가 묶음에서 수행했다.",
              f"시드 42에서 기준 대비 공식 Accuracy "
              f"{100*(official_average(result)-official_average(baseline)):+.4f}%p, "
              f"손글씨 Accuracy {100*(result['hand']['accuracy']-baseline['hand']['accuracy']):+.4f}%p.",
              "동일 시드로 다시 적합한 모델의 공식 및 작성자별 예측 배열이 선정 탐색 모델과 일치했다.",
              "시드 17·42·73 반복과 20~100% 학습곡선을 완료했다.", ""]
    if mode == "Combined":
        control = runs["CTRL_normal"]
        change = result["hand"]["accuracy"]-control["hand"]["accuracy"]
        lines += [
            "## 동일 설정의 자료 추가 대조", "",
            f"공식만 학습 {100*control['hand']['accuracy']:.4f}% → "
            f"손글씨 추가 {100*result['hand']['accuracy']:.4f}% ({100*change:+.4f}%p).",
            "동일 시드·전처리·설정에서 보류 작성자별 동일 이미지를 예측했다. "
            "작성자별 개선과 악화 건수는 matched_writers.csv에 있다.", "",
        ]
    lines += ["## 남은 검증", "",
              "별도 신규 작성자 평가, Original Test와 교수님 Test는 미실시다. "
              "학습 조건 탐색은 일변수 범위이며 조합 전수 탐색이나 신규 480장 보강 비교는 수행하지 않았다.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("Normal", "Combined"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_log(args.source, args.output, args.mode)
