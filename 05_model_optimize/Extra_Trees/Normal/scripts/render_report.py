"""Render a Korean PDF from measured Extra Trees results and generated charts."""
import argparse
import csv
import json
from html import escape
from pathlib import Path

import numpy as np
from weasyprint import HTML

from run_extra_trees import LABELS, PREPROCESSING, TUNING, WRITERS, choose, official_average

STYLE = """
@page {
  size: A4;
  margin: 14mm 15mm 17mm;
  border-top: 3pt solid #1261a0;
  padding-top: 4mm;
  @bottom-left { content: string(report-label); font: 8pt sans-serif; color: #496575; }
  @bottom-right { content: counter(page, decimal-leading-zero); font: 8pt sans-serif; color: #496575; }
}
html {font-family:"Noto Sans CJK KR",sans-serif;}
body {font-size:10pt;color:#172f41;line-height:1.41;padding:0;margin:0;}
article {break-after:page;}
article:last-child {break-after:auto;}
.report-label {string-set:report-label content();font-size:0;height:0;overflow:hidden;}
h1 {font-size:30pt;color:#0c3656;line-height:1.20;margin:4pt 0 12pt;}
h2 {font-size:22pt;color:#0c3656;line-height:1.25;margin:4pt 0 12pt;}
h3 {font-size:13pt;color:#1261a0;margin:15pt 0 5pt;}
p {margin:0 0 9pt;}
.kicker {color:#1261a0;font-size:9pt;font-weight:bold;margin-bottom:6pt;}
.note {font-size:9pt;color:#3c576a;}
.lead {font-size:12pt;margin-bottom:14pt;}
table {width:100%;border-collapse:collapse;margin:7pt 0 12pt;}
th {font-weight:bold;background:#e6eff5;text-align:left;color:#103b57;}
td,th {border-bottom:1px solid #d9e2ea;padding:5pt 6pt;font-size:8.8pt;}
img {display:block;width:100%;height:auto;max-height:65mm;object-fit:contain;}
.errors img {max-height:46mm;}
.closing img {max-height:47mm;}
.review-errors img {max-height:28mm;}
.review-errors td,.review-errors th {padding:3pt 6pt;}

"""

PREP_TEXT = {
    "P00_raw": "원본 픽셀",
    "P01_com": "CoM 정렬",
    "P02_size": "CoM + 크기",
    "P03_stroke": "CoM + 크기·획",
    "P04_hybrid": "크기·획 + 픽셀·HOG",
    "P05_hog": "크기·획 + HOG 단독",
}
TUNE_TEXT = {
    "T01_feat08": "max_features = 0.08",
    "T02_feat18": "max_features = 0.18",
    "T03_leaf2": "min_samples_leaf = 2",
    "T04_depth24": "max_depth = 24",
    "T05_trees64": "n_estimators = 64",
    "T06_trees160": "n_estimators = 160",
}


def paragraph(value, style=""):
    return f'<p class="{style}">{escape(str(value))}</p>'


def title(kicker, heading, first=False):
    tag = "h1" if first else "h2"
    return paragraph(kicker, "kicker") + f"<{tag}>{escape(heading)}</{tag}>"


def sub(heading):
    return f"<h3>{escape(heading)}</h3>"


def table(headers, rows):
    head = "".join(f"<th>{escape(str(v))}</th>" for v in headers)
    lines = "".join("<tr>"+"".join(f"<td>{escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{lines}</table>"


def figure(name, caption):
    return f'<img src="{name}.png" alt=""/>' + paragraph(caption, "note")


def pct(value):
    return f"{100*value:.2f}%"


def delta(value):
    return f"{100*value:+.2f}%p"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def average(rows, function):
    return float(np.mean([function(row) for row in rows]))


def seed_stats(seeds, function):
    values = [function(r) for r in seeds]
    return f"{pct(np.mean(values))} ± {100*np.std(values, ddof=1):.2f}%p"


def pages(out, results, mode, review=None):
    comb = mode == "Combined"
    runs = {p.parent.name: json.loads(p.read_text()) for p in (out / "runs").glob("*/summary.json")}
    selection = json.loads((out / "selection.json").read_text())
    audit = json.loads((out / "data_audit.json").read_text())
    env = json.loads((out / "environment.json").read_text())
    baseline, final = runs["P00_raw"], runs["R42_selected"]
    seeds = [runs[f"R{s}_selected"] for s in (42, 17, 73)]
    sel, config = selection["selected"], final["config"]
    hand = lambda r: r["hand"]["accuracy"]
    description = PREP_TEXT.get(selection["preprocessing"])
    final_method = (PREP_TEXT.get(sel) or description) + (f" · {TUNE_TEXT[sel]}" if sel in TUNE_TEXT else "")
    count = sum(len(r["official"]) for r in runs.values())
    hand_kind = "작성자 보류 OOF" if comb else "팀 손글씨 개발 평가"
    split = "공식 Train 76,800 + 손글씨 1,920장/Fold" if comb else "공식 Train 76,800장"
    summary = title("머신러닝 · 16클래스 손글씨 분류", f"Extra Trees {mode}", True)
    summary += paragraph(f"{hand_kind} 정확도가 {pct(hand(baseline))}에서 "
                         f"{pct(hand(final))}로 증가했다. 공식 Validation은 "
                         f"{pct(official_average(baseline))}에서 {pct(official_average(final))}로 변했다.", "lead")
    summary += table(("평가 지표", "원본 픽셀 · 시드 42", "선정 · 시드 42", "선정 · 3시드 평균"),
                     ((label, pct(fn(baseline)), pct(fn(final)), seed_stats(seeds, fn)) for label, fn in (
                         ("공식 Accuracy", official_average),
                         ("공식 Macro F1", lambda r: official_average(r, "macro_f1")),
                         ("공식 Macro Precision", lambda r: official_average(r, "macro_precision")),
                         ("공식 Macro Recall", lambda r: official_average(r, "macro_recall")),
                         ("손글씨 Accuracy", hand),
                         ("손글씨 Macro Precision", lambda r: r["hand"]["macro_precision"]),
                         ("손글씨 Macro Recall", lambda r: r["hand"]["macro_recall"]),
                         ("손글씨 Macro F1", lambda r: r["hand"]["macro_f1"]),
                     )))
    summary += sub("선정 구성") + paragraph(f"{sel} · {final_method} · {config['n_estimators']}개 트리. "
                                               f"학습 구성은 {split}이다.")
    if comb:
        left = runs["CTRL_normal"]
        summary += paragraph(f"선정 설정을 고정한 대조에서 손글씨 1,920장을 추가하자 "
                             f"OOF와 동일한 2,560장의 정확도가 {pct(hand(left))}에서 "
                             f"{pct(hand(final))}로 변했다({delta(hand(final)-hand(left))}).")
    summary += sub("기존 팀 보고서와의 비교")
    comparators = (("MLP Combined", "97.08%", "92.36%"),
                   ("HGB Combined", "98.26%", "93.39%")) if comb else (
                   ("MLP Normal", "96.54%", "92.07%"),
                   ("HGB Normal", "98.28%", "91.43%"))
    summary += table(("보고서 최종 후보", "공식 Validation", "손글씨 개발"),
                     [(f"Extra Trees {mode}", pct(average(seeds, official_average)), pct(average(seeds, hand))),
                      *comparators])
    summary += paragraph("MLP·HGB 수치는 각 보고서의 최종 3시드 평균이다. "
                         "Extra Trees 역시 시드별 점수를 평균했으며 ±값은 표본 표준편차다. 보고서마다 시드, 후보 선정 기준과 실행 환경은 다르다. "
                         "Original Test 16,000장과 교수님 Test는 이 연구에서 평가하지 않았다.", "note")
    if comb and review and (review / "paired_controls.csv").is_file():
        summary += paragraph("후속 대조의 시드별 결과와 독립 평가용 후보 결정은 마지막 쪽에 정리했다.", "note")
    yield summary

    data = title("데이터 감사와 기준선", "공식 자료와 작성자별 분리")
    data += table(("자료", "표본 수", "사용 방식"),
                  [("공식 Train", "76,800", "모든 실험의 학습 원천"),
                   ("공식 Validation", "19,200", "동일 분할로 후보를 비교"),
                   ("41·42·43·44번", "각 640", "작성자별 16클래스 × 40장"),
                   ("Fold 학습 손글씨" if comb else "손글씨 개발 평가",
                    "1,920 / Fold" if comb else "2,560",
                    "보류 작성자 제외" if comb else "공식 Train으로 학습한 모델의 예측")])
    data += paragraph("공식 원천 96,000장은 StratifiedGroupKFold 5분할의 첫 Fold를 사용했다. "
                      "원본과 공식 증강본의 source group을 함께 묶었다. 팀 보존 인덱스·그룹 ID와 완전히 일치했고 "
                      "Train–Validation 그룹 교집합은 0이었다. 다섯 NPZ의 SHA-256은 팀 기록과 일치했다. "
                      "작성자 간 및 공식 자료와 손글씨 간 동일 원본 픽셀 배열 교집합도 0이었다.")
    if comb:
        data += paragraph("Fold마다 공식 Train과 다른 작성자 세 명의 데이터로 새 모델을 학습했다. "
                          "보류 손글씨 예측 640장씩을 모아 OOF를 계산했다. 공식 Validation은 "
                          "같은 19,200장에 대한 Fold별 지표의 평균이다.")
    else:
        data += paragraph("P00의 공식 Validation 96.96875%와 Macro F1 0.96954는 W02 실험 2의 "
                          "공표값 96.97%와 0.9695를 재현했다. 최신 팀 손글씨 정확도 61.60%도 기존 기록과 일치했다.")
    data += figure("input_geometry", "획 픽셀은 밝기 32 이상을 세었다. 공식 문자의 긴 변은 숫자보다 3.91 px 길다. "
                   "팀 손글씨의 숫자·문자 크기는 작성자별로 비슷하다.")
    data += paragraph("작성자 41번의 숫자 획 픽셀 수는 평균 86.0개로 공식 Validation 숫자 134.0개의 64.2%였다. "
                      "이 분포 차이를 크기와 획 혼합 대조의 가설로 사용했다.")
    data += paragraph(f"입력 파일 해시·레이블 분포·분할 근거: data_audit.json, official_split_indices.npz. "
                      f"기준 소스 커밋 {audit['repository_commit'][:12]}.", "note")
    yield data

    prep = title("전처리와 특징 비교", "CoM·크기·획·HOG")
    rows = []
    for name in PREPROCESSING:
        r = runs[name]
        rows.append((name[:3], PREP_TEXT[name], pct(official_average(r)), pct(hand(r)),
                     f"{r['hand']['macro_f1']:.4f}",
                     "후속 탐색" if name == selection["preprocessing"] else
                     "기준 미달" if official_average(r)<selection["official_accuracy_floor"] else "비교군"))
    prep += table(("ID", "변경 조건", "공식 Acc.", "손글씨 Acc.", "손글씨 F1", "판정"), rows)
    for a, b, label in (("P00_raw", "P01_com", "CoM 효과"),
                        ("P01_com", "P02_size", "크기 증강 효과"),
                        ("P02_size", "P03_stroke", "획 혼합 효과"),
                        ("P03_stroke", "P04_hybrid", "HOG 추가 효과"),
                        ("P04_hybrid", "P05_hog", "혼합 특징의 픽셀 제거")):
        l, r = runs[a], runs[b]
        prep += paragraph(f"{label} · 공식 {delta(official_average(r)-official_average(l))}, "
                          f"손글씨 {delta(hand(r)-hand(l))}. 손글씨 Macro F1 "
                          f"{l['hand']['macro_f1']:.4f} → {r['hand']['macro_f1']:.4f}.")
    aug = runs["P03_stroke"]["preprocessing"]["augmentation"]
    prep += paragraph(f"CoM은 픽셀 밝기 무게중심을 중앙에 옮긴다. 크기 조건은 공식 Train {aug['n_aug']:,}장을 "
                      "긴 변 16·18·20·22·24 px에 클래스별 균형 배정해 교체했다. "
                      f"획 조건은 침식·팽창 혼합 0.28을 추가해 {aug['n_stroke_changed']:,}장에 적용했고 "
                      f"위상이 바뀐 {aug['n_stroke_rejected']:,}장은 혼합을 취소했다.")
    prep += paragraph("HOG는 9방향, 7×7 셀, 2×2 셀 블록, L2-Hys의 324차원이다. "
                      "혼합 특징은 원 픽셀 784개와 합쳐 1,108차원이다. 모든 전처리 비교에서 "
                      "트리 100개와 기본 분기 설정을 사용했다. 크기·획 증강은 공식 Train에만 적용했다. CoM은 밝기 32 이상 획의 무게중심을 (13.5, 13.5)로 이동하며 정수 이동 후 0으로 채운다.")
    yield prep

    tune = title("Extra Trees 튜닝", "공식 검증과 손글씨의 절충")
    tune += figure("stage_comparison", "P군은 전처리·특징을 비교했다. T군은 선택한 P 설정을 기준으로 한 변수만 바꿨다. "
                   "주황 점선은 공식 Validation 자격선이며 원형 표시는 최종 선택 조건이다.")
    tune += table(("실험", "변경 변수", "공식 Acc.", "손글씨 Acc.", "손글씨 F1", "학습/모델"),
                  [(n[:3], TUNE_TEXT[n], pct(official_average(runs[n])), pct(hand(runs[n])),
                    f"{runs[n]['hand']['macro_f1']:.4f}",
                    f"{average(list(runs[n]['timing'].values()), lambda z: z['fit_seconds']):.1f}s")
                   for n in TUNING])
    tune += paragraph(f"자격선은 P00 기준선 −0.50%p에서 정한 {pct(selection['official_accuracy_floor'])}였다. "
                      "이를 충족한 후보 중 손글씨 Macro F1, 손글씨 Accuracy, 공식 Accuracy, 적은 트리 수 순으로 "
                      f"선정했다. 최종 {sel[:3]}의 공식 Accuracy는 {pct(official_average(final))}, "
                      f"손글씨 Macro F1은 {final['hand']['macro_f1']:.4f}였다.")
    ranked = [runs[n] for n in (*PREPROCESSING, *TUNING) if n != sel]
    runner_up = choose(ranked, selection["official_accuracy_floor"])
    top_time = average(list(runs[sel]["timing"].values()), lambda row: row["fit_seconds"])
    next_time = average(list(runner_up["timing"].values()), lambda row: row["fit_seconds"])
    tied = hand(final) == hand(runner_up)
    comparison = (f"두 조건의 손글씨 정확도는 {pct(hand(final))}로 같았다." if tied else
                  f"손글씨 정확도 차이는 {delta(hand(final)-hand(runner_up))}였다.")
    tune += paragraph(f"차순위 {runner_up['name'][:3]} 대비 손글씨 Macro F1 차이는 "
                      f"{final['hand']['macro_f1']-runner_up['hand']['macro_f1']:+.5f}였다. "
                      f"{comparison} 탐색 중 평균 적합 시간은 차순위 {next_time:.1f}초, "
                      f"선정 조건 {top_time:.1f}초였다.")
    tune += paragraph(f"최종 설정 · {config['n_estimators']} trees, max_features={config['max_features']}, "
                      f"min_samples_leaf={config['min_samples_leaf']}, max_depth={config['max_depth']}, bootstrap=False. "
                      "결과는 비교한 일변수 범위에 해당하며 하이퍼파라미터 조합의 전수 탐색은 수행하지 않았다.", "note")
    yield tune

    errors = title("클래스별 오류 분석", "유사 글자의 오분류 방향")
    errors += figure("confusion_matrices", "행은 정답, 열은 예측이다. 공식 Validation은 "
                     + ("Fold별 같은 행의 정규화 행렬을 평균했다." if comb else "시드 42 모델의 행 정규화 행렬이다."))
    errors += figure("directional_errors", "팀 손글씨는 정답 클래스마다 160장이다. 막대 높이는 오답 건수다.")
    before, after = baseline["hand"]["focused_errors"], final["hand"]["focused_errors"]
    errors += table(("방향", *(k.replace("->", "→") for k in before)),
                    [("기준", *before.values()), ("선정", *after.values())])
    changed = sorted(before, key=lambda k: after[k]-before[k])
    best, weakest = changed[0], changed[-1]
    errors += paragraph(f"{best.replace('->', '→')} {before[best]}→{after[best]}건으로 가장 크게 줄었다. "
                        f"{weakest.replace('->', '→')} {before[weakest]}→{after[weakest]}건은 여전히 확인할 오류다.")
    worst = sorted(final["hand"]["classes"], key=lambda row: row["recall"])[:3]
    errors += paragraph("최종 손글씨에서 Recall이 가장 낮은 클래스는 "
                        + ", ".join(f"{r['label']} {pct(r['recall'])} ({r['errors']}건 오류)" for r in worst)
                        + "였다. 방향별 건수 감소와 클래스별 Recall은 CSV에서 함께 확인할 수 있다.")
    errors += figure("error_examples", "원본 손글씨의 행 인덱스를 별도 CSV에 기록했다. "
                     "윗줄은 동일 방향 오답이 정답으로 개선된 사례, 아랫줄은 같은 방향 오답이 남은 사례다.")
    yield errors

    learning = title("학습 규모와 실행 시간", "데이터 수 기반 Learning Curve")
    learning += figure("learning_curve", "시드 42. 공식 Train의 클래스 층화 중첩 부분집합을 사용했다. "
                       "훈련 정확도는 적합에 사용한 표본에 대한 값이다. "
                       + ("Combined는 모든 점과 Fold에서 손글씨 1,920장을 고정했다." if comb else "손글씨는 평가에만 사용했다."))
    curve = read_csv(results / "learning_curve.csv")
    learning += table(("공식 Train", "손글씨 Train", "Train Acc.", "공식 Val.", "손글씨 Acc.", "적합 시간"),
                      [(f"{int(r['official_train_n']):,}", f"{int(r['handmade_train_n']):,}",
                        pct(float(r["train_accuracy"])), pct(float(r["official_accuracy"])),
                        pct(float(r["hand_accuracy"])), f"{float(r['fit_seconds_mean']):.1f}s") for r in curve])
    first, last = curve[0], curve[-1]
    peak = max(curve, key=lambda row: float(row["hand_accuracy"]))
    learning += paragraph(f"공식 학습량을 20%에서 100%로 늘리자 공식 Validation은 "
                          f"{delta(float(last['official_accuracy'])-float(first['official_accuracy']))}, "
                          f"손글씨 정확도는 {delta(float(last['hand_accuracy'])-float(first['hand_accuracy']))} 변했다.")
    if comb:
        at_sixty = next(row for row in curve if row["run"] == "LC60")
        hand_share = lambda row: int(row["handmade_train_n"]) / (
            int(row["official_train_n"]) + int(row["handmade_train_n"]))
        learning += paragraph(
            f"46,079장과 61,440장에서 손글씨 정확도 {pct(float(at_sixty['hand_accuracy']))}가 공동 최고였다. "
            f"76,800장의 공식 Validation은 46,079장 대비 "
            f"{delta(float(last['official_accuracy'])-float(at_sixty['official_accuracy']))}, "
            f"손글씨는 {delta(float(last['hand_accuracy'])-float(at_sixty['hand_accuracy']))}였다. "
            f"손글씨 학습 비중은 {pct(hand_share(at_sixty))}에서 {pct(hand_share(last))}로 줄었다. "
            "시드 42에서 관측한 결과이며 공식 학습량과 손글씨 비중이 함께 변했다.")
    else:
        learning += paragraph(
            f"손글씨 최고 정확도는 공식 학습 {int(peak['official_train_n']):,}장에서 "
            f"{pct(float(peak['hand_accuracy']))}였다.")
    times = list(final["timing"].values())
    avg_fit = average(times, lambda r: r["fit_seconds"])
    avg_val = average(times, lambda r: r["val_predict_seconds"]*1000/19200)
    avg_hand = average(times, lambda r: r["hand_predict_seconds"]*1000/r["hand_eval_n"])
    learning += paragraph(f"시드 42 선정 모델의 평균 적합 시간은 모델당 {avg_fit:.2f}초였다. "
                          f"배치 추론은 공식 Validation 장당 {avg_val:.3f} ms, 손글씨 장당 {avg_hand:.3f} ms였다. "
                          "특징 추출과 입력 변환은 모델 추론 시간에서 제외했다. "
                          "Normal과 Combined가 같은 CPU에서 일부 병행 실행돼 시간은 해당 실행의 관측값이다.")
    learning += paragraph(f"Python {env['python']}, scikit-learn {env['packages']['scikit-learn']}, "
                          f"{env['cpu']}, 모델당 n_jobs={env['jobs']}. "
                          f"완료한 적합은 총 {count}회이며 반복과 학습곡선 적합을 포함한다.", "note")
    yield learning

    closing = title("통제 실험과 결론" if comb else "작성자별 진단과 결론",
                    "손글씨 추가 학습 효과" if comb else "최종 개발 후보")
    if comb:
        matched = read_csv(results / "matched_writers.csv")
        closing += figure("matched_writers", "Combined 선정 조건·시드 42를 고정했다. "
                          "공식 Train만 사용한 모델과 작성자 3명 손글씨를 추가한 Fold 모델이 같은 보류 작성자를 예측했다.")
        closing += table(("작성자", "공식만 학습", "손글씨 추가", "차이", "개선 / 악화 건수"),
                         [(row["writer"], pct(float(row["normal_accuracy"])), pct(float(row["combined_accuracy"])),
                           delta(float(row["delta_accuracy"])), f"{row['corrected']} / {row['regressed']}")
                          for row in matched])
        left = runs["CTRL_normal"]
        weakest = min(matched, key=lambda row: float(row["delta_accuracy"]))
        closing += paragraph(f"작성자 간 변화 폭이 달랐다. 변화 폭이 가장 낮은 {weakest['writer']}번은 "
                             f"{pct(float(weakest['normal_accuracy']))} → {pct(float(weakest['combined_accuracy']))}였다. "
                             f"D→0은 {left['hand']['focused_errors']['D->0']}→{final['hand']['focused_errors']['D->0']}건, "
                             f"B→8은 {left['hand']['focused_errors']['B->8']}→{final['hand']['focused_errors']['B->8']}건이었다.")
        closing += paragraph(f"통제 비교의 정확도 차이는 {delta(hand(final)-hand(left))}, 공식 검증 차이는 "
                             f"{delta(official_average(final)-official_average(left))}였다. "
                             "작성자 4명을 대상으로 한 관측 차이이며 모집단 효과의 신뢰구간으로 해석할 수 없다.")
    else:
        closing += table(("작성자", "기준 Acc.", "선정 Acc.", "선정 Macro F1"),
                         [(w, pct(baseline["writers"][str(w)]["accuracy"]),
                           pct(final["writers"][str(w)]["accuracy"]),
                           f"{final['writers'][str(w)]['macro_f1']:.4f}") for w in WRITERS])
        closing += paragraph(f"최종 손글씨 정확도는 기준 대비 {delta(hand(final)-hand(baseline))}였다. "
                             "세 시드 반복은 추정기 초기 난수에 대한 안정성을 확인했다. "
                             "각 시드에서 동일한 네 작성자를 평가했다.")
    closing += sub("선정 범위") + paragraph("후보 탐색에 같은 공식 Validation과 팀 손글씨를 반복 사용했다. "
                        "따라서 성능 수치는 개발 결과이며 독립 작성자나 최종 Test의 정확도는 추정하지 않았다. "
                        "시드 평균은 개별 모델 점수의 평균이다. 교수님 Test와 공식 Original Test 평가는 미실시다.")
    freeze = review / "freeze.json" if review else None
    if freeze and freeze.is_file():
        recipe = json.loads(freeze.read_text()).get(mode.lower())
        if recipe:
            closing += sub("독립 평가용 후보 구성")
            closing += paragraph(
                f"{recipe['candidate']} · 시드 {recipe['seed']} · 공식 Train "
                f"{recipe['official_training_n']:,}장 · 손글씨 {recipe['handmade_training_n']:,}장. "
                "Extra Trees 한 모델을 학습하고 추론에는 CoM 정렬과 픽셀·HOG를 적용한다. "
                "손글씨 개발 OOF는 후보 비교에 사용한 Fold별 점수다.")
    closing += sub("비교 근거") + paragraph("팀 저장소 README·RESEARCH_PLAN·W02 실험 2/3 및 02_data 검증 문서. "
                    "팀 선행 보고서 MLP_Normal(19p), MLP_Combined(18p), HGB_Normal(23p), HGB_Combined(15p). "
                    "실험별 원자료는 runs/summary.json 및 predictions.npz, 통합 결과는 results의 CSV를 참조한다.", "note")
    yield closing

    if review:
        yield paired_error_page(review, mode)
        if comb and (review / "paired_controls.csv").is_file():
            yield controls_page(review)


def paired_error_page(review, mode):
    rows = [r for r in read_csv(review / "seven_to_nine_by_writer.csv") if r["mode"] == mode]
    totals = {key: sum(int(r[key]) for r in rows) for key in rows[0] if key not in ("mode", "writer")}
    output = title("동일 샘플의 예측 대조", "7→9 오류와 작성자별 차이")
    output += table(("작성자", "7 정답 기존 → 선정", "7→9 기존 → 선정", "기존 정답 → 9", "기존 기타 오답 → 9"),
                    [(r["writer"], f"{r['correct_before']} → {r['correct_after']}",
                      f"{r['before']} → {r['after']}", r["introduced_from_correct"],
                      r["introduced_from_other_error"]) for r in rows])
    output += paragraph(
        f"전체 숫자 7 {totals['true_sevens']}장 중 정답은 {totals['correct_before']}→"
        f"{totals['correct_after']}건, 7→9 오답은 {totals['before']}→{totals['after']}건이었다. "
        f"새 7→9 오류에는 기존 정답 {totals['introduced_from_correct']}건과 "
        f"기존의 다른 오답 {totals['introduced_from_other_error']}건이 포함됐다. "
        f"기존 7→9 가운데 {totals['resolved_to_correct']}건은 정답으로 바뀌었다.")
    output += paragraph("44번에서 7→9 오류가 가장 크게 늘었다. 전체 숫자 7의 인식률은 개선됐고 "
                        "잔존 오류는 9에 더 집중됐다. 여러 전처리와 트리 설정이 함께 바뀐 기준선 비교이므로 "
                        "개별 설정이 7→9 오류에 기여한 크기는 분리해 추정하지 않았다.")
    if mode == "Combined":
        classes = read_csv(review / "writer41_by_class.csv")
        changed = [r for r in classes if r["errors_before"] != r["errors_after"]]
        output += sub("41번에서 손글씨 추가 학습의 영향")
        output += table(("정답", "오류 기존 → 추가", "오답 → 정답", "정답 → 오답"),
                        [(r["actual"], f"{r['errors_before']} → {r['errors_after']}", r["corrected"],
                          r["regressed"]) for r in changed])
        output += paragraph("같은 설정의 공식 자료 단독 모델 대비 15건이 개선되고 21건이 악화됐다. "
                            "순오류는 숫자 6에서 4건, 숫자 9에서 3건 증가했다. "
                            "6에서 새로 틀린 4건은 C 2건, B와 5 각 1건이었다. D의 오류는 3건 줄었다. "
                            "41번의 획 밀도가 낮았다는 입력 특성은 관찰값이며, "
                            "획 굵기와 정확도 하락의 인과관계를 이 비교로 추정하지 않았다.")
    gallery = review / f"seven_to_nine_{mode.lower()}.png"
    if gallery.is_file():
        output += figure(gallery.resolve().as_uri().removesuffix(".png"),
                         "모두 정답 7. 왼쪽 4장은 새 7→9 오류, 오른쪽 4장은 9에서 7로 교정된 예시다. "
                         "위 숫자는 작성자와 원본 행 번호, 아래는 예측의 변화다.")
        output += paragraph("윗획 왼쪽이 아래로 꺾인 7이 새 오류와 교정 사례 모두에 보였다. "
                            "같은 형태에서도 예측이 달라져 이 시각적 특징만으로 오류 증가를 설명하기 어렵다.",
                            "note")
    output += paragraph("예측이 바뀐 원본 행 번호는 paired_error_examples.csv에 기록했다.", "note")
    return output


def controls_page(review):
    rows = read_csv(review / "paired_controls.csv")
    output = title("후속 대조 실험 · 같은 작성자 분할과 시드", "획 혼합과 분기 후보 수")
    for arm, heading in (("stroke00", "픽셀·HOG에서 획 혼합 0.00과 0.28"),
                         ("features08", "분기 특징 비율 0.08과 0.18")):
        control = [r for r in rows if r["comparison"] == arm and r["arm"] == arm]
        reference = [r for r in rows if r["comparison"] == arm and r["arm"] == "T02_reference"]
        score_delta = lambda key: [float(b[key])-float(a[key]) for a, b in zip(control, reference)]
        output += sub(heading)
        output += table(("시드", "공식 Acc. 차이", "손글씨 Acc. 차이", "손글씨 Macro F1 차이"),
                        [(a["seed"], delta(float(b["official_accuracy"])-float(a["official_accuracy"])),
                          delta(float(b["hand_accuracy"])-float(a["hand_accuracy"])),
                          f"{float(b['hand_macro_f1'])-float(a['hand_macro_f1']):+.5f}")
                         for a, b in zip(control, reference)])
        output += paragraph(
            f"차이는 T02에서 대조군을 뺀 값이다. 3시드 평균 차이는 공식 "
            f"{delta(np.mean(score_delta('official_accuracy')))}, 손글씨 "
            f"{delta(np.mean(score_delta('hand_accuracy')))}, "
            f"Macro F1 {np.mean(score_delta('hand_macro_f1')):+.5f}였다.")
    output += paragraph(
        "획 혼합 대조는 크기 증강 30%와 픽셀·HOG를 고정했다. 0.28은 획 혼합의 강도다. "
        "픽셀 단독 단계의 대조는 트리 설정이 달라 HOG와 획의 상호작용 크기를 직접 추정할 수 없다.")
    old_fit = average(control, lambda row: float(row["fit_seconds_mean"]))
    reference_fit = average(reference, lambda row: float(row["fit_seconds_mean"]))
    output += paragraph(
        "1,108개 입력 특징 중 각 분기에서 고려하는 기본 특징 수는 0.08에서 88개, 0.18에서 199개다. "
        "후보 수를 늘리면 무작위 분기에서 더 많은 변수를 비교하므로 계산량이 커지고 트리 간 다양성에도 영향을 줄 수 있다. "
        f"3시드에서 모델당 평균 적합 시간은 각각 {old_fit:.1f}초, {reference_fit:.1f}초였다. "
        "실행 시점과 CPU 부하가 달라 시간은 관측값이다.")
    freeze = review / "freeze.json"
    if freeze.is_file():
        decision = json.loads(freeze.read_text())["combined"]["report_decision"]
        output += paragraph(decision, "note")
    return output


def render(source, results, output, mode, review=None):
    report_pages = list(pages(source, results, mode, review))
    section_styles = {4: "errors", 6: "closing", 7: "review-errors"}
    articles = "".join(f'<article class="{section_styles.get(i, "")}">{body}</article>'
                       for i, body in enumerate(report_pages))
    date = "2026-11-23" if review else "2026-11-22"
    label = f"EXTRA TREES / {mode.upper()}  |  {date}"
    html = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8">'
            f'<title>Extra Trees {mode} Optimization</title><style>{STYLE}</style></head>'
            f'<body><div class="report-label">{label}</div><main>{articles}</main></body></html>')
    document = HTML(string=html, base_url=str(results.resolve()) + "/").render()
    if len(document.pages) != len(report_pages):
        raise ValueError(f"Layout overflow: {len(document.pages)} pages for {len(report_pages)} sections")
    document.write_pdf(output)
    print(f"Rendered {len(document.pages)} pages: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("Normal", "Combined"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    args = parser.parse_args()
    render(args.source, args.results, args.output, args.mode, args.review_dir)
