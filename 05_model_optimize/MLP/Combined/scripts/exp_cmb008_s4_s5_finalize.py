#!/usr/bin/env python3
"""Finalize CMB-008 S1-S5 only from verified stage results; no model deployment."""
import argparse,json
from pathlib import Path

def pct(x):return f"{x*100:.3f}%"
def with_banner(s,body):
    a="<!-- CMB008_S4S5_FINAL_START -->"
    b="<!-- CMB008_S4S5_FINAL_END -->"
    if a in s and b in s:return s[:s.index(a)]+a+"\n"+body+"\n"+s[s.index(b):]
    return s.rstrip()+"\n\n"+a+"\n"+body+"\n"+b+"\n"
def table(stage):
    rows=["| 조건 | OOF 정확도 | OfficialVal | D→0 | B→8 | 8→B | 0→D | 4방향 <5% | 악화 Gate |",
          "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for key,score in stage["scores"].items():
        vals=[f"{score['pairs'][k]} ({pct(score['pairs'][k]/480)})" for k in ("D->0","B->8","8->B","0->D")]
        gaps=[k for k,v in stage["gates"][key].items() if not v]
        rows.append("| "+str(key)+" | "+pct(score["oof"])+" | "+pct(score["val"])+
            " | "+" | ".join(vals)+" | "+
            ("통과" if all(score["pairs"][k]<24 for k in ("D->0","B->8","8->B","0->D")) else "미달")+
            " | "+(", ".join(gaps) if gaps else "없음")+" |")
    return rows
def main():
    p=argparse.ArgumentParser()
    for x in ("s4","s5","logs","hp-index","combined-readme","master-log-index"):
        p.add_argument("--"+x,type=Path,required=True)
    a=p.parse_args()
    s4=json.loads(a.s4.read_text(encoding="utf-8"))
    s5=json.loads(a.s5.read_text(encoding="utf-8"))
    assert s4["stage"]=="S4_learning_rate" and s5["stage"]=="S5_early_stopping"
    lr=s4["selected_learning_rate"]
    es=s5["selected_early_stopping"]
    assert lr==s5["selected_learning_rate"] and str(es)==s5["selected"]
    assert s4["no_independent_test_used"] and s5["no_independent_test_used"]
    assert s4["same_official_and_handmade_sha_confirmed"] and s5["same_official_and_handmade_sha_confirmed"]
    assert all(s4["gates"][lr].values()) and all(s5["gates"][str(es)].values())
    score=s5["scores"][str(es)]
    lines=["# EXP-CMB-008 · 11 S1~S5 최종 HP 연구 결론","",
      "> **개발용 OOF/OfficialVal 기준 잠정 선택. Original Test·교수님 Test·신규 손글씨 480장 미사용. 최종 배포 체크포인트 미변경.**","",
      "## 단계별 가설 → 실측 → 결정 → 다음 질문","",
      "| 단계 | 비교와 선정 이유 | 결정 | 다음으로 넘어간 이유 |",
      "|---|---|---|---|",
      "| S1 너비 | (64,)/(100,)/(128,)/(256,)을 비교 | (256,)의 OOF 92.214%와 주요 오류 개선 | 표현 용량을 확인한 후 깊이 검증 |",
      "| S2 깊이 | (256,128)/(256,256)으로 모델 복잡도 증가 | 두 층에서 8→B 등의 오류 악화 → 단층 (256,) 유지 | 구조를 동결하고 L2 강도 확인 |",
      "| S3 L2 | alpha 1e-5/1e-4/1e-3/1e-2 | 강한 규제에서 D→0 악화 → alpha=1e-4 유지 | L2를 고정하고 최적화 속도 효과 분리 |",
      f"| S4 LR | 0.0005 / 0.001 / 0.002 | **{lr} 선택** ({s4['status']}) | LR 고정 후 Early Stopping 효과 검증 |",
      f"| S5 종료 | Early Stopping False vs True (Train 내부 10%) | **{es} 선택** ({s5['status']}) | HP 잠정 동결 |","",
      "## S4 Learning Rate 실측·선정",*table(s4),"",
      "## S5 Early Stopping 실측·선정",*table(s5),"",
      "## 현재 동결할 학습 조건","",
      "| 파라미터 | 선정값 |",
      "|---|---|",
      "| Train augmentation | Size 30% × Stroke 강도 28% (Official Train 일부 1:1 대체) |",
      "| Center | CoM |","| Hidden layers | (256,) |",
      "| Solver/activation | Adam / ReLU |",
      "| alpha | 0.0001 |",
      f"| learning_rate_init | **{lr}** |",
      f"| early_stopping | **{es}** |",
      "| max_iter | 200 |",
      f"| OOF Accuracy | **{pct(score['oof'])}** |",
      f"| OfficialVal Accuracy | **{pct(score['val'])}** |",
      f"| D→0 | {score['pairs']['D->0']}/480 = {pct(score['pairs']['D->0']/480)} |",
      f"| B→8 | {score['pairs']['B->8']}/480 = {pct(score['pairs']['B->8']/480)} |",
      f"| 8→B | {score['pairs']['8->B']}/480 = {pct(score['pairs']['8->B']/480)} |",
      f"| 0→D | {score['pairs']['0->D']}/480 = {pct(score['pairs']['0->D']/480)} |",""]
    target=all(score["pairs"][k]<24 for k in ("D->0","B->8","8->B","0->D"))
    lines+=[("**네 방향 각 5% 미만의 목표 달성**" if target else
        "**네 방향 각 5% 미만의 목표 미달.** 오분류 비증가와 작성자/Val/Flip 안전 Gate를 통과한 최선의 개발용 대안을 선택했다."),
      "**과적합 한계:** 보류 Writer 4명을 반복 탐색에 사용해 OOF 자체의 선택 편향 가능성은 남는다. ES 내부 10%의 검증은 Training Fold만 사용하며 독립 Test를 대체하지 않는다.","",
      "## 추적 가능한 원본","",
      "- [S1 너비](EXP-CMB-008_02_Stage1_Width_Results.md)",
      "- [S2 깊이](EXP-CMB-008_04_Stage2_Depth_Results.md)",
      "- [S3 L2](EXP-CMB-008_06_Stage3_Alpha_Results.md)",
      "- [S4 사전등록](EXP-CMB-008_07_Plan_S4_LearningRate.md) → [S4 실측](EXP-CMB-008_08_Stage4_LearningRate_Results.md)",
      "- [S5 사전등록](EXP-CMB-008_09_Plan_S5_EarlyStopping.md) → [S5 실측](EXP-CMB-008_10_Stage5_EarlyStopping_Results.md)",
      "- 원본 실측 CSV/JSON: ../../results/Hyperparameter_Optimization/S4_LearningRate/ 및 S5_EarlyStopping/",""]
    a.logs.mkdir(parents=True,exist_ok=True)
    (a.logs/"EXP-CMB-008_11_Final_HP_Decision.md").write_text("\n".join(lines),encoding="utf-8")
    for name,file in (("hp",a.hp_index),("combined",a.combined_readme),("master",a.master_log_index)):
        if name=="hp":
            body=f"**S1~S5 완료:** 단층 (256,), alpha=1e-4, LR **{lr}**, Early Stopping **{es}**, OOF **{pct(score['oof'])}**, OfficialVal **{pct(score['val'])}**. [S4 결과](EXP-CMB-008_08_Stage4_LearningRate_Results.md) → [S5 결과](EXP-CMB-008_10_Stage5_EarlyStopping_Results.md) → [최종 연구결정](EXP-CMB-008_11_Final_HP_Decision.md). 최종 Test·배포 미변경."
        elif name=="combined":
            body=f"**CMB-008 S1~S5 검증 완료:** (256,), alpha=1e-4, LR **{lr}**, Early Stopping **{es}**, OOF **{pct(score['oof'])}**, OfficialVal **{pct(score['val'])}**. [선정·탈락 근거](logs/하이퍼파라미터_최적화/EXP-CMB-008_11_Final_HP_Decision.md). 최종 Test·배포 미변경."
        else:
            body=f"**CMB-008 S1~S5 완료:** (256,), alpha=1e-4, LR **{lr}**, Early Stopping **{es}**, OOF **{pct(score['oof'])}**, OfficialVal **{pct(score['val'])}**. [S4 결과](하이퍼파라미터_최적화/EXP-CMB-008_08_Stage4_LearningRate_Results.md) → [S5 결과](하이퍼파라미터_최적화/EXP-CMB-008_10_Stage5_EarlyStopping_Results.md) → [최종 결정](하이퍼파라미터_최적화/EXP-CMB-008_11_Final_HP_Decision.md). 최종 Test·배포 미변경."
        file.write_text(with_banner(file.read_text(encoding="utf-8"),body),encoding="utf-8")
    print("FINAL_DECISION",json.dumps({"learning_rate":lr,"early_stopping":es,"oof":score["oof"],"official_val":score["val"],"four_directions_below_5pct":target},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
