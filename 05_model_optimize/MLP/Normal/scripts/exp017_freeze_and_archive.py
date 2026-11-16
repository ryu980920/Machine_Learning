#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Archive EXP017 OfficialVal-only results and freeze winning 3-seed joblibs.

Do not train or load team handwriting. This is a copy-and-SHA-verification step.
No main/final model mutation. The workflow controls git commit target branch.
"""
from __future__ import annotations
import argparse,json,hashlib,shutil
from pathlib import Path
from exp017_frozen_officialval_five_scale_audit import path_for

def copy_tree(src,dst):
    src=Path(src);dst=Path(dst)
    assert src.is_dir() and not dst.exists(),(src,dst)
    dst.mkdir(parents=True)
    for f in src.iterdir():
        assert f.is_file(),f
        shutil.copy2(f,dst/f.name)

def run(coarse_review,final_review,audit_review,coarse_models,fine_models,old_models,archive_root,model_root):
    archive_root=Path(archive_root);model_root=Path(model_root)
    assert not archive_root.exists() and not model_root.exists()
    archive_root.mkdir(parents=True);model_root.mkdir(parents=True)
    copy_tree(coarse_review,archive_root/"coarse_26config_3seed")
    copy_tree(final_review,archive_root/"joint_final_46config_3seed")
    copy_tree(audit_review,archive_root/"frozen_officialval_five_scales")
    pick=json.loads((Path(audit_review)/"SELECTION.json").read_text(encoding="utf-8"))
    selected=pick.get("new_candidate")
    if selected is None:
        (model_root/"README.md").write_text("# EXP유017 동결 후보 없음\n\n안전 기준을 통과한 후보가 없어 새 모델을 선택하지 않았다. 팀 손글씨는 평가하지 않았다.\n",encoding="utf-8")
        print("EXP017_NO_NEW_FROZEN_CANDIDATE",flush=True);return
    joint=json.loads((Path(final_review)/"metadata.json").read_text(encoding="utf-8"))
    chosen=next((v for v in (joint["selected_frozen_candidate"],joint["runner_up"])
                 if v and v["variant"]==selected),None)
    assert chosen is not None
    expected={(r["label"],int(r["seed"])):r for r in pick["checkpoints"]}
    archive=[]
    for seed in (42,43,44):
        old=path_for(chosen,seed,coarse_models,fine_models,old_models)
        assert old.is_file()
        digest=hashlib.sha256(old.read_bytes()).hexdigest()
        assert expected[pick["new_candidate_label"],seed]["sha256"]==digest,(seed,digest)
        dst=model_root/f"seed{seed}.joblib"
        shutil.copy2(old,dst)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
        archive.append({"seed":seed,"file":dst.name,"sha256":digest,
               "bytes":dst.stat().st_size,"source_variant":selected,
               "source_study":chosen["source"],"source_action_artifact":str(old),
               "team_handwriting_used":False})
    data={"experiment":"EXP017","stage":"frozen_hyperparameters_before_team_validation",
        "chosen_variant":selected,"source":chosen["source"],
        "hidden":chosen["hidden"],"alpha":chosen["alpha"],
        "learning_rate_init":chosen["learning_rate_init"],
        "early_stopping":chosen["early_stopping"],
        "preprocessing":"80pct_bbox_size_14pct_stroke_training__com_inference",
        "official_source_sha256":joint["official_val_source_sha256"],
        "checkpoints":archive,
        "official_val_only_hyperparam_selection":True,
        "team_handwriting_evaluated":False,
        "original_test_used":False,"professor_test_used":False,
        "final_deployment_performed":False}
    (model_root/"MANIFEST.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (model_root/"README.md").write_text(
        "# EXP유017 — 80-14 고정 MLP 하이퍼파라미터 최적화 동결 후보\n\n"+
        f"- 선택된 모델: **{selected}**\n"+
        f"- 은닉층: **{chosen['hidden']}**, alpha={chosen['alpha']}, learning_rate_init={chosen['learning_rate_init']}, early_stopping={chosen['early_stopping']}\n"+
        "- 각 Seed (42·43·44)의 학습된 joblib 3개를 원본 Actions SHA-256과 동일하게 보존\n"+
        "- 기존 70-0 / 80-14 동결 체크포인트는 그대로 유지, main 및 final 체크포인트 변경 없음\n"+
        "- **팀 손글씨 Validation 미수행**. 이 모델은 독립 손글씨 평가를 위한 가중치 동결 후보이며 최종 배포 모델 아님\n"+
        "- [실측 46조합 및 5배율 결과](../크기편향_원인분석/EXP유017_하이퍼파라미터_실측/)\n",
        encoding="utf-8")
    print("EXP017_THREE_SEED_FREEZE_COMPLETE",selected,flush=True)
if __name__=="__main__":
    a=argparse.ArgumentParser()
    for nm in ("coarse-review","final-review","audit-review","coarse-models","fine-models","old-models","archive-root","model-root"):
        a.add_argument("--"+nm,type=Path,required=True)
    x=a.parse_args()
    run(x.coarse_review,x.final_review,x.audit_review,x.coarse_models,x.fine_models,
        x.old_models,x.archive_root,x.model_root)
