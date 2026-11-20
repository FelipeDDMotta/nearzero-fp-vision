"""Turn agent decisions (+ human review labels) into training data.

Two exports from the same decisions:
  * classifier/   crops sorted into  true/<class>/  and  false_positive/  folders
                  (hard negatives are exactly the false positives the agent rejected)
  * vlm_chat.jsonl  prompt -> verdict JSON pairs for distilling the verifier into an open VLM
Only auto-resolved decisions are used; escalated cases wait for a human label.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

LABEL_SOURCES = ("human", "agent")


def decisions(results: dict[str, dict], human: dict | None = None):
    human = human or {}
    for clip_id, res in results.items():
        for t in res["tracks"]:
            key = f"{clip_id}:{t['id']}"
            if key in human:
                yield clip_id, res, t, human[key], "human"
                continue
            tri = t.get("triage")
            if tri and tri["status"] in ("confirmed", "rejected"):
                yield clip_id, res, t, tri["final"], "agent"


def build_zip(zf: zipfile.ZipFile, results: dict[str, dict], public: Path) -> None:
    human_path = public / "results" / "human_labels.json"
    human = json.loads(human_path.read_text()) if human_path.exists() else {}
    manifest = []
    for clip_id, res, t, label, source in decisions(results, human):
        crop = public / t["crops"]["best"]
        if not crop.exists():
            continue
        if label["verdict"] == "false_positive":
            arc = f"classifier/false_positive/{clip_id}_{t['id']}.jpg"
        else:
            arc = f"classifier/true/{label.get('corrected_class') or t['cls']}/{clip_id}_{t['id']}.jpg"
        zf.write(crop, arc)
        manifest.append({"clip": clip_id, "track": t["id"], "detector_cls": t["cls"],
                         "score": t["score"], "verdict": label["verdict"],
                         "corrected_class": label.get("corrected_class"), "label_source": source,
                         "image": arc})
    zf.writestr("manifest.jsonl", "\n".join(json.dumps(m) for m in manifest))
    zf.writestr("README.txt", __doc__ or "")
