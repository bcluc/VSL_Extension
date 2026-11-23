"""Score headless live-demo runs on the held-out conversation videos.

For each video, compares the demo's events with the scripted conversation:
* sign segmentation: number of segments the demo classified vs. signs actually performed;
* word error rate over the whole conversation (variants collapsed, sclite weights);
* sentences: how many scripted English sentences the demo produced exactly, in order.

    python -X utf8 scripts/score_demo_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from signbridge.metrics import EditCounts, edit_counts  # noqa: E402
from signbridge.signs.phrases import base_gloss, compose  # noqa: E402

DATA = REPO.parent / "datasets" / "asl_citizen"


def lcs(a: list[str], b: list[str]) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--logs", type=Path, help="log folder (default outputs/signs/demo_<split>)")
    args = ap.parse_args()
    gt = json.loads((DATA / f"demo_{args.split}" / "ground_truth.json").read_text(encoding="utf-8"))
    LOGS = args.logs or REPO / "outputs" / "signs" / f"demo_{args.split}"
    total = EditCounts()
    rows = []
    for g in gt:
        log_path = LOGS / f"{Path(g['video']).stem}.json"
        if not log_path.exists():
            print(f"missing log for {g['video']}")
            continue
        log = json.loads(log_path.read_text(encoding="utf-8"))
        ref_words = [base_gloss(x) for s in g["sentences"] for x in s]
        segs = [e for e in log["events"] if e["type"] == "sign"]
        hyp_words = [base_gloss(e["gloss"]) for e in segs if e["accepted"]]
        c = edit_counts(ref_words, hyp_words)
        total = total + c
        ref_sent = [compose(s).english for s in g["sentences"]]
        hyp_sent = [e["en"] for e in log["events"] if e["type"] == "sentence"]
        rows.append({"signer": g["signer"], "signs_performed": len(ref_words), "segments": len(segs),
                     "accepted": len(hyp_words), "WER": round(100 * c.wer, 1),
                     "S/D/I": f"{c.substitutions}/{c.deletions}/{c.insertions}",
                     "sentences_ref": len(ref_sent), "sentences_out": len(hyp_sent),
                     "sentences_exact": lcs(ref_sent, hyp_sent), "ms_per_frame": round(log["mean_ms"], 1)})
    for r in rows:
        print(r)
    n_ref = sum(r["sentences_ref"] for r in rows)
    summary = {"videos": len(rows), "signs": total.reference_length,
               "WER": round(100 * total.wer, 1),
               "S/D/I": [total.substitutions, total.deletions, total.insertions],
               "sentences_exact": f"{sum(r['sentences_exact'] for r in rows)}/{n_ref}",
               "segments_vs_signs": f"{sum(r['segments'] for r in rows)}/{sum(r['signs_performed'] for r in rows)}"}
    print(json.dumps(summary, indent=1))
    (LOGS / "score.json").write_text(json.dumps({"summary": summary, "videos": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
