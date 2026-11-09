"""
Digbodha 1 - pool the per-sequence KITTI evaluations into one set of numbers.

Medians and rates can't be averaged across sequences, so this re-computes every metric from the
matched.csv files written by eval/evaluate_kitti.py.

Usage (repo root):
  python eval/aggregate.py outputs/eval_0013 outputs/eval_0001 outputs/eval_0020 --out outputs/eval_pooled
"""
import argparse
import csv
import json
import os

import numpy as np

RISKS = ["safe", "uncertain", "caution", "danger"]
TTC_MAX = 10.0


def num(v):
    return float(v) if v not in ("", "None", None) else None


def summ(rows):
    if not rows:
        return {"n": 0}
    a = np.array([abs(p - g) for p, g in rows])
    r = np.array([abs(p - g) / g for p, g in rows])
    return {"n": len(rows), "median_abs_s": round(float(np.median(a)), 2),
            "median_rel_pct": round(float(np.median(r)) * 100, 1),
            "within_20pct": round(float((r <= 0.2).mean()) * 100, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+")
    ap.add_argument("--out", default="outputs/eval_pooled")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for d in args.eval_dirs:
        seq = os.path.basename(os.path.normpath(d)).replace("eval_", "")
        with open(os.path.join(d, "matched.csv")) as f:
            for r in csv.DictReader(f):
                r["seq"] = seq
                rows.append(r)

    res = {"sequences": [os.path.basename(os.path.normpath(d)) for d in args.eval_dirs],
           "matched_object_frames": len(rows), "per_sequence": {}}

    def metrics(sel):
        out = {}
        both = [(num(r["pred_ttc"]), num(r["gt_ttc"])) for r in sel
                if num(r["pred_ttc"]) is not None and num(r["gt_ttc"]) is not None and num(r["gt_ttc"]) < TTC_MAX]
        rel = [r for r in sel if r["reliable"] == "True"]
        both_rel = [(num(r["pred_ttc"]), num(r["gt_ttc"])) for r in rel
                    if num(r["pred_ttc"]) is not None and num(r["gt_ttc"]) is not None and num(r["gt_ttc"]) < TTC_MAX]
        out["ttc_all"] = summ(both)
        out["ttc_reliable"] = summ(both_rel)
        out["ttc_reliable_by_depth"] = {}
        for lo, hi in [(0, 10), (10, 20), (20, 40)]:
            out["ttc_reliable_by_depth"][f"{lo}-{hi} m"] = summ(
                [(num(r["pred_ttc"]), num(r["gt_ttc"])) for r in rel
                 if num(r["pred_ttc"]) is not None and num(r["gt_ttc"]) is not None
                 and num(r["gt_ttc"]) < TTC_MAX and lo <= float(r["gt_depth_m"]) < hi])
        gt_app = lambda r: num(r["gt_ttc"]) is not None and num(r["gt_ttc"]) < TTC_MAX
        pr_app = lambda r: num(r["pred_ttc"]) is not None
        tp = sum(gt_app(r) and pr_app(r) for r in sel)
        fp = sum(pr_app(r) and not gt_app(r) for r in sel)
        fn = sum(gt_app(r) and not pr_app(r) for r in sel)
        out["approach"] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / max(tp + fp, 1), 3),
                           "recall": round(tp / max(tp + fn, 1), 3)}
        conf = {g: {p: 0 for p in RISKS} for g in ["safe", "caution", "danger"]}
        for r in sel:
            conf[r["gt_risk"]][r["pred_risk"]] += 1
        alert = lambda x: x in ("caution", "danger")
        a_tp = sum(alert(r["pred_risk"]) and alert(r["gt_risk"]) for r in sel)
        a_fp = sum(alert(r["pred_risk"]) and not alert(r["gt_risk"]) for r in sel)
        a_fn = sum(alert(r["gt_risk"]) and not alert(r["pred_risk"]) for r in sel)
        unc = sum(alert(r["gt_risk"]) and r["pred_risk"] == "uncertain" for r in sel)
        silent = sum(alert(r["gt_risk"]) and r["pred_risk"] == "safe" for r in sel)
        out["risk_confusion_gt_rows_pred_cols"] = conf
        out["alerts"] = {"tp": a_tp, "fp": a_fp, "fn": a_fn, "fn_flagged_uncertain": unc,
                         "fn_called_safe": silent,
                         "precision": round(a_tp / max(a_tp + a_fp, 1), 3),
                         "recall": round(a_tp / max(a_tp + a_fn, 1), 3),
                         "recall_counting_uncertain": round((a_tp + unc) / max(a_tp + a_fn, 1), 3)}
        return out

    res["pooled"] = metrics(rows)
    for seq in sorted({r["seq"] for r in rows}):
        res["per_sequence"][seq] = metrics([r for r in rows if r["seq"] == seq])

    with open(os.path.join(args.out, "pooled_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)

    p = res["pooled"]
    print(f"\n=== Pooled over {', '.join(res['sequences'])}: {len(rows)} matched object-frames ===")
    print(f"TTC all       : {p['ttc_all']}")
    print(f"TTC reliable  : {p['ttc_reliable']}")
    for k, v in p["ttc_reliable_by_depth"].items():
        print(f"   {k:<8}  {v}")
    print(f"Approach      : {p['approach']}")
    print("Risk (rows = truth, cols = ours):")
    print(f"  {'':10}" + "".join(f"{r:>10}" for r in RISKS))
    for g, row in p["risk_confusion_gt_rows_pred_cols"].items():
        print(f"  {g:<10}" + "".join(f"{row[r]:>10}" for r in RISKS))
    print(f"Alerts        : {p['alerts']}")
    print("\nPer sequence (alerts precision / recall / recall incl. uncertain, TTC reliable median rel %):")
    for seq, m in res["per_sequence"].items():
        a = m["alerts"]
        print(f"  {seq:<6} P {a['precision']:<6} R {a['recall']:<6} R+unc {a['recall_counting_uncertain']:<6} "
              f"TTC {m['ttc_reliable'].get('median_rel_pct', '-')}%  (n={m['ttc_reliable']['n']})")
    print(f"\nwritten: {args.out}/pooled_metrics.json")


if __name__ == "__main__":
    main()