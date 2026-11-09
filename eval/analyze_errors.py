"""
Digbodha 1 - why are the risk labels wrong, and how good are the warnings per EVENT?

Reads the matched.csv files from eval/evaluate_kitti.py.

1. Error attribution (frame level). For every object-frame where our risk disagrees with the truth
   on "alert or not" (alert = caution/danger), find the cause by swapping in true values:
     no_approach   : we saw no approach at all (no TTC)
     abstained     : we said "uncertain" (the reliability reasons are counted)
     ttc           : using the TRUE TTC with OUR miss distance gives the right answer
     on_course     : using OUR TTC with the TRUE miss distance gives the right answer
     both          : only fixing both gives the right answer
     other         : e.g. true value missing (object receding in truth)

2. Event-level warnings. A hazard EVENT = a run of consecutive frames in which one true object is
   at caution/danger. It counts as warned if we raised caution/danger for it at any point from
   `--early` frames before the run starts to the end of the run; lead time = time from our first
   warning to the event's last frame. A false-alarm EVENT = a run of our caution/danger frames on
   an object with no true caution/danger within +-`--early` frames.

Usage (repo root):
  python eval/analyze_errors.py outputs/eval_0013 outputs/eval_0001 outputs/eval_0020
"""
import argparse
import csv
import os
from collections import Counter, defaultdict

import numpy as np

TTC_MAX, MISS, DANGER, CAUTION, NEAR = 10.0, 1.0, 2.0, 4.0, 1.5


def num(v):
    return float(v) if v not in ("", "None", None) else None


def risk(ttc, miss):
    """Same rule as src/ttc.py (without the reliability gate)."""
    if ttc is None or ttc >= TTC_MAX:
        return "safe"
    on = miss is not None and abs(miss) < MISS
    if on and ttc < DANGER:
        return "danger"
    if (on and ttc < CAUTION) or ttc < NEAR:
        return "caution"
    return "safe"


alert = lambda r: r in ("caution", "danger")


def runs(frames):
    """Sorted frame numbers -> list of (start, end) runs of consecutive frames."""
    out = []
    for f in sorted(frames):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return [tuple(r) for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dirs", nargs="+")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--early", type=int, default=5, help="frames of tolerance before/after an event")
    args = ap.parse_args()

    rows = []
    for d in args.eval_dirs:
        seq = os.path.basename(os.path.normpath(d)).replace("eval_", "")
        with open(os.path.join(d, "matched.csv")) as f:
            for r in csv.DictReader(f):
                r["seq"] = seq
                rows.append(r)

    # ---------------------------------------------------------------- 1. attribution
    causes = {"missed": Counter(), "false": Counter()}
    abstain_reasons = Counter()
    for r in rows:
        g, p = r["gt_risk"], r["pred_risk"]
        if alert(g) == alert(p):
            continue
        kind = "missed" if alert(g) else "false"
        pt, pm, gt, gm = num(r["pred_ttc"]), num(r["pred_miss"]), num(r["gt_ttc"]), num(r["gt_miss"])
        if p == "uncertain":
            causes[kind]["abstained"] += 1
            for reason in (r["reasons"] or "").split(";"):
                if reason:
                    abstain_reasons[reason] += 1
        elif pt is None:
            causes[kind]["no_approach"] += 1
        elif gt is None:
            causes[kind]["other"] += 1
        elif alert(risk(gt, pm)) == alert(g):
            causes[kind]["ttc"] += 1
        elif alert(risk(pt, gm)) == alert(g):
            causes[kind]["on_course"] += 1
        elif alert(risk(gt, gm)) == alert(g):
            causes[kind]["both"] += 1
        else:
            causes[kind]["other"] += 1

    print(f"\n=== Error attribution over {len(rows)} matched object-frames ({', '.join(sorted({r['seq'] for r in rows}))}) ===")
    for kind, title in (("missed", "MISSED warnings (truth alert, ours not)"),
                        ("false", "FALSE warnings (ours alert, truth not)")):
        tot = sum(causes[kind].values())
        print(f"\n{title}: {tot}")
        for c, n in causes[kind].most_common():
            print(f"  {c:<12} {n:>6}  ({100 * n / max(tot, 1):.0f}%)")
    print(f"\nReasons behind 'uncertain' on true hazards: {dict(abstain_reasons.most_common())}")

    # ---------------------------------------------------------------- 2. events
    by_track = defaultdict(list)
    for r in rows:
        by_track[(r["seq"], r["gt_track"])].append(r)
    n_events, warned, uncertain_only, leads = 0, 0, 0, []
    fa_events = 0
    for key, rs in by_track.items():
        gt_alert = [int(r["frame"]) for r in rs if alert(r["gt_risk"])]
        pr_alert = [int(r["frame"]) for r in rs if alert(r["pred_risk"])]
        pr_unc = [int(r["frame"]) for r in rs if r["pred_risk"] == "uncertain"]
        for s, e in runs(gt_alert):
            n_events += 1
            hits = [f for f in pr_alert if s - args.early <= f <= e]
            if hits:
                warned += 1
                leads.append((e - min(hits)) / args.fps)
            elif any(s - args.early <= f <= e for f in pr_unc):
                uncertain_only += 1
        for s, e in runs(pr_alert):
            if not any(s - args.early <= f <= e + args.early for f in gt_alert):
                fa_events += 1

    print(f"\n=== Event-level warnings (tolerance {args.early} frames = {args.early / args.fps:.1f} s) ===")
    print(f"  hazard events (true caution/danger runs): {n_events}")
    print(f"  warned (caution/danger raised)          : {warned}  ({100 * warned / max(n_events, 1):.0f}%)")
    print(f"  only flagged 'uncertain'                : {uncertain_only}  ({100 * uncertain_only / max(n_events, 1):.0f}%)")
    print(f"  missed silently                         : {n_events - warned - uncertain_only}")
    if leads:
        print(f"  warning lead time (first warning -> end of event): median {np.median(leads):.1f} s, "
              f"25-75% {np.percentile(leads, 25):.1f}-{np.percentile(leads, 75):.1f} s")
    print(f"  false-alarm events (no true hazard within tolerance): {fa_events}")


if __name__ == "__main__":
    main()