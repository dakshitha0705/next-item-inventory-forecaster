"""Milestone 5: top-N accuracy of the model against a best-seller (frequency) baseline.

    python scripts/evaluate.py --cases 300            # writes results/eval.md and results/eval.json
    python scripts/evaluate.py --cases 5 --no-write   # time a few cases first

The cases are the drill's: held-out baskets with one item hidden (`thilimarketbasket.evaluate.make_cases`, seed 0).
Each method sees the rest of the basket and suggests 10 items it does not already hold:
- model: `predict_next` exactly as the app runs it (20 samples, short lists filled from the baseline);
- model only: the same list cut at the first baseline fill, so only the model's own votes count;
- frequency baseline: the shop's best sellers;
- item-to-item counting: items that most often shared a basket with the visible ones (the drill's `ItemToItem`).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from thilimarketbasket.evaluate import make_cases
from thilimarketbasket.recommend import ItemToItem, top_k

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import core  # noqa: E402

KS = (1, 5, 10)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--model", default=core.model_dir().as_posix())
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--note", default="", help="a sentence for the report, e.g. why this many cases")
    args = parser.parse_args()

    names = core.catalog_names()
    train, val = core.split(core.make_baskets(100_000, seed=0))
    baseline = core.frequency_baseline(train)
    counting = ItemToItem(train, names)
    cases = make_cases(val, names, n_cases=args.cases, seed=0)
    sampler = core.model_sampler(*core.load_model(args.model))

    ranked = {"model": [], "model only": [], "frequency baseline": [], "item-to-item counting": []}
    raw, started = [], time.time()
    for k, case in enumerate(cases):
        visible = case["visible"]
        preds = core.predict_next(visible, catalog=names, top_n=10, n_samples=args.samples, seed=k,
                                  baseline=baseline, sampler=sampler)
        ranked["model"].append([p.item for p in preds])
        ranked["model only"].append([p.item for p in preds if p.source == "model"])
        ranked["frequency baseline"].append([p.item for p in core.baseline_predictions(visible, baseline, 10)])
        if visible:
            held = [counting.position[x] for x in visible]
            ranked["item-to-item counting"].append([names[i] for i in top_k(counting.score(visible), 10, held)])
        else:   # nothing to count from: fall back to best sellers
            ranked["item-to-item counting"].append(ranked["frequency baseline"][-1])
        raw.append({"case": k, "visible": visible, "target": case["target"],
                    "model": [(p.item, p.confidence, p.source) for p in preds]})
        if (k + 1) % 25 == 0 or k + 1 == len(cases):
            print(f"{k + 1}/{len(cases)} cases, {time.time() - started:.0f} s", flush=True)
    seconds = time.time() - started

    targets = [c["target"] for c in cases]
    table = {method: {f"hits@{n}": core.top_n_accuracy(lists, targets, n) for n in KS}
             for method, lists in ranked.items()}
    differences = {}
    for other in ("frequency baseline", "item-to-item counting"):
        for n in KS:
            mean, low, high = core.bootstrap_difference(core.hits(ranked["model"], targets, n),
                                                        core.hits(ranked[other], targets, n))
            differences[f"model - {other}, hits@{n}"] = {"mean": mean, "low": low, "high": high}
    result = {"cases": len(cases), "note": args.note, "samples_per_prediction": args.samples, "model_dir": args.model,
              "seconds": round(seconds, 1), "seconds_per_case": round(seconds / len(cases), 2),
              "model_share_of_suggestions": float(np.mean([len(m) for m in ranked["model only"]]) / 10),
              "hits": table, "differences_95pct_bootstrap": differences}
    print(json.dumps(result, indent=1))
    if args.no_write:
        return
    out = Path("results")
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "raw" / "eval-cases.jsonl").write_text("\n".join(json.dumps(r) for r in raw) + "\n")
    (out / "eval.md").write_text(markdown(result))
    print("wrote results/eval.md and results/eval.json")


def markdown(r) -> str:
    lines = [f"# Evaluation: next-item top-N accuracy ({r['cases']} held-out cases)", "",
             f"Each case is a held-out basket with one item hidden; a method gets a hit at N if the hidden item is in "
             f"its first N suggestions. Model: `{r['model_dir']}`, {r['samples_per_prediction']} samples per "
             f"prediction, {r['seconds_per_case']} s per case on a laptop CPU "
             f"({r['seconds'] / 60:.1f} min in all).", "",
             *([f"**Note:** {r['note']}", ""] if r.get("note") else []),
             "| method | hits@1 | hits@5 | hits@10 |", "|---|---|---|---|"]
    for method, h in r["hits"].items():
        lines.append(f"| {method} | " + " | ".join(f"{h[f'hits@{n}']:.3f}" for n in KS) + " |")
    lines += ["", f"On average {r['model_share_of_suggestions']:.0%} of the model's 10 suggestions come from the "
              "model itself; the rest are filled from the baseline.", "",
              "Paired differences with a bootstrap 95% range (resampling cases):", "",
              "| comparison | difference | 95% range |", "|---|---|---|"]
    for name, d in r["differences_95pct_bootstrap"].items():
        lines.append(f"| {name} | {d['mean']:+.3f} | {d['low']:+.3f} to {d['high']:+.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
