import csv
import random
from pathlib import Path
from statistics import mean

SOURCES = {
    "original": Path("outputs/step103_pso_original/results_36.csv"),
    "holdout": Path("outputs/step103_pso_holdout/results_36.csv"),
}

OUT = Path("outputs/step104_pso_statistics")
OUT.mkdir(parents=True, exist_ok=True)

METRICS = {
    "total_time_s": "Total Time",
    "avg_wait_s": "Avg Wait",
    "max_wait_s": "Max Wait",
    "p95_wait_s": "P95 Wait",
    "jain": "Jain",
    "over30": "Wait >=30s",
    "over50": "Wait >=50s",
}

def percentile(values, q):
    a = sorted(values)
    pos = (len(a) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(a) - 1)
    frac = pos - lo
    return a[lo] * (1 - frac) + a[hi] * frac

def bootstrap_ci(diffs, seed, repeats=5000):
    rng = random.Random(seed)
    n = len(diffs)
    samples = [
        mean(diffs[rng.randrange(n)] for _ in range(n))
        for _ in range(repeats)
    ]
    return percentile(samples, 0.025), percentile(samples, 0.975)

data = {}

for dataset, path in SOURCES.items():
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        key = (dataset, int(r["scenario"]), r["policy"])
        if key in data:
            raise RuntimeError(f"Duplicate row: {key}")
        data[key] = r

targets = ["FairPSO_20s", "FairPSO_40s"]
results = []

for target in targets:
    print(f"\n========== PSO VS {target} ==========")

    for dataset in ["original", "holdout", "combined"]:
        ids = [
            (d, s)
            for d in SOURCES
            for s in range(1, 37)
            if dataset == "combined" or d == dataset
        ]

        print(f"\n[{dataset}] N={len(ids)}")

        for index, (metric, label) in enumerate(METRICS.items()):
            diffs = []

            for d, s in ids:
                old = data[(d, s, "PSO")]
                new = data[(d, s, target)]

                diffs.append(
                    float(new[metric]) - float(old[metric])
                )

            avg = mean(diffs)
            lower, upper = bootstrap_ci(
                diffs,
                seed=20261101 + index
                + (0 if dataset == "original" else
                   100 if dataset == "holdout" else 200)
            )

            eps = 1e-9
            better = sum(
                x > eps if metric == "jain" else x < -eps
                for x in diffs
            )
            equal = sum(abs(x) <= eps for x in diffs)
            worse = len(diffs) - better - equal

            print(
                f"{label:15s} "
                f"Diff={avg:+8.3f} "
                f"95%CI=[{lower:+.3f}, {upper:+.3f}] "
                f"Better/Equal/Worse={better}/{equal}/{worse}"
            )

            results.append({
                "policy": target,
                "dataset": dataset,
                "metric": metric,
                "n": len(ids),
                "mean_difference": round(avg, 5),
                "ci95_lower": round(lower, 5),
                "ci95_upper": round(upper, 5),
                "better": better,
                "equal": equal,
                "worse": worse,
            })

path = OUT / "paired_pso_fairpso.csv"

with path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0])
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", path)
