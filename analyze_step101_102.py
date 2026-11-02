import csv
import random
import math
from pathlib import Path
from statistics import mean

SOURCES = {
    "original": Path("outputs/step99_fairness/results_36.csv"),
    "holdout": Path("outputs/step100_holdout/results_36.csv"),
}

OUT = Path("outputs/step101_102_final")
OUT.mkdir(parents=True, exist_ok=True)

POLICIES = [
    "FCFS", "Basic", "BO",
    "FairBO_20s", "FairBO_40s"
]

METRICS = {
    "total_time_s": "Total Time",
    "avg_wait_s": "Avg Wait",
    "max_wait_s": "Max Wait",
    "p95_wait_s": "P95 Wait",
    "jain": "Jain",
    "over30": "AGVs Wait >=30s",
    "over50": "AGVs Wait >=50s",
}

records = {}
all_rows = []

for dataset, path in SOURCES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        scenario = int(r["scenario"])
        policy = r["policy"]
        key = (dataset, scenario, policy)

        if key in records:
            raise RuntimeError(f"Duplicate record: {key}")

        record = {
            "dataset": dataset,
            "scenario": scenario,
            "case": r["case"],
            "agvs": int(r["agvs"]),
            "policy": policy,
        }

        for metric in METRICS:
            record[metric] = float(r[metric])

        records[key] = record
        all_rows.append(record)

for dataset in SOURCES:
    for scenario in range(1, 37):
        for policy in POLICIES:
            key = (dataset, scenario, policy)
            if key not in records:
                raise RuntimeError(f"Missing record: {key}")

print("\n========== STEP 101: ALL 72 SCENARIOS ==========")

for policy in POLICIES:
    selected = [
        r for r in all_rows
        if r["policy"] == policy
    ]

    print(
        f"{policy:12s} | "
        f"Total={mean(r['total_time_s'] for r in selected):.2f}s | "
        f"AvgWait={mean(r['avg_wait_s'] for r in selected):.2f}s | "
        f"MaxWait={mean(r['max_wait_s'] for r in selected):.2f}s | "
        f"P95={mean(r['p95_wait_s'] for r in selected):.2f}s | "
        f"Jain={mean(r['jain'] for r in selected):.3f} | "
        f"Over30={int(sum(r['over30'] for r in selected))} | "
        f"Over50={int(sum(r['over50'] for r in selected))}"
    )

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    ratio = position - low
    return values[low] * (1 - ratio) + values[high] * ratio

def bootstrap_ci(differences, seed=20261101, repeats=5000):
    rng = random.Random(seed)
    n = len(differences)
    estimates = []

    for _ in range(repeats):
        sample = [
            differences[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(mean(sample))

    return (
        percentile(estimates, 0.025),
        percentile(estimates, 0.975),
    )

print("\n========== STEP 102: PAIRED BO VS FAIRBO ==========")
print("Difference = FairBO - BO")
print("Negative is better for time/wait; positive is better for Jain.")

comparison_rows = []

for target in ["FairBO_20s", "FairBO_40s"]:
    print(f"\n[{target}]")

    for dataset in ["original", "holdout", "combined"]:
        scenario_keys = [
            (d, s)
            for d in SOURCES
            for s in range(1, 37)
            if dataset == "combined" or d == dataset
        ]

        print(f"\n  Dataset: {dataset} / N={len(scenario_keys)}")

        for metric, label in METRICS.items():
            differences = [
                records[(d, s, target)][metric]
                - records[(d, s, "BO")][metric]
                for d, s in scenario_keys
            ]

            avg_diff = mean(differences)
            lower, upper = bootstrap_ci(
                differences,
                seed=20261101 + len(label) + len(dataset)
            )

            better = sum(
                x > 0 if metric == "jain" else x < 0
                for x in differences
            )
            equal = sum(abs(x) < 1e-9 for x in differences)
            worse = len(differences) - better - equal

            print(
                f"  {label:18s} "
                f"Diff={avg_diff:+8.3f} "
                f"95%CI=[{lower:+.3f}, {upper:+.3f}] "
                f"Better/Equal/Worse={better}/{equal}/{worse}"
            )

            comparison_rows.append({
                "policy": target,
                "dataset": dataset,
                "metric": metric,
                "n": len(differences),
                "mean_difference": round(avg_diff, 5),
                "ci95_lower": round(lower, 5),
                "ci95_upper": round(upper, 5),
                "better": better,
                "equal": equal,
                "worse": worse,
            })

def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

save_csv(OUT / "all_72_results.csv", all_rows)
save_csv(OUT / "paired_bootstrap_comparison.csv", comparison_rows)

print("\n========== SAVED ==========")
print(OUT / "all_72_results.csv")
print(OUT / "paired_bootstrap_comparison.csv")
