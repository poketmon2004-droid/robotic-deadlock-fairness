import ast
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig
from scripts.run_total_time_bo_policy_comparison import FIXED_HEURISTIC_WEIGHTS

OUT = Path("outputs/step103_pso_original")
OUT.mkdir(parents=True, exist_ok=True)

def extract_class(filename, name, namespace):
    tree = ast.parse(
        Path(filename).read_text(encoding="utf-8-sig")
    )
    node = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            filename,
            "exec"
        ),
        namespace
    )
    return namespace[name]

TurnModel = extract_class(
    "analyze_step78.py",
    "TurnModel",
    {
        "math": math,
        "RealisticCrossExperiment": RealisticCrossExperiment
    }
)

IndividualSpeedModel = extract_class(
    "analyze_step93.py",
    "IndividualSpeedModel",
    {"math": math, "TurnModel": TurnModel}
)

FixedQueueModel = extract_class(
    "analyze_step96_queue_fix.py",
    "FixedQueueModel",
    {"IndividualSpeedModel": IndividualSpeedModel}
)

class FairBOModel(FixedQueueModel):
    def __init__(self, *args, wait_threshold_s=20.0, **kwargs):
        self.wait_threshold_s = wait_threshold_s
        super().__init__(*args, **kwargs)

    def _admission_score(self, robot, candidates):
        original_score = super()._admission_score(robot, candidates)

        if self.policy_type != "heuristic":
            return original_score

        if robot.waiting_point_step is None:
            return original_score

        wait_s = (
            self.step_count - robot.waiting_point_step
        ) * self.physical.simulation_dt_s

        if wait_s >= self.wait_threshold_s:
            # 오래 기다린 차량을 우선하되,
            # 보호 대상이 여러 대라면 가장 오래 기다린 차량 우선
            return 1000000.0 + wait_s

        return original_score

with Path(
    "outputs/step89_original_scenarios/scenario_manifest.csv"
).open(encoding="utf-8-sig", newline="") as f:
    all_scenarios = list(csv.DictReader(f))

mode = sys.argv[1] if len(sys.argv) > 1 else "test"

if mode == "test":
    scenario_ids = [25, 26, 36]
elif mode == "36":
    scenario_ids = list(range(1, 37))
else:
    raise ValueError("Use: test or 36")

weights = json.loads(
    Path("outputs/realistic_retraining_full_01/best_weights.json")
    .read_text(encoding="utf-8-sig")
)["weights"]

POLICIES = [
    ("FCFS", None, None),
    ("Basic", FIXED_HEURISTIC_WEIGHTS, None),
    ("BO", weights["Original_BO"], None),
    ("PSO", weights["Original_PSO"], None),
    ("FairBO_20s", weights["Original_BO"], 20.0),
    ("FairBO_40s", weights["Original_BO"], 40.0),
    ("FairPSO_20s", weights["Original_PSO"], 20.0),
    ("FairPSO_40s", weights["Original_PSO"], 40.0),
]

def run(s, policy, w, threshold):
    model_class = (
        FairBOModel if threshold is not None
        else FixedQueueModel
    )

    kwargs = dict(
        robots_by_direction=json.loads(s["allocation_json"]),
        random_seed=int(s["seed"]),
        corridor_length=8,
        west_exit_extension=0,
        spawn_gap_steps=0,
        spawn_plan_by_direction={
            d: [int(t)*15 for t in times]
            for d, times in json.loads(
                s["spawn_plan_json"]
            ).items()
        },
        goal_plan_by_direction=json.loads(s["goal_plan_json"]),
        policy_type="fcfs" if w is None else "heuristic",
        heuristic_weights=w or {},
        normalize_heuristic_features=True,
        admission_window_steps=0,
        shared_area_capacity=2,
        physical_config=AGVPhysicalConfig(
            cell_size_m=1.5,
            speed_mps=1.0,
            turn_delay_s=0.0,
            simulation_dt_s=0.1,
        ),
        omega=1.0
    )

    if threshold is not None:
        kwargs["wait_threshold_s"] = threshold

    sim = model_class(**kwargs)
    result = sim.run(max_steps=30000)

    if result["completed"] != int(s["agvs"]):
        raise RuntimeError(
            f"Incomplete: {policy}, scenario={s['case']}"
        )

    if result.get("safety_violations", 0):
        raise RuntimeError("Safety violation")

    if sim.selection_count != int(s["agvs"]):
        raise RuntimeError("Unexpected admission count")

    waits = [
        robot.shared_wait_time * 0.1
        for robot in sim.completed
    ]

    sorted_waits = sorted(waits)
    pos = 0.95 * (len(sorted_waits) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_waits) - 1)
    p95 = (
        sorted_waits[lo] * (hi - pos)
        + sorted_waits[hi] * (pos - lo)
        if hi != lo else sorted_waits[lo]
    )

    total_wait = sum(waits)
    squared_wait = sum(w * w for w in waits)
    jain = (
        total_wait ** 2 / (len(waits) * squared_wait)
        if squared_wait > 0 else 1.0
    )

    return {
        "p95_wait_s": round(p95, 3),
        "jain": round(jain, 4),
        "total_time_s": round(result["total_time"] * 0.1, 3),
        "avg_wait_s": round(mean(waits), 3),
        "max_wait_s": round(max(waits), 3),
        "over30": sum(t >= 30 for t in waits),
        "over50": sum(t >= 50 for t in waits),
    }

rows = []

for i in scenario_ids:
    s = all_scenarios[i - 1]

    for policy, w, threshold in POLICIES:
        r = run(s, policy, w, threshold)

        rows.append({
            "scenario": i,
            "case": s["case"],
            "agvs": int(s["agvs"]),
            "policy": policy,
            **r,
        })

    print(f"Scenario {i} complete", flush=True)

path = OUT / f"results_{mode}.csv"

with path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n========== STEP 96 RESULTS ==========")

averages = {}

for policy, _, _ in POLICIES:
    selected = [r for r in rows if r["policy"] == policy]

    averages[policy] = {
        "total": mean(r["total_time_s"] for r in selected),
        "avgwait": mean(r["avg_wait_s"] for r in selected),
        "maxwait": mean(r["max_wait_s"] for r in selected),
    }

    a = averages[policy]
    print(
        f"{policy:12s} | "
        f"Total={a['total']:.2f}s | "
        f"AvgWait={a['avgwait']:.2f}s | "
        f"MaxWait={a['maxwait']:.2f}s"
    )

base = averages["BO"]

print("\nChange relative to original BO:")
for policy in ("FairBO_20s", "FairBO_40s", "FairPSO_20s", "FairPSO_40s"):
    a = averages[policy]
    print(
        f"{policy}: "
        f"Total={a['total']-base['total']:+.2f}s, "
        f"MaxWait={a['maxwait']-base['maxwait']:+.2f}s"
    )

print("\nSaved:", path)


print("\n========== STEP 99 FAIRNESS RESULTS ==========")

for policy, _, _ in POLICIES:
    selected = [r for r in rows if r["policy"] == policy]

    print(
        f"{policy:12s} | "
        f"P95={mean(r['p95_wait_s'] for r in selected):.2f}s | "
        f"Jain={mean(r['jain'] for r in selected):.3f} | "
        f"Over30={sum(r['over30'] for r in selected)} | "
        f"Over50={sum(r['over50'] for r in selected)}"
    )
