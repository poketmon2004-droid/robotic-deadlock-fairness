import csv, json, math
from pathlib import Path
from statistics import mean
import numpy as np

from scripts.run_total_time_bo_policy_comparison import (
    _generate_pdf_case_scenarios,
    FIXED_HEURISTIC_WEIGHTS,
)
from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig

OUT = Path("outputs/step78_new_scenarios")
OUT.mkdir(parents=True, exist_ok=True)

WEIGHTS = json.loads(
    Path("outputs/realistic_retraining_full_01/best_weights.json")
    .read_text(encoding="utf-8-sig")
)["weights"]

CONFIG = AGVPhysicalConfig(
    cell_size_m=1.5,
    speed_mps=1.0,
    turn_delay_s=0.0,
    simulation_dt_s=0.1,
)

class TurnModel(RealisticCrossExperiment):
    def __init__(self, *args, omega, **kwargs):
        self.omega = omega
        super().__init__(*args, **kwargs)

    def _movement_ticks(self, robot):
        ticks, is_turn = super()._movement_ticks(robot)
        if not is_turn:
            return ticks, False

        p = robot.path
        nodes = (
            p[robot.path_index - 1],
            robot.current_node,
            robot.next_node
        )
        a, b, c = [
            self.graph.nodes[n].position for n in nodes
        ]
        v = (b[0]-a[0], b[1]-a[1])
        w = (c[0]-b[0], c[1]-b[1])

        angle = abs(math.atan2(
            v[0]*w[1]-v[1]*w[0],
            v[0]*w[0]+v[1]*w[1]
        ))

        seconds = ticks * 0.1 + angle / self.omega
        return max(1, math.ceil(seconds/0.1-1e-9)), True

# 기존 STEP 70과 다른 난수 사용
SCENARIOS = _generate_pdf_case_scenarios(
    case_ids=(1,2,3,4,5),
    scenarios_per_case=5,
    total_agv_options=(16,20,24),
    seed_base=880000,
    scenario_seed=20270126,
    split="test",
)

POLICIES = {
    "FCFS": None,
    "Basic": FIXED_HEURISTIC_WEIGHTS,
    "BO": WEIGHTS["Original_BO"],
    "PSO": WEIGHTS["Original_PSO"],
}

def run(s, omega, weights):
    sim = TurnModel(
        robots_by_direction=s.allocation,
        random_seed=s.seed,
        corridor_length=8,
        west_exit_extension=0,
        spawn_gap_steps=0,
        spawn_plan_by_direction={
            d: [int(t)*15 for t in plan]
            for d, plan in s.spawn_plan.items()
        },
        goal_plan_by_direction=s.goal_plan,
        policy_type="fcfs" if weights is None else "heuristic",
        heuristic_weights=weights or {},
        normalize_heuristic_features=True,
        admission_window_steps=0,
        shared_area_capacity=2,
        physical_config=CONFIG,
        omega=omega,
    )

    result = sim.run(max_steps=30000)

    if result["completed"] != s.total_agvs:
        raise RuntimeError("AGV completion failure")

    if result["safety_violations"]:
        raise RuntimeError("Safety violation detected")

    return round(float(result["total_time"]) * 0.1, 6)

# 시뮬레이션 재현성 검증
print("Checking reproducibility...", flush=True)
for omega in (1.0, 1.5):
    a = run(SCENARIOS[0], omega, POLICIES["BO"])
    b = run(SCENARIOS[0], omega, POLICIES["BO"])
    assert a == b, f"Non-reproducible result: {a} != {b}"
print("Reproducibility PASS", flush=True)

rows = []

for omega in (1.0, 1.5):
    for i, s in enumerate(SCENARIOS, 1):
        times = {
            policy: run(s, omega, w)
            for policy, w in POLICIES.items()
        }

        for policy, total_time in times.items():
            rows.append({
                "omega": omega,
                "scenario": i,
                "seed": s.seed,
                "case": s.scenario_case_name,
                "agvs": s.total_agvs,
                "policy": policy,
                "total_time_s": total_time,
            })

        print(
            f"Omega={omega} | {i}/{len(SCENARIOS)}",
            flush=True
        )

with (OUT/"results.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

# 같은 시나리오에서 FCFS 대비 차이 계산
paired = {}
for r in rows:
    key = (r["omega"], r["scenario"])
    paired.setdefault(key, {})[r["policy"]] = r["total_time_s"]

rng = np.random.default_rng(20261026)

print("\n========== STEP 78 RESULTS ==========")

for omega in (1.0, 1.5):
    print(f"\nOmega={omega}")

    group = [
        v for (o, _), v in sorted(paired.items())
        if o == omega
    ]

    fcfs_avg = mean(v["FCFS"] for v in group)
    print(f"FCFS: {fcfs_avg:.3f}s")

    for policy in ("Basic", "BO", "PSO"):
        avg = mean(v[policy] for v in group)

        # 양수 = FCFS 대비 완료시간 절약
        diff = np.array([
            v["FCFS"] - v[policy]
            for v in group
        ])

        boot = np.array([
            rng.choice(diff, len(diff), replace=True).mean()
            for _ in range(10000)
        ])

        lo, hi = np.percentile(boot, [2.5, 97.5])

        print(
            f"{policy}: avg={avg:.3f}s | "
            f"improve={(fcfs_avg-avg)/fcfs_avg*100:.2f}% | "
            f"saved={diff.mean():+.3f}s | "
            f"CI95=[{lo:+.3f}, {hi:+.3f}] | "
            f"better={sum(diff>1e-8)}, "
            f"worse={sum(diff<-1e-8)}, "
            f"same={sum(abs(diff)<=1e-8)}"
        )

print("\nSaved:", OUT/"results.csv")
