import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean

from scripts.run_total_time_bo_policy_comparison import (
    _generate_pdf_case_scenarios,
    FIXED_HEURISTIC_WEIGHTS,
)
from rware.realistic_agv_config import AGVPhysicalConfig
from rware.realistic_cross_simulation import RealisticCrossExperiment

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 15
OUT = Path("outputs/realistic_turn_speed_step70")
OUT.mkdir(parents=True, exist_ok=True)

weights = json.loads(
    Path("outputs/realistic_retraining_full_01/best_weights.json")
    .read_text(encoding="utf-8-sig")
)["weights"]

policies = {
    "FCFS": None,
    "Basic": FIXED_HEURISTIC_WEIGHTS,
    "BO": weights["Original_BO"],
    "PSO": weights["Original_PSO"],
}

class TurnSpeedExperiment(RealisticCrossExperiment):
    def __init__(self, *args, angular_speed=None, **kwargs):
        self.angular_speed = angular_speed
        super().__init__(*args, **kwargs)

    def _movement_ticks(self, robot):
        base_ticks, is_turn = super()._movement_ticks(robot)

        if not is_turn or self.angular_speed is None:
            return base_ticks, is_turn

        prev = robot.path[robot.path_index - 1]
        curr = robot.current_node
        nxt = robot.next_node

        p0 = self.graph.nodes[prev].position
        p1 = self.graph.nodes[curr].position
        p2 = self.graph.nodes[nxt].position

        v1 = (p1[0]-p0[0], p1[1]-p0[1])
        v2 = (p2[0]-p1[0], p2[1]-p1[1])

        cross = v1[0]*v2[1] - v1[1]*v2[0]
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        angle = abs(math.atan2(cross, dot))

        dt = self.physical.simulation_dt_s
        move_time = base_ticks * dt
        turn_time = angle / self.angular_speed

        ticks = math.ceil((move_time + turn_time) / dt - 1e-9)
        return max(1, ticks), True

scenarios = _generate_pdf_case_scenarios(
    case_ids=(1, 2, 3, 4, 5),
    scenarios_per_case=5,
    total_agv_options=(16, 20, 24),
    seed_base=660000,
    scenario_seed=20261126,
    split="test",
)[:LIMIT]

config = AGVPhysicalConfig(
    cell_size_m=1.5,
    speed_mps=1.0,
    turn_delay_s=0.0,
    simulation_dt_s=0.1,
)

conditions = {
    "No_turn_delay": None,
    "Omega_1_0": 1.0,
    "Omega_1_5": 1.5,
}

rows = []

for i, scenario in enumerate(scenarios, 1):
    for condition, omega in conditions.items():
        for policy, policy_weights in policies.items():
            sim = TurnSpeedExperiment(
                robots_by_direction=scenario.allocation,
                random_seed=scenario.seed,
                corridor_length=8,
                west_exit_extension=0,
                spawn_gap_steps=0,
                spawn_plan_by_direction={
                    d: [int(t)*15 for t in plan]
                    for d, plan in scenario.spawn_plan.items()
                },
                goal_plan_by_direction=scenario.goal_plan,
                policy_type="fcfs" if policy == "FCFS" else "heuristic",
                heuristic_weights=policy_weights or {},
                normalize_heuristic_features=True,
                admission_window_steps=0,
                shared_area_capacity=2,
                physical_config=config,
                angular_speed=omega,
            )

            result = sim.run(max_steps=20000)

            if result["completed"] != scenario.total_agvs:
                raise RuntimeError(
                    f"Incomplete: {i}, {condition}, {policy}"
                )
            if result["safety_violations"]:
                raise RuntimeError(
                    f"Safety violation: {i}, {condition}, {policy}"
                )

            rows.append({
                "scenario": i,
                "case": scenario.scenario_case_name,
                "agvs": scenario.total_agvs,
                "condition": condition,
                "policy": policy,
                "total_time_s": float(result["total_time"])*0.1,
            })

    print(f"Completed {i}/{len(scenarios)} scenarios", flush=True)

path = OUT / f"results_{LIMIT}.csv"

with path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n=== STEP 70 RESULTS ===")

for condition in conditions:
    print(f"\n[{condition}]")
    averages = {}

    for policy in policies:
        values = [
            r["total_time_s"] for r in rows
            if r["condition"] == condition and r["policy"] == policy
        ]
        averages[policy] = mean(values)
        print(f"{policy}: {averages[policy]:.2f}s")

    for policy in ("Basic", "BO", "PSO"):
        gain = 100 * (
            averages["FCFS"] - averages[policy]
        ) / averages["FCFS"]
        print(f"{policy} improvement vs FCFS: {gain:.2f}%")

print("\nSaved:", path)
