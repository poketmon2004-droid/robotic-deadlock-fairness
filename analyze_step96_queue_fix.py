import ast
import csv
import json
import math
from pathlib import Path

from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig
from scripts.run_total_time_bo_policy_comparison import FIXED_HEURISTIC_WEIGHTS

def extract_class(filename, name, ns):
    src = Path(filename).read_text(encoding="utf-8-sig")
    tree = ast.parse(src)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    exec(compile(ast.Module(body=[cls], type_ignores=[]), filename, "exec"), ns)
    return ns[name]

TurnModel = extract_class(
    "analyze_step78.py", "TurnModel",
    {"math": math, "RealisticCrossExperiment": RealisticCrossExperiment}
)

IndividualSpeedModel = extract_class(
    "analyze_step93.py", "IndividualSpeedModel",
    {"math": math, "TurnModel": TurnModel}
)

class FixedQueueModel(IndividualSpeedModel):
    def _enqueue_waiting_robots(self):
        for robot in sorted(self.active.values(), key=lambda r: r.robot_id):
            area = self._area_for_waiting_transition(robot)
            if area is None:
                continue

            # 핵심 수정: 이미 진입 허가된 차량은 재등록 금지
            if robot.robot_id in self.shared_robot_ids_by_area[area.area_id]:
                continue

            queue = self.fcfs_queues[area.area_id]
            if robot.robot_id in queue:
                continue

            robot.waiting_point_step = self.step_count
            queue.append(robot.robot_id)
            self._event(
                "fcfs_waiting_point_arrival",
                robot,
                node=robot.current_node,
                metadata={"area_id": area.area_id},
            )

with open(
    "outputs/step89_original_scenarios/scenario_manifest.csv",
    encoding="utf-8-sig", newline=""
) as f:
    scenario = list(csv.DictReader(f))[24]

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

def run(model_class, policy, w):
    sim = model_class(
        robots_by_direction=json.loads(scenario["allocation_json"]),
        random_seed=int(scenario["seed"]),
        corridor_length=8,
        west_exit_extension=0,
        spawn_gap_steps=0,
        spawn_plan_by_direction={
            d: [int(t)*15 for t in times]
            for d, times in json.loads(
                scenario["spawn_plan_json"]
            ).items()
        },
        goal_plan_by_direction=json.loads(scenario["goal_plan_json"]),
        policy_type="fcfs" if w is None else "heuristic",
        heuristic_weights=w or {},
        normalize_heuristic_features=True,
        admission_window_steps=0,
        shared_area_capacity=2,
        physical_config=AGVPhysicalConfig(
            cell_size_m=1.5,
            speed_mps=1.0,
            turn_delay_s=0.0,
            simulation_dt_s=0.1
        ),
        omega=1.0
    )

    result = sim.run(max_steps=30000)

    if result["completed"] != int(scenario["agvs"]):
        raise RuntimeError("Incomplete simulation")

    if result.get("safety_violations", 0):
        raise RuntimeError("Safety violation")

    waits = {
        r.robot_id: r.shared_wait_time * 0.1
        for r in sim.completed
    }

    from collections import Counter
    event_counts = Counter(
        e["event_type"] for e in sim.event_log
    )

    return {
        "total": result["total_time"] * 0.1,
        "avg_wait": sum(waits.values()) / len(waits),
        "max_wait": max(waits.values()),
        "w1_wait": waits.get("AGV_W1"),
        "arrival_events": event_counts["fcfs_waiting_point_arrival"],
        "admissions": sim.selection_count,
    }

print("\n========== STEP 96-2 QUEUE FIX TEST ==========")
print("Scenario 25 / S5 / 28 AGVs / baseline speed")

for policy, weights_for_policy in policies.items():
    original = run(IndividualSpeedModel, policy, weights_for_policy)
    fixed = run(FixedQueueModel, policy, weights_for_policy)

    print(f"\n{policy}")
    for name, old, new in [
        ("Total Time", original["total"], fixed["total"]),
        ("Avg Wait", original["avg_wait"], fixed["avg_wait"]),
        ("Max Wait", original["max_wait"], fixed["max_wait"]),
        ("AGV_W1 Wait", original["w1_wait"], fixed["w1_wait"]),
        ("Waiting events", original["arrival_events"], fixed["arrival_events"]),
        ("Admissions", original["admissions"], fixed["admissions"]),
    ]:
        print(f"  {name:16s}: before={old} / after={new}")

print("\nOriginal source files unchanged.")
