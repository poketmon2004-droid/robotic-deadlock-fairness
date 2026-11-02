import ast
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean

from scripts.run_total_time_bo_policy_comparison import FIXED_HEURISTIC_WEIGHTS
from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = Path("outputs/step93_individual_speeds")
OUT.mkdir(parents=True, exist_ok=True)

with Path(
    "outputs/step89_original_scenarios/scenario_manifest.csv"
).open(encoding="utf-8-sig", newline="") as f:
    scenarios = list(csv.DictReader(f))[:LIMIT]

# STEP 78의 회전 모델 클래스만 가져오기
tree = ast.parse(
    Path("analyze_step78.py").read_text(encoding="utf-8-sig")
)
cls = next(
    n for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "TurnModel"
)
ns = {
    "math": math,
    "RealisticCrossExperiment": RealisticCrossExperiment
}
exec(
    compile(
        ast.Module(body=[cls], type_ignores=[]),
        "analyze_step78.py",
        "exec"
    ),
    ns
)
TurnModel = ns["TurnModel"]

weights = json.loads(
    Path("outputs/realistic_retraining_full_01/best_weights.json")
    .read_text(encoding="utf-8-sig")
)["weights"]

POLICIES = {
    "FCFS": None,
    "Basic": FIXED_HEURISTIC_WEIGHTS,
    "BO": weights["Original_BO"],
    "PSO": weights["Original_PSO"],
}

CONDITIONS = (
    "baseline",
    "individual_variation",
    "north_slow",
    "mixed_slow",
)
OMEGA = 1.0

config = AGVPhysicalConfig(
    cell_size_m=1.5,
    speed_mps=1.0,
    turn_delay_s=0.0,
    simulation_dt_s=0.1,
)


class IndividualSpeedModel(TurnModel):

    def __init__(self, *args, **kwargs):
        self.robot_speeds = {}
        self.speed_move_checks = 0
        self.different_duration_checks = 0
        super().__init__(*args, **kwargs)

    def _movement_ticks(self, robot):
        speed = self.robot_speeds.get(robot.robot_id, 1.0)

        if speed <= 0:
            raise ValueError("AGV speed must be positive")

        start = robot.current_node
        end = robot.next_node

        p1 = self.graph.nodes[start].position
        p2 = self.graph.nodes[end].position

        distance = math.dist(p1, p2) * self.physical.cell_size_m
        move_time = distance / speed

        is_turn = False
        angle = 0.0

        if robot.path_index >= 1:
            previous = robot.path[robot.path_index - 1]
            p0 = self.graph.nodes[previous].position

            before = (
                p1[0] - p0[0],
                p1[1] - p0[1]
            )
            after = (
                p2[0] - p1[0],
                p2[1] - p1[1]
            )

            is_turn = before != after

            if is_turn:
                cross = (
                    before[0] * after[1]
                    - before[1] * after[0]
                )
                dot = (
                    before[0] * after[0]
                    + before[1] * after[1]
                )
                angle = abs(math.atan2(cross, dot))

        turn_time = (
            self.physical.turn_delay_s + angle / self.omega
            if is_turn else 0.0
        )

        dt = self.physical.simulation_dt_s
        total_time = move_time + turn_time

        ticks = max(
            1, math.ceil(total_time / dt - 1e-9)
        )

        # 실제로 차량별 속도가 이동시간을 바꾸는지 확인
        reference_ticks, reference_turn = super()._movement_ticks(robot)

        if is_turn != reference_turn:
            raise RuntimeError("Turn detection mismatch")

        if abs(speed - 1.0) < 1e-10 and ticks != reference_ticks:
            raise RuntimeError("Baseline movement mismatch")

        self.speed_move_checks += 1

        if ticks != reference_ticks:
            self.different_duration_checks += 1

        return ticks, is_turn


def get_robots(sim):
    robots = {}

    for direction, queue in sim.pending_by_direction.items():
        for robot in queue:
            robots[robot.robot_id] = robot.direction

    for robot in sim.active.values():
        robots[robot.robot_id] = robot.direction

    for robot in sim.completed:
        robots[robot.robot_id] = robot.direction

    return robots


def build_speeds(robots, condition, seed):
    speeds = {rid: 1.0 for rid in robots}
    rng = random.Random(seed)
    ids = sorted(robots)

    if condition == "individual_variation":
        for rid in ids:
            speeds[rid] = rng.uniform(0.8, 1.2)

    elif condition == "north_slow":
        for rid in ids:
            if robots[rid] == "NORTH":
                speeds[rid] = 0.7

    elif condition == "mixed_slow":
        rng.shuffle(ids)
        slow_count = round(len(ids) * 0.25)
        for rid in ids[:slow_count]:
            speeds[rid] = 0.6

    elif condition != "baseline":
        raise ValueError(condition)

    return speeds


rows = []
speed_rows = []

for i, s in enumerate(scenarios, 1):
    allocation = json.loads(s["allocation_json"])
    spawn = json.loads(s["spawn_plan_json"])
    goals = json.loads(s["goal_plan_json"])

    # 동일한 시나리오의 정책 간 속도 배정을 고정
    scenario_seed = int(s["seed"])

    for condition in CONDITIONS:
        expected_speeds = None

        for policy, w in POLICIES.items():
            sim = IndividualSpeedModel(
                robots_by_direction=allocation,
                random_seed=scenario_seed,
                corridor_length=8,
                west_exit_extension=0,
                spawn_gap_steps=0,
                spawn_plan_by_direction={
                    d: [int(t) * 15 for t in times]
                    for d, times in spawn.items()
                },
                goal_plan_by_direction=goals,
                policy_type=(
                    "fcfs" if w is None else "heuristic"
                ),
                heuristic_weights=w or {},
                normalize_heuristic_features=True,
                admission_window_steps=0,
                shared_area_capacity=2,
                physical_config=config,
                omega=OMEGA,
            )

            robots = get_robots(sim)

            if len(robots) != int(s["agvs"]):
                raise RuntimeError(
                    f"Robot count mismatch: {len(robots)}"
                )

            speeds = build_speeds(
                robots,
                condition,
                scenario_seed + 93000
            )

            if expected_speeds is None:
                expected_speeds = speeds.copy()
                for rid, speed in sorted(speeds.items()):
                    speed_rows.append({
                        "scenario": i,
                        "condition": condition,
                        "robot_id": rid,
                        "direction": robots[rid],
                        "speed_mps": round(speed, 6),
                    })
            elif speeds != expected_speeds:
                raise RuntimeError(
                    "Different AGV speeds across policies"
                )

            sim.robot_speeds = speeds

            result = sim.run(max_steps=30000)

            if result["completed"] != int(s["agvs"]):
                raise RuntimeError(
                    f"Incomplete: {i}, {condition}, {policy}"
                )

            if result.get("safety_violations", 0):
                raise RuntimeError("Safety violation")

            rows.append({
                "scenario": i,
                "case": s["case"],
                "agvs": int(s["agvs"]),
                "condition": condition,
                "policy": policy,
                "total_time_s": round(
                    float(result["total_time"]) * 0.1, 4
                ),
                "min_speed": round(min(speeds.values()), 4),
                "max_speed": round(max(speeds.values()), 4),
                "movement_checks": sim.speed_move_checks,
                "changed_duration_checks":
                    sim.different_duration_checks,
            })

    print(
        f"Scenario {i}/{len(scenarios)} complete",
        flush=True
    )


def save_csv(path, data):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(data[0])
        )
        writer.writeheader()
        writer.writerows(data)


save_csv(OUT / f"results_{LIMIT}.csv", rows)
save_csv(OUT / f"speed_assignments_{LIMIT}.csv", speed_rows)

print("\n========== STEP 93 RESULTS ==========")

for condition in CONDITIONS:
    subset = [
        r for r in rows
        if r["condition"] == condition
    ]

    averages = {
        policy: mean(
            r["total_time_s"]
            for r in subset
            if r["policy"] == policy
        )
        for policy in POLICIES
    }

    fcfs = averages["FCFS"]

    print(f"\nCondition: {condition}")
    for policy, avg in averages.items():
        improvement = (fcfs - avg) / fcfs * 100
        print(
            f"  {policy:5s}: {avg:.3f}s "
            f"(FCFS 대비 {improvement:+.2f}%)"
        )

    changed = sum(
        r["changed_duration_checks"]
        for r in subset
    )
    print(f"  Changed movement durations: {changed}")

# 기존 STEP 90의 Omega=1.0 결과와 기준 모델 대조
baseline_rows = [
    r for r in rows if r["condition"] == "baseline"
]

previous_path = Path(
    "outputs/step90_model_comparison/results_36.csv"
)

if previous_path.exists():
    with previous_path.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        previous = list(csv.DictReader(f))

    previous_map = {
        (int(r["scenario"]), r["policy"]):
            float(r["physical_time_s"])
        for r in previous
        if float(r["omega"]) == OMEGA
    }

    mismatches = []
    for r in baseline_rows:
        key = (r["scenario"], r["policy"])
        expected = previous_map.get(key)
        if expected is not None:
            if abs(r["total_time_s"] - expected) > 0.0001:
                mismatches.append((key, expected, r["total_time_s"]))

    if mismatches:
        raise RuntimeError(
            f"Baseline validation FAILED: {mismatches[:5]}"
        )
    print("\nSTEP 90 baseline validation: PASS")

print("\nSaved:", OUT)
