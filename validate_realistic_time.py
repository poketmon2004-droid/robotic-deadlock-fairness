from collections import defaultdict
from pathlib import Path
import csv

from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig
from scripts.run_total_time_bo_policy_comparison import (
    _generate_pdf_case_scenarios,
)

DT = 0.1
OUTPUT = Path("outputs/realistic_time_validation")
OUTPUT.mkdir(parents=True, exist_ok=True)

scenarios = _generate_pdf_case_scenarios(
    case_ids=(1,),
    scenarios_per_case=1,
    total_agv_options=(16, 20, 24),
    seed_base=120000,
    scenario_seed=20261201,
    split="test",
)

rows = []

print("=== STEP 35 TIME VALIDATION ===", flush=True)

for scenario in scenarios:
    sim = RealisticCrossExperiment(
        robots_by_direction=scenario.allocation,
        random_seed=scenario.seed,
        corridor_length=8,
        west_exit_extension=0,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario.spawn_plan,
        goal_plan_by_direction=scenario.goal_plan,
        policy_type="fcfs",
        admission_window_steps=0,
        shared_area_capacity=2,
        physical_config=AGVPhysicalConfig(
            cell_size_m=1.5,
            speed_mps=1.0,
            turn_delay_s=0.0,
            simulation_dt_s=DT,
        ),
    )

    result = sim.run(max_steps=10000)

    events = defaultdict(dict)

    for event in sim.event_log:
        rid = event["robot_id"]
        kind = event["event_type"]
        step = event["step"]

        if kind == "fcfs_waiting_point_arrival":
            events[rid].setdefault("waiting", step)

        elif kind in ("fcfs_admitted", "heuristic_admitted"):
            events[rid].setdefault("admitted", step)

        elif kind == "shared_area_entered":
            events[rid].setdefault("entered", step)

        elif kind in ("spawned", "initially_placed"):
            events[rid].setdefault("spawned", step)

    completed = sorted(
        sim.completed,
        key=lambda robot: robot.robot_id,
    )

    waiting_matches = 0
    waiting_checked = 0
    post_admission_delays = []
    missing_events = []

    for robot in completed:
        rid = robot.robot_id
        e = events[rid]

        waiting = e.get("waiting")
        admitted = e.get("admitted")
        entered = e.get("entered")

        if waiting is not None and admitted is not None:
            waiting_checked += 1
            expected_wait = admitted - waiting

            if expected_wait == robot.shared_wait_time:
                waiting_matches += 1

        else:
            missing_events.append(rid)

        if admitted is not None and entered is not None:
            # Approximate delay from authorization
            # to recorded conflict-zone entry.
            post_admission_delays.append(
                max(0, entered - admitted) * DT
            )

    planned_ticks = [
        step
        for plan in scenario.spawn_plan.values()
        for step in plan
    ]

    actual_spawn_seconds = [
        robot.spawn_step * DT
        for robot in completed
        if robot.spawn_step is not None
    ]

    row = {
        "agvs": scenario.total_agvs,
        "completed": result["completed"],
        "total_time_s": result["total_time_seconds"],
        "max_wait_recorded_s": result["max_wait_seconds"],
        "waiting_checked": waiting_checked,
        "waiting_matches": waiting_matches,
        "missing_event_count": len(missing_events),
        "max_authorization_to_entry_s": (
            round(max(post_admission_delays), 2)
            if post_admission_delays else None
        ),
        "planned_last_spawn_tick": max(planned_ticks),
        "planned_last_spawn_as_seconds": round(
            max(planned_ticks) * DT, 2
        ),
        "actual_last_spawn_seconds": (
            round(max(actual_spawn_seconds), 2)
            if actual_spawn_seconds else None
        ),
        "safety_violations": result["safety_violations"],
    }

    rows.append(row)

    print(f"\n--- AGVs: {scenario.total_agvs} ---")
    print(
        f"Completed: {result['completed']}/"
        f"{scenario.total_agvs}"
    )
    print(
        f"Total completion time: "
        f"{result['total_time_seconds']:.2f}s"
    )
    print(
        f"Recorded max wait: "
        f"{result['max_wait_seconds']:.2f}s"
    )
    print(
        f"Wait accounting matches: "
        f"{waiting_matches}/{waiting_checked}"
    )
    print(
        f"Missing wait/admission events: "
        f"{len(missing_events)}"
    )
    print(
        f"Max authorization-to-entry delay: "
        f"{row['max_authorization_to_entry_s']}s"
    )
    print(
        f"Planned last spawn: "
        f"{row['planned_last_spawn_tick']} ticks "
        f"= {row['planned_last_spawn_as_seconds']}s"
    )
    print(
        f"Actual last spawn: "
        f"{row['actual_last_spawn_seconds']}s"
    )
    print(
        f"Safety violations: "
        f"{result['safety_violations']}"
    )

    if result["completed"] != scenario.total_agvs:
        print("WARNING: Not all AGVs completed.")

    if waiting_matches != waiting_checked:
        print("WARNING: Wait accounting mismatch.")

    if missing_events:
        print("WARNING: Missing events:", missing_events[:5])

file = OUTPUT / "validation_results.csv"

with file.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\nSaved:", file)
print("STEP 35 DIAGNOSTIC COMPLETE")
