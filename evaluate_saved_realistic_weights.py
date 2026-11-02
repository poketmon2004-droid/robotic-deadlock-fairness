import json
import csv
import statistics
from pathlib import Path

from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig
from scripts.run_total_time_bo_policy_comparison import (
    _generate_pdf_case_scenarios,
    FIXED_HEURISTIC_WEIGHTS,
)

weights_file = Path(
    "outputs/realistic_retraining_pilot/best_weights.json"
)

data = json.loads(weights_file.read_text(encoding="utf-8"))
policies = data["weights"]

scenarios = _generate_pdf_case_scenarios(
    case_ids=(1, 2, 3, 4, 5),
    scenarios_per_case=1,
    total_agv_options=(16, 20, 24),
    seed_base=310000,
    scenario_seed=20261023,
    split="test",
)

output = Path("outputs/realistic_retraining_pilot")
results = []

print("=== SAVED WEIGHTS TEST ===", flush=True)

for scenario in scenarios:
    for name, weights in policies.items():

        sim = RealisticCrossExperiment(
            robots_by_direction=scenario.allocation,
            random_seed=scenario.seed,
            corridor_length=8,
            spawn_gap_steps=0,
            spawn_plan_by_direction=scenario.spawn_plan,
            goal_plan_by_direction=scenario.goal_plan,
            policy_type="fcfs" if name == "FCFS" else "heuristic",
            heuristic_weights=weights,
            normalize_heuristic_features=True,
            admission_window_steps=0,
            shared_area_capacity=2,
            physical_config=AGVPhysicalConfig(
                cell_size_m=1.5,
                speed_mps=1.0,
                turn_delay_s=0.0,
                simulation_dt_s=0.1,
            ),
        )

        metrics = sim.run(max_steps=10000)

        if (
            metrics["completed"] != scenario.total_agvs
            or metrics["safety_violations"] != 0
        ):
            raise RuntimeError(
                f"Validation failed: {name}, "
                f"scenario={scenario.run_index}"
            )

        results.append({
            "scenario": scenario.run_index,
            "case": scenario.scenario_case_name,
            "agvs": scenario.total_agvs,
            "policy": name,
            "total_time_s": metrics["total_time_seconds"],
            "max_wait_s": metrics["max_wait_seconds"],
        })

    print(
        f"Scenario {scenario.run_index + 1}/{len(scenarios)} completed",
        flush=True,
    )

def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

save_csv(output / "test_results.csv", results)

summary = []

for name in policies:
    selected = [r for r in results if r["policy"] == name]

    summary.append({
        "policy": name,
        "avg_total_time_s": round(
            statistics.fmean(r["total_time_s"] for r in selected), 3
        ),
        "avg_max_wait_s": round(
            statistics.fmean(r["max_wait_s"] for r in selected), 3
        ),
    })

save_csv(output / "summary.csv", summary)

print("\n=== FINAL TEST SUMMARY ===")

for row in summary:
    print(
        f"{row['policy']:13s} | "
        f"Total={row['avg_total_time_s']:7.2f}s | "
        f"MaxWait={row['avg_max_wait_s']:6.2f}s"
    )

print("\nSaved:", output)
