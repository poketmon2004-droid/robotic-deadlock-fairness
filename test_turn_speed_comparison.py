import csv
from pathlib import Path

from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig

output = Path("outputs/turn_speed_comparison")
output.mkdir(parents=True, exist_ok=True)

rows = []

conditions = [
    ("A_no_slowdown", 0.0),
    ("B_turn_slowdown", 1.5),
]

for agv_count in (8, 16, 20, 24):
    for name, turn_delay in conditions:
        config = AGVPhysicalConfig(
            cell_size_m=1.5,
            speed_mps=1.0,
            turn_delay_s=turn_delay,
            simulation_dt_s=0.1,
        )

        sim = RealisticCrossExperiment(
            robots_per_direction=agv_count // 4,
            random_seed=123,
            corridor_length=8,
            policy_type="fcfs",
            admission_window_steps=2,
            shared_area_capacity=2,
            physical_config=config,
        )

        result = sim.run(max_steps=30000)

        if result["completed"] != agv_count:
            raise RuntimeError(
                f"{name}: Only {result['completed']}/{agv_count} AGVs completed"
            )

        row = {
            "agv_count": agv_count,
            "condition": name,
            "turn_delay_s": turn_delay,
            "total_time_s": round(result["total_time_seconds"], 2),
            "max_wait_s_preliminary": round(result["max_wait_seconds"], 2),
        }
        rows.append(row)

        print(
            f"AGVs={agv_count:2d} | "
            f"{name:16s} | "
            f"Total={row['total_time_s']:6.1f}s | "
            f"MaxWait(preliminary)={row['max_wait_s_preliminary']:5.1f}s",
            flush=True,
        )

csv_path = output / "turn_speed_results.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n=== A/B DIFFERENCES ===")

for agv_count in (8, 16, 20, 24):
    a = next(
        r for r in rows
        if r["agv_count"] == agv_count
        and r["condition"] == "A_no_slowdown"
    )
    b = next(
        r for r in rows
        if r["agv_count"] == agv_count
        and r["condition"] == "B_turn_slowdown"
    )

    delta = b["total_time_s"] - a["total_time_s"]

    print(
        f"{agv_count:2d} AGVs | "
        f"A={a['total_time_s']:.1f}s | "
        f"B={b['total_time_s']:.1f}s | "
        f"Difference={delta:+.1f}s"
    )

print(f"\nSaved: {csv_path}")
