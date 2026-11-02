from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig

print("=== REALISTIC AGV SPEED TEST ===")

results = []

for speed in (0.5, 1.0, 1.5):
    config = AGVPhysicalConfig(
        cell_size_m=1.0,
        speed_mps=speed,
        turn_delay_s=1.0,
        simulation_dt_s=0.1,
    )

    experiment = RealisticCrossExperiment(
        robots_per_direction=2,
        random_seed=123,
        corridor_length=8,
        policy_type="fcfs",
        admission_window_steps=2,
        shared_area_capacity=2,
        physical_config=config,
    )

    result = experiment.run(max_steps=30000)

    assert result["completed"] == 8, (
        f"Only {result['completed']}/8 AGVs completed "
        f"at speed {speed}"
    )

    results.append(result)

    print(
        f"Speed={speed:.1f} m/s | "
        f"Completed={result['completed']}/8 | "
        f"Total={result['total_time_seconds']:.1f} s | "
        f"MaxWait={result['max_wait_seconds']:.1f} s"
    )

print("\nSpeed comparison completed!")
print("Note: This is an approximate grid-based timing model.")
