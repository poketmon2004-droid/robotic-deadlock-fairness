from collections import Counter
from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig

print("=== REALISTIC AGV VALIDATION ===", flush=True)

for count in (1, 2, 4):
    for speed in (0.5, 1.0, 1.5):
        config = AGVPhysicalConfig(
            cell_size_m=1.0,
            speed_mps=speed,
            turn_delay_s=1.0,
            simulation_dt_s=0.1,
        )

        sim = RealisticCrossExperiment(
            robots_per_direction=count,
            random_seed=123,
            corridor_length=8,
            policy_type="fcfs",
            admission_window_steps=2,
            shared_area_capacity=2,
            physical_config=config,
        )

        initial_robots = sim.total_robots
        over_capacity = 0
        overlapping_shared_nodes = 0

        while not sim.is_done and sim.step_count < 30000:
            sim.step()

            for area in sim.areas:
                occupants = sim.shared_robot_ids_by_area[area.area_id]

                if len(occupants) > sim.shared_area_capacity:
                    over_capacity += 1

                positions = [
                    robot.current_node
                    for robot in sim.active.values()
                    if robot.current_node in area.conflict_zone_nodes
                ]

                counts = Counter(positions)
                if any(n > 1 for n in counts.values()):
                    overlapping_shared_nodes += 1

        result = sim.metrics()

        print(
            f"AGVs={initial_robots:2d} | "
            f"Speed={speed:.1f} | "
            f"Completed={result['completed']}/{initial_robots} | "
            f"Total={result['total_time_seconds']:.1f}s | "
            f"MaxWait={result['max_wait_seconds']:.1f}s | "
            f"CapacityViolations={over_capacity} | "
            f"SharedNodeOverlaps={overlapping_shared_nodes}",
            flush=True,
        )

        if result["completed"] != initial_robots:
            print("WARNING: Not every AGV completed.", flush=True)

        if over_capacity or overlapping_shared_nodes:
            print("WARNING: Shared-area safety check failed.", flush=True)

print("\nVALIDATION FINISHED")
print("Note: Passing these checks does not prove physical safety.")
