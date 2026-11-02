import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from rware.realistic_cross_simulation import RealisticCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig
import scripts.run_total_time_bo_policy_comparison as base


CONFIG = AGVPhysicalConfig(
    cell_size_m=1.5,
    speed_mps=1.0,
    turn_delay_s=0.0,
    simulation_dt_s=0.1,
)

MAX_STEPS = 10000
CURRENT_PENALTY = 0.0


class FixedSpeedExperiment(RealisticCrossExperiment):
    def __init__(self, *args, **kwargs):
        kwargs["physical_config"] = CONFIG
        super().__init__(*args, **kwargs)


# 기존 최적화 함수가 새 시뮬레이터를 사용하도록
# 이 실행 프로세스 안에서만 연결한다.
base.FCFSCrossExperiment = FixedSpeedExperiment

original_run = base._run_single_policy


def training_run(*args, **kwargs):
    result = original_run(*args, **kwargs)

    if result["completed"] != result["robots"]:
        raise RuntimeError(
            "Training scenario did not complete."
        )

    if result["safety_violations"] != 0:
        raise RuntimeError(
            "Safety validation failed."
        )

    result = dict(result)

    # 기존 최적화 함수가 읽는 total_time에
    # 학습 목적함수를 전달한다.
    result["total_time"] = (
        result["total_time"]
        + CURRENT_PENALTY * result["max_wait_time"]
    )

    return result


base._run_single_policy = training_run


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        fieldnames = list(dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        ))

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate(scenario, name, weights):
    policy_type = (
        "fcfs" if name == "FCFS" else "heuristic"
    )

    sim = FixedSpeedExperiment(
        robots_by_direction=scenario.allocation,
        random_seed=scenario.seed,
        corridor_length=8,
        west_exit_extension=0,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario.spawn_plan,
        goal_plan_by_direction=scenario.goal_plan,
        policy_type=policy_type,
        heuristic_weights=weights,
        normalize_heuristic_features=True,
        admission_window_steps=0,
        shared_area_capacity=2,
    )

    result = sim.run(max_steps=MAX_STEPS)

    if (
        result["completed"] != scenario.total_agvs
        or result["safety_violations"] != 0
    ):
        raise RuntimeError(
            f"Failed: {name}, scenario={scenario.run_index}, "
            f"completed={result['completed']}, "
            f"violations={result['safety_violations']}"
        )

    return {
        "scenario": scenario.run_index,
        "case": scenario.scenario_case_name,
        "agvs": scenario.total_agvs,
        "policy": name,
        "total_time_s": result["total_time_seconds"],
        "max_wait_s": result["max_wait_seconds"],
        "completed": result["completed"],
    }


def main():
    global CURRENT_PENALTY

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true"
    )
    args = parser.parse_args()

    if args.full:
        trials = 30
        evaluations = 30

        train = base._generate_pdf_case_scenarios(
            case_ids=(1, 2, 3, 4),
            scenarios_per_case=100,
            total_agv_options=(20,),
            seed_base=210000,
            scenario_seed=20261022,
            split="train",
        )

        test = base._generate_pdf_case_scenarios(
            case_ids=(1, 2, 3, 4, 5),
            scenarios_per_case=5,
            total_agv_options=(16, 20, 24),
            seed_base=310000,
            scenario_seed=20261023,
            split="test",
        )

        out = Path(
            "outputs/realistic_retraining_full_01"
        )

    else:
        # 적은 조건으로 실행 가능 여부부터 확인
        trials = 8
        evaluations = 8

        train = base._generate_pdf_case_scenarios(
            case_ids=(1, 2, 3, 4),
            scenarios_per_case=1,
            total_agv_options=(16, 20, 24),
            seed_base=210000,
            scenario_seed=20261022,
            split="train",
        )

        test = base._generate_pdf_case_scenarios(
            case_ids=(1, 2, 3, 4, 5),
            scenarios_per_case=1,
            total_agv_options=(16, 20, 24),
            seed_base=310000,
            scenario_seed=20261023,
            split="test",
        )

        out = Path(
            "outputs/realistic_retraining_pilot"
        )

    if out.exists():
        raise SystemExit(
            f"Output directory already exists: {out}\n"
            "Existing experiment results will not be overwritten."
        )

    out.mkdir(parents=True)

    print("=== REALISTIC BO/PSO RETRAINING ===")
    print("Train scenarios:", len(train))
    print("Test scenarios:", len(test))
    print("BO trials:", trials)
    print("PSO evaluations:", evaluations)
    print("Speed: 1.0m/s, turn delay: 0.0s")

    bounds = base.DEFAULT_BO_WEIGHT_BOUNDS

    rng = np.random.default_rng(20261022)
    initial_vectors = [
        base._random_weight_vector(rng, bounds)
        for _ in range(2)
    ]

    policies = {
        "FCFS": None,
        "Basic": dict(
            base.FIXED_HEURISTIC_WEIGHTS
        ),
    }

    trial_records = []

    jobs = [
        ("Original_BO", "BO", 0.0),
        ("Improved_BO", "BO", 0.25),
        ("Original_PSO", "PSO", 0.0),
        ("Improved_PSO", "PSO", 1.0),
    ]

    for name, algorithm, penalty in jobs:
        CURRENT_PENALTY = penalty

        print(
            f"\nTraining {name}, lambda={penalty}",
            flush=True,
        )

        if algorithm == "BO":
            result = base._run_bayesian_optimization(
                scenarios=train,
                initial_weights=base.FIXED_HEURISTIC_WEIGHTS,
                trials=trials,
                initial_local_trials=2,
                initial_random_trials=2,
                candidate_count=100,
                bounds=bounds,
                initial_random_vectors=initial_vectors,
                rng_seed=42,
                kappa=1.5,
                global_ratio=0.5,
                local_scale_ratio=0.2,
                length_scale=0.3,
                noise=1e-6,
                max_steps=MAX_STEPS,
                corridor_length=8,
                west_exit_extension=0,
                admission_window_steps=0,
                shared_area_capacity=2,
            )

        else:
            result = base._run_particle_swarm_optimization(
                scenarios=train,
                swarm_size=4 if not args.full else 6,
                evaluations=evaluations,
                bounds=bounds,
                initial_vectors=initial_vectors,
                rng_seed=42,
                inertia_start=0.7,
                inertia_end=0.4,
                cognitive_coefficient=1.5,
                social_coefficient=1.5,
                velocity_limit_ratio=0.2,
                max_steps=MAX_STEPS,
                corridor_length=8,
                west_exit_extension=0,
                admission_window_steps=0,
                shared_area_capacity=2,
            )

        policies[name] = result.best_weights

        trial_records.extend(
            {"policy": name, **row}
            for row in result.trial_rows
        )

        print(
            f"{name} training complete. "
            f"Objective={result.best_objective:.3f}",
            flush=True,
        )

    (out / "best_weights.json").write_text(
        json.dumps(
            {
                "weights": policies,
                "train_scenarios": len(train),
                "test_scenarios": len(test),
                "bo_trials": trials,
                "pso_evaluations": evaluations,
                "speed_mps": 1.0,
                "cell_size_m": 1.5,
                "turn_delay_s": 0.0,
                "simulation_dt_s": 0.1,
                "objective_unit": "simulation ticks",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(out / "training_trials.csv", trial_records)

    CURRENT_PENALTY = 0.0

    print("\n=== TEST EVALUATION ===", flush=True)

    rows = []

    for scenario in test:
        for name, weights in policies.items():
            row = evaluate(scenario, name, weights)
            rows.append(row)

        print(
            f"Test scenario {scenario.run_index + 1}"
            f"/{len(test)} complete",
            flush=True,
        )

    write_csv(out / "test_results.csv", rows)

    summary = []

    for name in policies:
        selected = [
            r for r in rows
            if r["policy"] == name
        ]

        summary.append({
            "policy": name,
            "scenarios": len(selected),
            "avg_total_time_s": round(
                statistics.fmean(
                    r["total_time_s"] for r in selected
                ), 3
            ),
            "avg_max_wait_s": round(
                statistics.fmean(
                    r["max_wait_s"] for r in selected
                ), 3
            ),
        })

    write_csv(out / "summary.csv", summary)

    print("\n=== FINAL TEST SUMMARY ===")

    for item in summary:
        print(
            f"{item['policy']:13s} | "
            f"Total={item['avg_total_time_s']:7.2f}s | "
            f"MaxWait={item['avg_max_wait_s']:6.2f}s"
        )

    print("\nSaved:", out.resolve())


if __name__ == "__main__":
    main()
