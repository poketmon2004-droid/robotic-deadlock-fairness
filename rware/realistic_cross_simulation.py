from __future__ import annotations

import math
from collections import Counter

from rware.fcfs_cross_simulation import FCFSCrossExperiment
from rware.realistic_agv_config import AGVPhysicalConfig


class RealisticCrossExperiment(FCFSCrossExperiment):
    """
    Discrete-time, grid-based AGV movement model.

    - Source cell stays occupied during movement.
    - Destination cell is reserved during movement.
    - Robot position changes only after travel time has elapsed.
    - Shared-zone admission and release use the original policy.
    """

    def __init__(self, *args, physical_config=None, **kwargs):
        self.physical = physical_config or AGVPhysicalConfig()
        self.in_flight = {}
        self.travel_ticks = {}
        self.turn_ticks = {}
        self.safety_violations = []

        super().__init__(*args, **kwargs)

    def _movement_ticks(self, robot):
        start = robot.current_node
        end = robot.next_node

        p1 = self.graph.nodes[start].position
        p2 = self.graph.nodes[end].position

        distance = math.dist(p1, p2) * self.physical.cell_size_m
        move_time = distance / self.physical.speed_mps

        is_turn = False

        if robot.path_index >= 1:
            previous = robot.path[robot.path_index - 1]
            p0 = self.graph.nodes[previous].position

            before = (p1[0] - p0[0], p1[1] - p0[1])
            after = (p2[0] - p1[0], p2[1] - p1[1])

            is_turn = before != after

        turn_time = self.physical.turn_delay_s if is_turn else 0.0

        total_time = move_time + turn_time
        dt = self.physical.simulation_dt_s

        ticks = max(1, math.ceil(total_time / dt - 1e-9))

        return ticks, is_turn

    def _reserved_nodes(self):
        return {
            trip["to_node"]
            for trip in self.in_flight.values()
        }

    def _start_movements(self, proposals):
        started = []
        occupied_or_reserved = self._reserved_nodes()

        for robot_id, proposal in proposals.items():
            if robot_id in self.in_flight:
                continue

            if not proposal["allowed"]:
                continue

            robot = self.active[robot_id]
            destination = proposal["to_node"]

            if destination in occupied_or_reserved:
                continue

            ticks, is_turn = self._movement_ticks(robot)

            self.in_flight[robot_id] = {
                "from_node": robot.current_node,
                "to_node": destination,
                "arrival_tick": self.step_count + ticks,
                "duration_ticks": ticks,
                "is_turn": is_turn,
            }

            occupied_or_reserved.add(destination)
            started.append(robot_id)

        return started

    def _finish_movements(self, arrival_tick):
        finished = []

        for robot_id, trip in list(self.in_flight.items()):
            if trip["arrival_tick"] > arrival_tick:
                continue

            robot = self.active[robot_id]

            previous = robot.current_node
            destination = trip["to_node"]

            robot.current_node = destination
            robot.path_index += 1

            self._update_heading(robot, previous, destination)

            self.travel_ticks[robot_id] = (
                self.travel_ticks.get(robot_id, 0)
                + trip["duration_ticks"]
            )

            if trip["is_turn"]:
                self.turn_ticks[robot_id] = (
                    self.turn_ticks.get(robot_id, 0)
                    + trip["duration_ticks"]
                )

            area = self._area_for_conflict_node(destination)

            if area is not None and robot.shared_entry_step is None:
                robot.shared_entry_step = arrival_tick

                self._event(
                    "shared_area_entered",
                    robot,
                    node=destination,
                    metadata={"area_id": area.area_id},
                )

            self._event(
                "moved",
                robot,
                node=destination,
                metadata={"from_node": previous},
            )

            del self.in_flight[robot_id]
            finished.append(robot_id)

        return finished


    def _release_shared_area_if_empty(self):
        """
        Keep admission while entering, traversing, or leaving
        the shared zone. Release only after the exit is complete.
        """
        for area in self.areas:
            occupants = self.shared_robot_ids_by_area[area.area_id]

            for robot_id in list(occupants):
                robot = self.active.get(robot_id)

                if robot is None:
                    occupants.remove(robot_id)
                    self._sync_shared_robot_id()
                    continue

                # Still inside the shared zone.
                if robot.current_node in area.conflict_zone_nodes:
                    continue

                # Movement into or out of the shared zone
                # has started but has not yet completed.
                trip = self.in_flight.get(robot_id)
                if trip is not None:
                    if (
                        trip["from_node"] in area.conflict_zone_nodes
                        or trip["to_node"] in area.conflict_zone_nodes
                    ):
                        continue

                # Robot has fully left the shared zone.
                occupants.remove(robot_id)
                self._sync_shared_robot_id()
                robot.status = "active"

                self._event(
                    "shared_area_released",
                    robot,
                    node=robot.current_node,
                    metadata={"area_id": area.area_id},
                )

    def _check_safety(self):
        source_nodes = [
            robot.current_node
            for robot in self.active.values()
            if robot.current_node is not None
        ]

        destination_nodes = list(self._reserved_nodes())

        # Initial staging can contain stacked robots in the
        # original model, so only shared-zone cells are checked
        # for source occupancy violations.
        for area in self.areas:
            shared_sources = [
                node for node in source_nodes
                if node in area.conflict_zone_nodes
            ]

            if any(v > 1 for v in Counter(shared_sources).values()):
                self.safety_violations.append(
                    (self.step_count, "shared_cell_overlap")
                )

            if len(self.shared_robot_ids_by_area[area.area_id]) > (
                self.shared_area_capacity
            ):
                self.safety_violations.append(
                    (self.step_count, "capacity_exceeded")
                )

        reserved_counts = Counter(destination_nodes)

        if any(v > 1 for v in reserved_counts.values()):
            self.safety_violations.append(
                (self.step_count, "duplicate_reservation")
            )

        if set(destination_nodes) & set(source_nodes):
            self.safety_violations.append(
                (self.step_count, "reserved_occupied_cell")
            )

    def step(self):
        spawned = self._spawn_due_robots()

        self._enqueue_waiting_robots()
        admitted = self._admit_next_if_possible()

        proposals = super()._build_proposals()

        # Moving robots cannot initiate another movement.
        for robot_id in self.in_flight:
            if robot_id in proposals:
                proposals[robot_id]["allowed"] = False
                proposals[robot_id]["reason"] = "in_transit"

        started = self._start_movements(proposals)

        self._check_safety()

        # One simulation tick elapses.
        next_tick = self.step_count + 1

        arrived = self._finish_movements(next_tick)

        self._release_shared_area_if_empty()
        self._complete_finished()

        self._record_shared_occupancy()

        result = {
            "step": self.step_count,
            "spawned": spawned,
            "admitted": admitted,
            "started": started,
            "arrived": arrived,
            "completed_count": len(self.completed),
        }

        self.step_log.append(result)
        self.step_count = next_tick

        return result

    def metrics(self):
        result = super().metrics()

        dt = self.physical.simulation_dt_s

        result["total_time_seconds"] = (
            self.step_count * dt
        )
        result["avg_wait_seconds"] = (
            result["avg_wait_time"] * dt
        )
        result["max_wait_seconds"] = (
            result["max_wait_time"] * dt
        )

        result["speed_mps"] = self.physical.speed_mps
        result["cell_size_m"] = self.physical.cell_size_m
        result["turn_delay_s"] = self.physical.turn_delay_s

        result["in_flight_count"] = len(self.in_flight)
        result["safety_violations"] = len(self.safety_violations)

        result["travel_time_seconds_by_agv"] = {
            k: v * dt
            for k, v in self.travel_ticks.items()
        }

        return result
