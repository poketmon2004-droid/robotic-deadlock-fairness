from dataclasses import dataclass

@dataclass(frozen=True)
class AGVPhysicalConfig:
    cell_size_m: float = 1.0
    speed_mps: float = 1.0
    turn_delay_s: float = 1.0
    simulation_dt_s: float = 0.1

    def __post_init__(self):
        if self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        if self.speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        if self.turn_delay_s < 0:
            raise ValueError("turn_delay_s cannot be negative")
        if self.simulation_dt_s <= 0:
            raise ValueError("simulation_dt_s must be positive")
