import math
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class Device:
    id: int
    x: float
    y: float
    cpu_demand: float
    memory_demand: float
    storage_demand: float

    def __hash__(self):
        return hash(self.id)


@dataclass
class CloudletType:
    id: int
    cpu_capacity: float
    memory_capacity: float
    storage_capacity: float
    coverage_radius: float
    base_cost: float


@dataclass
class CandidatePoint:
    id: int
    x: float
    y: float
    placement_cost_factor: float


class CloudletPlacementProblem:
    def __init__(self):
        self.devices: List[Device] = []
        self.candidate_points: List[CandidatePoint] = []
        self.cloudlet_types: List[CloudletType] = []
        self.device_to_candidate_distances: Dict[Tuple[int, int], float] = {}

    def calculate_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def precompute_distances(self):
        for device in self.devices:
            for candidate in self.candidate_points:
                dist = self.calculate_distance(device.x, device.y, candidate.x, candidate.y)
                self.device_to_candidate_distances[(device.id, candidate.id)] = dist

    def get_distance(self, device_id: int, candidate_id: int) -> float:
        return self.device_to_candidate_distances.get((device_id, candidate_id), float("inf"))

    def create_random_instance(self, num_devices: int = 100, num_candidates: int = 20, num_cloudlet_types: int = 3):
        import numpy as np

        np.random.seed(42)

        self.devices = []
        for i in range(num_devices):
            device = Device(
                id=i,
                x=np.random.uniform(0, 1000),
                y=np.random.uniform(0, 1000),
                cpu_demand=np.random.uniform(0.1, 2.0),
                memory_demand=np.random.uniform(0.5, 4.0),
                storage_demand=np.random.uniform(1.0, 10.0),
            )
            self.devices.append(device)

        self.candidate_points = []
        for i in range(num_candidates):
            candidate = CandidatePoint(
                id=i,
                x=np.random.uniform(0, 1000),
                y=np.random.uniform(0, 1000),
                placement_cost_factor=np.random.uniform(0.8, 1.2),
            )
            self.candidate_points.append(candidate)

        self.cloudlet_types = []
        type_specs = [
            (4.0, 8.0, 100.0, 150.0, 1000.0),
            (8.0, 16.0, 200.0, 200.0, 1800.0),
            (16.0, 32.0, 500.0, 300.0, 3200.0),
        ]

        for i, (cpu, mem, stor, radius, cost) in enumerate(type_specs[:num_cloudlet_types]):
            self.cloudlet_types.append(
                CloudletType(id=i, cpu_capacity=cpu, memory_capacity=mem, storage_capacity=stor, coverage_radius=radius, base_cost=cost)
            )

        self.precompute_distances()
