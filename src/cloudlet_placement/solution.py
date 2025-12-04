from typing import List, Dict, Optional
from .problem import Device, CloudletType, CandidatePoint, CloudletPlacementProblem
import math


class PlacementSolution:
    def __init__(self, problem: CloudletPlacementProblem):
        self.problem = problem
        self.placement_map: Dict[int, Optional[int]] = {c.id: None for c in problem.candidate_points}
        self.assignment_map: Dict[int, Optional[int]] = {d.id: None for d in problem.devices}
        self.total_cost: float = 0.0
        self.total_latency: float = 0.0
        self.fitness: float = 0.0
        self.is_feasible: bool = False

    def clone(self):
        new = PlacementSolution(self.problem)
        new.placement_map = self.placement_map.copy()
        new.assignment_map = self.assignment_map.copy()
        new.total_cost = self.total_cost
        new.total_latency = self.total_latency
        new.fitness = self.fitness
        new.is_feasible = self.is_feasible
        return new

    def place_cloudlet(self, candidate_id: int, cloudlet_type_id: int):
        if candidate_id in self.placement_map:
            self.placement_map[candidate_id] = cloudlet_type_id

    def remove_cloudlet(self, candidate_id: int):
        if candidate_id in self.placement_map:
            self.placement_map[candidate_id] = None
            for device_id, assigned in list(self.assignment_map.items()):
                if assigned == candidate_id:
                    self.assignment_map[device_id] = None

    def assign_device(self, device_id: int, candidate_id: int):
        if candidate_id in self.placement_map and self.placement_map[candidate_id] is not None:
            self.assignment_map[device_id] = candidate_id

    def get_devices_assigned_to(self, candidate_id: int) -> List[Device]:
        ids = [d_id for d_id, c_id in self.assignment_map.items() if c_id == candidate_id]
        return [d for d in self.problem.devices if d.id in ids]

    def evaluate(self, alpha: float = 0.5):
        self.total_cost = 0.0
        self.total_latency = 0.0
        self.is_feasible = True
        resource_usage = {}

        for device_id, candidate_id in self.assignment_map.items():
            if candidate_id is None or self.placement_map.get(candidate_id) is None:
                self.is_feasible = False

        for candidate_id, cloudlet_type_id in self.placement_map.items():
            if cloudlet_type_id is not None:
                ct = self.problem.cloudlet_types[cloudlet_type_id]
                cp = next(c for c in self.problem.candidate_points if c.id == candidate_id)
                self.total_cost += ct.base_cost * cp.placement_cost_factor
                resource_usage[candidate_id] = {'cpu': 0.0, 'memory': 0.0, 'storage': 0.0}

        for device in self.problem.devices:
            candidate_id = self.assignment_map.get(device.id)
            if candidate_id is None:
                self.is_feasible = False
                continue

            cloudlet_type_id = self.placement_map.get(candidate_id)
            if cloudlet_type_id is None:
                self.is_feasible = False
                continue

            cloudlet_type = self.problem.cloudlet_types[cloudlet_type_id]
            distance = self.problem.get_distance(device.id, candidate_id)
            if distance > cloudlet_type.coverage_radius:
                self.is_feasible = False
                continue

            resource_usage[candidate_id]['cpu'] += device.cpu_demand
            resource_usage[candidate_id]['memory'] += device.memory_demand
            resource_usage[candidate_id]['storage'] += device.storage_demand
            self.total_latency += distance

        for candidate_id, usage in resource_usage.items():
            cloudlet_type_id = self.placement_map[candidate_id]
            cloudlet_type = self.problem.cloudlet_types[cloudlet_type_id]
            if (usage['cpu'] > cloudlet_type.cpu_capacity or
                usage['memory'] > cloudlet_type.memory_capacity or
                usage['storage'] > cloudlet_type.storage_capacity):
                self.is_feasible = False
                break

        if None in self.assignment_map.values():
            self.is_feasible = False

        if self.is_feasible:
            max_possible_cost = sum(ct.base_cost * 1.2 for ct in self.problem.cloudlet_types)
            max_possible_latency = len(self.problem.devices) * 1000 * math.sqrt(2)
            normalized_cost = self.total_cost / max_possible_cost
            normalized_latency = self.total_latency / max_possible_latency
            self.fitness = alpha * normalized_cost + (1 - alpha) * normalized_latency
        else:
            self.fitness = float('inf')

        return self.fitness
