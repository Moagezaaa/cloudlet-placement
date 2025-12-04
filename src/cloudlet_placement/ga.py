import random
import math
from typing import List, Tuple, Optional
from .problem import CloudletPlacementProblem, CandidatePoint
from .solution import PlacementSolution


class HybridGeneticAlgorithm:
    def __init__(self, problem: CloudletPlacementProblem,
                 population_size: int = 50,
                 generations: int = 200,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 alpha: float = 0.5,
                 use_sa: bool = True):
        self.problem = problem
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.use_sa = use_sa

        self.population: List[PlacementSolution] = []
        self.best_solution: Optional[PlacementSolution] = None
        self.fitness_history: List[float] = []
        self.cost_history: List[float] = []
        self.latency_history: List[float] = []

    def initialize_population(self):
        self.population = [self.create_random_solution() for _ in range(self.population_size)]
        self.update_best_solution()

    def create_random_solution(self) -> PlacementSolution:
        solution = PlacementSolution(self.problem)

        num_cloudlets = random.randint(1, min(len(self.problem.candidate_points), len(self.problem.cloudlet_types) * 2))
        selected_candidates = random.sample(self.problem.candidate_points, num_cloudlets)

        for candidate in selected_candidates:
            cloudlet_type = random.choice(self.problem.cloudlet_types)
            solution.place_cloudlet(candidate.id, cloudlet_type.id)

        unassigned_devices = list(self.problem.devices)
        random.shuffle(unassigned_devices)

        for device in unassigned_devices:
            feasible = []
            for candidate in self.problem.candidate_points:
                if solution.placement_map[candidate.id] is not None:
                    ct_id = solution.placement_map[candidate.id]
                    ct = self.problem.cloudlet_types[ct_id]
                    distance = self.problem.get_distance(device.id, candidate.id)
                    if distance <= ct.coverage_radius:
                        feasible.append(candidate)
            if feasible:
                best = min(feasible, key=lambda c: self.problem.get_distance(device.id, c.id))
                solution.assign_device(device.id, best.id)

        solution.evaluate(self.alpha)
        if not solution.is_feasible:
            solution = self.repair_solution(solution)
        return solution

    def repair_solution(self, solution: PlacementSolution) -> PlacementSolution:
        repaired = solution.clone()
        unassigned = [d for d in self.problem.devices if repaired.assignment_map[d.id] is None]

        for device in unassigned:
            candidate_with_cloudlets = [c for c in self.problem.candidate_points if repaired.placement_map[c.id] is not None]
            feasible = []
            for candidate in candidate_with_cloudlets:
                ct_id = repaired.placement_map[candidate.id]
                ct = self.problem.cloudlet_types[ct_id]
                distance = self.problem.get_distance(device.id, candidate.id)
                if distance <= ct.coverage_radius:
                    assigned = repaired.get_devices_assigned_to(candidate.id)
                    if (sum(d.cpu_demand for d in assigned) + device.cpu_demand <= ct.cpu_capacity and
                        sum(d.memory_demand for d in assigned) + device.memory_demand <= ct.memory_capacity and
                        sum(d.storage_demand for d in assigned) + device.storage_demand <= ct.storage_capacity):
                        feasible.append(candidate)
            if feasible:
                best = min(feasible, key=lambda c: self.problem.get_distance(device.id, c.id))
                repaired.assign_device(device.id, best.id)
            else:
                for candidate in self.problem.candidate_points:
                    if repaired.placement_map[candidate.id] is None:
                        for ct in self.problem.cloudlet_types:
                            distance = self.problem.get_distance(device.id, candidate.id)
                            if distance <= ct.coverage_radius:
                                repaired.place_cloudlet(candidate.id, ct.id)
                                repaired.assign_device(device.id, candidate.id)
                                break

        repaired.evaluate(self.alpha)
        return repaired

    def tournament_selection(self, tournament_size: int = 3) -> PlacementSolution:
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key=lambda s: s.fitness)

    def crossover(self, parent1: PlacementSolution, parent2: PlacementSolution) -> Tuple[PlacementSolution, PlacementSolution]:
        child1 = parent1.clone()
        child2 = parent2.clone()
        if random.random() < self.crossover_rate:
            keys = list(parent1.placement_map.keys())
            if len(keys) >= 2:
                p1 = random.randint(0, len(keys) - 2)
                p2 = random.randint(p1 + 1, len(keys) - 1)
                for i in range(p1, p2 + 1):
                    k = keys[i]
                    child1.placement_map[k], child2.placement_map[k] = parent2.placement_map[k], parent1.placement_map[k]
                child1 = self.reassign_devices(child1)
                child2 = self.reassign_devices(child2)
        child1.evaluate(self.alpha)
        child2.evaluate(self.alpha)
        return child1, child2

    def reassign_devices(self, solution: PlacementSolution) -> PlacementSolution:
        new_solution = solution.clone()
        for device_id in new_solution.assignment_map:
            new_solution.assignment_map[device_id] = None
        for device in self.problem.devices:
            best_cand = None
            best_dist = float('inf')
            for candidate in self.problem.candidate_points:
                if new_solution.placement_map[candidate.id] is not None:
                    ct_id = new_solution.placement_map[candidate.id]
                    ct = self.problem.cloudlet_types[ct_id]
                    dist = self.problem.get_distance(device.id, candidate.id)
                    if dist <= ct.coverage_radius:
                        assigned = new_solution.get_devices_assigned_to(candidate.id)
                        if (sum(d.cpu_demand for d in assigned) + device.cpu_demand <= ct.cpu_capacity and
                            sum(d.memory_demand for d in assigned) + device.memory_demand <= ct.memory_capacity and
                            sum(d.storage_demand for d in assigned) + device.storage_demand <= ct.storage_capacity):
                            if dist < best_dist:
                                best_dist = dist
                                best_cand = candidate.id
            if best_cand is not None:
                new_solution.assign_device(device.id, best_cand)
        return new_solution

    def mutate(self, solution: PlacementSolution) -> PlacementSolution:
        mutated = solution.clone()
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['add', 'remove', 'change', 'swap'])
            if mutation_type == 'add':
                unused = [c for c in self.problem.candidate_points if mutated.placement_map[c.id] is None]
                if unused:
                    c = random.choice(unused)
                    ct = random.choice(self.problem.cloudlet_types)
                    mutated.place_cloudlet(c.id, ct.id)
            elif mutation_type == 'remove':
                used = [cid for cid, ct in mutated.placement_map.items() if ct is not None]
                if used:
                    cid = random.choice(used)
                    mutated.remove_cloudlet(cid)
            elif mutation_type == 'change':
                used = [cid for cid, ct in mutated.placement_map.items() if ct is not None]
                if used:
                    cid = random.choice(used)
                    current = mutated.placement_map[cid]
                    avail = [ct.id for ct in self.problem.cloudlet_types if ct.id != current]
                    if avail:
                        mutated.place_cloudlet(cid, random.choice(avail))
            elif mutation_type == 'swap':
                used = [cid for cid, ct in mutated.placement_map.items() if ct is not None]
                if len(used) >= 2:
                    c1, c2 = random.sample(used, 2)
                    t1, t2 = mutated.placement_map[c1], mutated.placement_map[c2]
                    mutated.placement_map[c1], mutated.placement_map[c2] = t2, t1
            mutated = self.reassign_devices(mutated)
        mutated.evaluate(self.alpha)
        return mutated

    def simulated_annealing_improvement(self, solution: PlacementSolution, temperature: float) -> PlacementSolution:
        if not self.use_sa or temperature < 0.1:
            return solution
        current = solution.clone()
        current_fitness = current.fitness
        for _ in range(5):
            neighbor = self.mutate(current)
            if neighbor.fitness < current_fitness:
                current = neighbor
                current_fitness = neighbor.fitness
            else:
                delta = neighbor.fitness - current_fitness
                if random.random() < math.exp(-delta / temperature):
                    current = neighbor
                    current_fitness = neighbor.fitness
        return current

    def update_best_solution(self):
        feasible = [s for s in self.population if s.is_feasible]
        if feasible:
            current_best = min(feasible, key=lambda s: s.fitness)
            if self.best_solution is None or current_best.fitness < self.best_solution.fitness:
                self.best_solution = current_best.clone()

    def evolve(self):
        print("Starting Hybrid Genetic Algorithm...")
        print(f"Population size: {self.population_size}, Generations: {self.generations}")
        print(f"Crossover rate: {self.crossover_rate}, Mutation rate: {self.mutation_rate}")
        print(f"Alpha (cost weight): {self.alpha}")
        self.initialize_population()
        for generation in range(self.generations):
            temperature = 1.0 - (generation / self.generations)
            new_population = []
            elite_size = max(2, self.population_size // 10)
            sorted_pop = sorted([s for s in self.population if s.is_feasible], key=lambda s: s.fitness)
            new_population.extend(sorted_pop[:elite_size])
            while len(new_population) < self.population_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                if self.use_sa:
                    c1 = self.simulated_annealing_improvement(c1, temperature)
                    c2 = self.simulated_annealing_improvement(c2, temperature)
                if not c1.is_feasible:
                    c1 = self.repair_solution(c1)
                if not c2.is_feasible:
                    c2 = self.repair_solution(c2)
                new_population.extend([c1, c2])
            self.population = new_population[:self.population_size]
            self.update_best_solution()
            feasible = [s for s in self.population if s.is_feasible]
            if feasible:
                best_fitness = min(s.fitness for s in feasible)
                self.fitness_history.append(best_fitness)
                if self.best_solution:
                    self.cost_history.append(self.best_solution.total_cost)
                    self.latency_history.append(self.best_solution.total_latency)
            else:
                self.fitness_history.append(float('inf'))
                self.cost_history.append(float('inf'))
                self.latency_history.append(float('inf'))
            if generation % 20 == 0 or generation == self.generations - 1:
                print(f"Generation {generation:3d}: Best Fitness = {self.fitness_history[-1]:.6f}, Cost = {self.cost_history[-1]:.2f}, Latency = {self.latency_history[-1]:.2f}")
        print("\nOptimization completed!")
        if self.best_solution:
            print(f"\nBest Solution Found:")
            print(f"Total Cost: ${self.best_solution.total_cost:.2f}")
            print(f"Total Latency: {self.best_solution.total_latency:.2f}")
            print(f"Fitness: {self.best_solution.fitness:.6f}")
            print(f"Feasible: {self.best_solution.is_feasible}")
            num_cloudlets = sum(1 for ct in self.best_solution.placement_map.values() if ct is not None)
            print(f"Number of Cloudlets Placed: {num_cloudlets}")
            unassigned = sum(1 for a in self.best_solution.assignment_map.values() if a is None)
            print(f"Unassigned Devices: {unassigned}")
        return self.best_solution
