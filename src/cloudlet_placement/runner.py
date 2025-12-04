import os
import time
import argparse
from .problem import CloudletPlacementProblem
from .ga import HybridGeneticAlgorithm
from .viz import SolutionVisualizer


def main(args=None):
    parser = argparse.ArgumentParser(description='Cloudlet Placement Optimization')
    parser.add_argument('--demo', action='store_true', help='Run in demo mode (fewer iterations, faster)')
    parser.add_argument('--pop', type=int, default=None, help='Population size (default: 50 or 20 in demo)')
    parser.add_argument('--gen', type=int, default=None, help='Number of generations (default: 100 or 30 in demo)')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark on different problem sizes')
    parsed_args = parser.parse_args(args)

    if parsed_args.benchmark:
        benchmark_algorithm()
        return

    print("=" * 70)
    print("CLOUDLET PLACEMENT IN EDGE COMPUTING - HYBRID GENETIC ALGORITHM")
    print("=" * 70)

    print("\n1. Creating problem instance...")
    problem = CloudletPlacementProblem()
    problem.create_random_instance(num_devices=100, num_candidates=20, num_cloudlet_types=3)

    print(f"   Devices: {len(problem.devices)}")
    print(f"   Candidate Points: {len(problem.candidate_points)}")
    print(f"   Cloudlet Types: {len(problem.cloudlet_types)}")

    demo = parsed_args.demo or os.getenv('CLOUDLET_DEMO', '0') == '1'
    if demo:
        alphas = [0.5]
        pop_size = parsed_args.pop if parsed_args.pop else 20
        generations = parsed_args.gen if parsed_args.gen else 30
    else:
        alphas = [0.3, 0.5, 0.7]
        pop_size = parsed_args.pop if parsed_args.pop else 50
        generations = parsed_args.gen if parsed_args.gen else 100

    best_solutions = []
    for alpha in alphas:
        print(f"\n   Running with alpha = {alpha} (cost weight)...")
        ga = HybridGeneticAlgorithm(problem=problem, population_size=pop_size, generations=generations, crossover_rate=0.8, mutation_rate=0.2, alpha=alpha, use_sa=True)
        best_solution = ga.evolve()
        best_solutions.append((alpha, best_solution))
        SolutionVisualizer.plot_convergence(ga)

    print("\n3. Comparing solutions with different weights...")
    print("\n" + "=" * 70)
    print(f"{'Alpha':<10} {'Cost':<15} {'Latency':<15} {'Fitness':<15} {'Cloudlets':<10}")
    print("-" * 70)
    for alpha, solution in best_solutions:
        if solution and solution.is_feasible:
            num_cloudlets = sum(1 for ct in solution.placement_map.values() if ct is not None)
            print(f"{alpha:<10} ${solution.total_cost:<14.2f} {solution.total_latency:<14.2f} {solution.fitness:<14.6f} {num_cloudlets:<10}")

    print("=" * 70)

    print("\n4. Visualizing the best solution...")
    feasible_solutions = [s for _, s in best_solutions if s and s.is_feasible]
    if feasible_solutions:
        best_overall = min(feasible_solutions, key=lambda s: s.total_cost + s.total_latency / 10)
        SolutionVisualizer.plot_solution(problem=problem, solution=best_overall, title=f"Best Cloudlet Placement Solution")

        print("\nDetailed Solution Analysis:")
        print("-" * 50)
        for candidate in problem.candidate_points:
            if best_overall.placement_map[candidate.id] is not None:
                cloudlet_type_id = best_overall.placement_map[candidate.id]
                cloudlet_type = problem.cloudlet_types[cloudlet_type_id]
                assigned_devices = best_overall.get_devices_assigned_to(candidate.id)
                print(f"\nCloudlet at Candidate {candidate.id}:")
                print(f"  Type: {cloudlet_type_id} (CPU: {cloudlet_type.cpu_capacity}GHz, Mem: {cloudlet_type.memory_capacity}GB, Storage: {cloudlet_type.storage_capacity}GB)")
                print(f"  Coverage Radius: {cloudlet_type.coverage_radius}m")
                print(f"  Cost: ${cloudlet_type.base_cost * candidate.placement_cost_factor:.2f}")
                print(f"  Assigned Devices: {len(assigned_devices)}")
                if assigned_devices:
                    total_cpu = sum(d.cpu_demand for d in assigned_devices)
                    total_mem = sum(d.memory_demand for d in assigned_devices)
                    total_stor = sum(d.storage_demand for d in assigned_devices)
                    print(f"  CPU Utilization: {total_cpu:.1f}/{cloudlet_type.cpu_capacity}GHz ({total_cpu/cloudlet_type.cpu_capacity*100:.1f}%)")
                    print(f"  Memory Utilization: {total_mem:.1f}/{cloudlet_type.memory_capacity}GB ({total_mem/cloudlet_type.memory_capacity*100:.1f}%)")
                    print(f"  Storage Utilization: {total_stor:.1f}/{cloudlet_type.storage_capacity}GB ({total_stor/cloudlet_type.storage_capacity*100:.1f}%)")

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def benchmark_algorithm():
    print("\nBenchmarking Algorithm Performance...")
    print("=" * 70)
    problem_sizes = [
        (50, 10, 2),
        (100, 20, 3),
        (200, 30, 3),
    ]
    results = []
    for num_devices, num_candidates, num_types in problem_sizes:
        print(f"\nProblem Size: {num_devices} devices, {num_candidates} candidates, {num_types} cloudlet types")
        problem = CloudletPlacementProblem()
        problem.create_random_instance(num_devices, num_candidates, num_types)
        start_time = time.time()
        ga = HybridGeneticAlgorithm(problem=problem, population_size=30, generations=50, alpha=0.5)
        solution = ga.evolve()
        execution_time = time.time() - start_time
        if solution and solution.is_feasible:
            results.append({'size': f"{num_devices}/{num_candidates}/{num_types}", 'time': execution_time, 'cost': solution.total_cost, 'latency': solution.total_latency, 'fitness': solution.fitness, 'feasible': solution.is_feasible})
            print(f"  Execution Time: {execution_time:.2f} seconds")
            print(f"  Solution Cost: ${solution.total_cost:.2f}")
            print(f"  Solution Latency: {solution.total_latency:.2f}")
            print(f"  Fitness: {solution.fitness:.6f}")
        else:
            print("  No feasible solution found!")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY:")
    print("=" * 70)
    print(f"{'Problem Size':<20} {'Time(s)':<10} {'Cost($)':<12} {'Latency':<12} {'Fitness':<12}")
    print("-" * 70)
    for result in results:
        print(f"{result['size']:<20} {result['time']:<10.2f} {result['cost']:<12.2f} {result['latency']:<12.2f} {result['fitness']:<12.6f}")
