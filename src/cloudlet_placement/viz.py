import matplotlib.pyplot as plt
import numpy as np
from .problem import CloudletPlacementProblem
from .solution import PlacementSolution


class SolutionVisualizer:
    @staticmethod
    def plot_solution(problem: CloudletPlacementProblem, solution: PlacementSolution, title: str = "Cloudlet Placement Solution"):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax1 = axes[0]
        device_x = [d.x for d in problem.devices]
        device_y = [d.y for d in problem.devices]
        ax1.scatter(device_x, device_y, c='blue', alpha=0.6, s=20, label='Devices', edgecolors='black', linewidth=0.5)
        candidate_x = [c.x for c in problem.candidate_points]
        candidate_y = [c.y for c in problem.candidate_points]
        ax1.scatter(candidate_x, candidate_y, c='gray', alpha=0.3, s=50, label='Candidate Points', marker='s')

        placed = []
        colors = ['red', 'green', 'purple', 'orange', 'brown']
        for candidate in problem.candidate_points:
            if solution.placement_map[candidate.id] is not None:
                placed.append(candidate)
                ct_id = solution.placement_map[candidate.id]
                ct = problem.cloudlet_types[ct_id]
                color = colors[ct_id % len(colors)]
                ax1.scatter(candidate.x, candidate.y, c=color, s=200, marker='*', edgecolors='black', linewidth=1.5, zorder=5)
                circle = plt.Circle((candidate.x, candidate.y), ct.coverage_radius, color=color, alpha=0.1, fill=True)
                ax1.add_patch(circle)

        for device in problem.devices:
            candidate_id = solution.assignment_map.get(device.id)
            if candidate_id is not None:
                candidate = next(c for c in problem.candidate_points if c.id == candidate_id)
                ax1.plot([device.x, candidate.x], [device.y, candidate.y], 'gray', alpha=0.2, linewidth=0.5)

        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title(title)
        ax1.legend(loc='upper right', fontsize='small')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')

        ax2 = axes[1]
        if placed:
            labels = []
            cpu_u = []
            mem_u = []
            stor_u = []
            for candidate in placed:
                ct_id = solution.placement_map[candidate.id]
                ct = problem.cloudlet_types[ct_id]
                assigned = solution.get_devices_assigned_to(candidate.id)
                total_cpu = sum(d.cpu_demand for d in assigned)
                total_mem = sum(d.memory_demand for d in assigned)
                total_stor = sum(d.storage_demand for d in assigned)
                labels.append(f'C{candidate.id}')
                cpu_u.append((total_cpu / ct.cpu_capacity) * 100)
                mem_u.append((total_mem / ct.memory_capacity) * 100)
                stor_u.append((total_stor / ct.storage_capacity) * 100)
            x = np.arange(len(labels))
            w = 0.25
            ax2.bar(x - w, cpu_u, w, label='CPU %', color='red', alpha=0.7)
            ax2.bar(x, mem_u, w, label='Memory %', color='green', alpha=0.7)
            ax2.bar(x + w, stor_u, w, label='Storage %', color='blue', alpha=0.7)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.set_ylim(0, 110)
            ax2.legend()

        plt.tight_layout()
        filename = 'cloudlet_placement_solution.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close()

    @staticmethod
    def plot_convergence(algorithm):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax1 = axes[0, 0]
        ax1.plot(algorithm.fitness_history, 'b-', linewidth=2)
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Best Fitness')
        ax1.set_title('Fitness Convergence')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(algorithm.cost_history, 'r-', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Total Cost')
        ax2.set_title('Cost Convergence')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(algorithm.latency_history, 'g-', linewidth=2)
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Total Latency')
        ax3.set_title('Latency Convergence')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.scatter(algorithm.cost_history, algorithm.latency_history, c=range(len(algorithm.cost_history)), cmap='viridis', alpha=0.6)
        ax4.set_xlabel('Total Cost')
        ax4.set_ylabel('Total Latency')
        ax4.set_title('Cost vs Latency Trade-off')
        ax4.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap='viridis')
        sm.set_array([])
        plt.colorbar(sm, ax=ax4).set_label('Generation')

        plt.tight_layout()
        filename = 'cloudlet_convergence.png'
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.close()
