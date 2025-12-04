# Cloudlet Placement (Hybrid GA)

A compact implementation of a hybrid Genetic Algorithm (GA) with Simulated Annealing enhancements for cloudlet placement in edge computing scenarios. The project implements problem modeling, evolutionary operators, evaluation, and visualization to explore near-optimal placement of cloudlets under different constraints.

**Repository structure**
- `src/cloudlet_placement/problem.py`: Problem definition and data structures
- `src/cloudlet_placement/solution.py`: Solution representation and evaluation
- `src/cloudlet_placement/ga.py`: Hybrid Genetic Algorithm implementation
- `src/cloudlet_placement/viz.py`: Visualization and plotting helpers
- `src/cloudlet_placement/runner.py`: Main execution logic and CLI
- `main.py`: Lightweight entry point / experiment launcher

**Features**
- Configurable GA parameters (population size, generations, mutation/crossover rates)
- Hybridization with Simulated Annealing for local refinement
- Visualization of convergence and final placement
- CLI flags for demos and benchmarks

**Requirements**
- Python 3.10+ (recommended)
- See `requirements.txt` for exact versions. Typical dependencies: `numpy`, `matplotlib`, `seaborn`.

Installation
```
# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Quick usage
```
# Run with default parameters
python main.py

# Fast demo (shorter run)
python main.py --demo

# Custom GA parameters: population and generations
python main.py --pop 30 --gen 50

# Run predefined benchmarks
python main.py --benchmark
```

Running (detailed)
- Activate virtual environment and run the main script (preferred):
```
# create + activate venv (if not already)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run with virtualenv python (avoids activation issues in scripts)
.venv/bin/python3 main.py
```
- Useful flags and examples:
```
# Run with specific population and generations
.venv/bin/python3 main.py --pop 100 --gen 200

# Quick demo run (short)
.venv/bin/python3 main.py --demo

# Run benchmark mode (if available)
.venv/bin/python3 main.py --benchmark
```

Outputs and where to find them
- During a run the program prints progress to stdout and saves plot images in the working directory by default. Typical files created:
  - `cloudlet_convergence.png` — GA convergence plot
  - `cloudlet_placement_solution.png` — visualization of the chosen placement
- To inspect results programmatically, open the generated PNGs or modify `src/cloudlet_placement/viz.py` to change outputs or formats.

CLI notes
- Use `--help` to list available flags and options: `python main.py --help`
- `--demo` runs a quick configuration useful for testing and development

Development
- The core logic is under `src/cloudlet_placement/`. To run experiments or modify operators, edit `ga.py` and `solution.py`.

Contributing
- Feel free to open issues or pull requests. For changes:
  1. Fork the repository
  2. Create a topic branch
  3. Run existing examples locally and include a short description of your change

License
- This repository does not include an explicit license file. Add one if you plan to publish or accept contributions.

Contact
- For questions or collaboration, open an issue or reach out via the repository's GitHub.

