import subprocess
import sys
import itertools
from datetime import datetime
import os

def run_simulation_batch():
    seeds = list(range(1,41)) # Seeds from 1 to 40
    methods = ["PBCJ", "BBTM"]
    pbcj_selections = ["random", "entropy", "round_robin", "no_repeating_pairs", "KG-Tau", "KG-Rho", "KG-Entropy"] # PBCJ selections
    bbtm_selections = ["random",  "pair_entropy", "round_robin", "no_repeating_pairs", "KG-Tau", "KG-Rho", "KG-Entropy"] # BBTM selections
    n_values = [5, 10, 20, 30, 40, 50]  # Students/items
    k_values = [5, 10, 20, 30, 40, 50]  # k budgets

    # Create output directory
    os.makedirs("output_logs", exist_ok=True)

    for seed, method, n, k in itertools.product(seeds, methods, n_values, k_values):
        selections = pbcj_selections if method == "PBCJ" else bbtm_selections

        for selection in selections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"output_logs/{method}_{selection}_N{n}_K{k}_seed{seed}_{timestamp}.log"

            with open(log_filename, 'w') as log_file:
                subprocess.run(
                    [
                        sys.executable, "main.py",
                        "-seed", str(seed),
                        "-m", method,
                        "-sel", selection,
                        "-n", str(n),
                        "-k", str(k)
                    ],
                    stdout=log_file,  # Save output to log file
                    stderr=subprocess.STDOUT  # Combine stderr with stdout
                )

if __name__ == "__main__":
    run_simulation_batch()