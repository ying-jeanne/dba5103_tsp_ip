
"""
Run Assignment Problem (AP) experiments on all TSP instances
Generates CSV file with raw data for analysis
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tsp_ap_solver import solve_ap
from tsp_dfj_solver import solve_dfj_ip

def compute_cv(dist_matrix):
    """Compute coefficient of variation of distance matrix"""
    # Extract non-diagonal elements (actual distances)
    distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    if len(distances) == 0 or np.mean(distances) == 0:
        return 0
    cv = np.std(distances) / np.mean(distances)
    return cv

def run_all_experiments(data_folder=Path(__file__).parent.parent / "data", sizes=[15, 18, 20]):
    structures = ['grid', 'random', 'clustered', 'hub_spoke']
    all_results = []

    for n in sizes:
        for structure in structures:
            for instance_idx in range(10):
                dist_file = data_folder / f"dist_n{n}_{structure}_i{instance_idx}.csv"
                if not dist_file.exists():
                    print(f"Warning: File {dist_file} not found. Skipping.")
                    continue
                    
                df_dist = pd.read_csv(dist_file, header=None)
                distance_matrix = df_dist.values

                # Compute CV
                cv = compute_cv(distance_matrix)

                # Solve AP (Lower Bound)
                ap_obj = solve_ap(distance_matrix)

                # Solve IP (True Optimum) using DFJ solver (much faster than MTZ)
                ip_obj = solve_dfj_ip(distance_matrix)

                gap_absolute = ip_obj - ap_obj if (ip_obj is not None and ap_obj is not None) else None
                gap_percent = (gap_absolute / ip_obj * 100) if (ip_obj and gap_absolute is not None) else None

                z_ap_str = f"{ap_obj:.2f}" if ap_obj is not None else "NaN"
                z_ip_str = f"{ip_obj:.2f}" if ip_obj is not None else "NaN"
                print(f"Solved {structure} (N={n}, ID={instance_idx}). Z_AP={z_ap_str}, Z_IP={z_ip_str}")

                all_results.append({
                    "n": n,
                    "structure": structure,
                    "instance_idx": instance_idx,
                    "cv": cv,
                    "Z_AP": ap_obj,
                    "Z_IP": ip_obj,
                    "gap_absolute": gap_absolute,
                    "gap_percent": gap_percent
                })

    df = pd.DataFrame(all_results)
    # Reorder columns
    column_order = ["n", "structure", "instance_idx", "cv", "Z_AP", "Z_IP", "gap_absolute", "gap_percent"]
    df = df[column_order]
    return df

if __name__ == "__main__":
    # Run experiments on all problem sizes
    results_df = run_all_experiments(sizes=[15, 18, 20])
    output_file = Path(__file__).parent.parent / "results" / "assignment_results.csv"
    output_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Experiments completed. Results saved to: {output_file}")
    print(f"Total instances: {len(results_df)}")
