"""
Run MTZ experiments on all TSP instances
Generates CSV file with raw data for analysis in Jupyter notebook
"""
from pathlib import Path
import pandas as pd
from tsp_mtz_solver import solve_mtz
from tsp_dfj_solver import solve_dfj_ip

def run_all_experiments(data_folder=Path(__file__).parent.parent / "data", sizes=[15, 18, 20]):
    structures = ['grid', 'random', 'clustered', 'hub_spoke']
    all_results = []

    for n in sizes:
        for structure in structures:
            for instance_idx in range(10):
                dist_file = data_folder / f"dist_n{n}_{structure}_i{instance_idx}.csv"
                distance_matrix = pd.read_csv(dist_file, header=None).values

                # Solve MTZ LP relaxation (fast)
                lp_obj, _ = solve_mtz(distance_matrix, is_ip=False)
                # Solve IP using DFJ (much faster than MTZ IP)
                # Note: DFJ_IP = MTZ_IP = true TSP optimum
                ip_obj = solve_dfj_ip(distance_matrix)

                gap_absolute = ip_obj - lp_obj if ip_obj is not None else None
                gap_percent = (gap_absolute / ip_obj * 100) if ip_obj else None

                all_results.append({
                    "n": n,
                    "structure": structure,
                    "instance_idx": instance_idx,
                    "IP_obj": ip_obj,
                    "LP_obj": lp_obj,
                    "gap_absolute": gap_absolute,
                    "gap_percent": gap_percent
                })

    df = pd.DataFrame(all_results)
    column_order = ["n", "structure", "instance_idx", "IP_obj", "LP_obj", "gap_absolute", "gap_percent"]
    df = df[column_order]
    return df

if __name__ == "__main__":
    results_df = run_all_experiments(sizes=[15, 18, 20, 100])
    output_file = Path(__file__).parent.parent / "results" / "mtz_results.csv"
    output_file.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"Experiments completed. Results saved to: {output_file}")
    print(f"Total instances: {len(results_df)}")