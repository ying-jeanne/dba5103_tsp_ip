import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from helpers.tsp_dfj_solver import build_dfj_model, subtour_callback, find_subtours

DATA_DIR = 'data'
RESULTS_DIR = 'results'
OUTPUT_FILE = os.path.join(RESULTS_DIR, 'optimal_tours.csv')

def load_distance_matrix(filepath):
    """Load distance matrix from CSV file"""
    df = pd.read_csv(filepath, header=None)
    return df.values

def extract_tour(n, x_vals):
    """
    Extract the single tour from the solution.
    Assumes the solution is a single connected component (valid TSP tour).
    """
    # Build adjacency list
    edges = [(i, j) for (i, j), val in x_vals.items() if val > 0.5]
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
    tour = []
    curr = 0
    visited = [False] * n
    while not visited[curr]:
        visited[curr] = True
        tour.append(curr)
        # Find next node
        if adj[curr]:
            curr = adj[curr][0]
        else:
            break
    return tour

def solve_and_get_tour(dist_matrix_path):
    dist_matrix = load_distance_matrix(dist_matrix_path)
    n = len(dist_matrix)
    
    # Solve IP
    model, x, subtour_list = build_dfj_model(dist_matrix, relaxation=False)
    model.optimize(lambda model, where: subtour_callback(model, where, x, n, subtour_list))
    if model.status == GRB.OPTIMAL:
        # Get solution values
        x_vals = model.getAttr('x', x)
        # Extract tour
        tour = extract_tour(n, x_vals)
        return tour
    else:
        print(f"Failed to solve {dist_matrix_path}")
        return []

def main():
    metadata_path = os.path.join(DATA_DIR, 'metadata.csv')
    if not os.path.exists(metadata_path):
        print("Metadata file not found.")
        return

    metadata_df = pd.read_csv(metadata_path)
    results = []
    print(f"Found {len(metadata_df)} instances. Starting pre-computation...")
    for index, row in metadata_df.iterrows():
        dist_file = row['dist_file']
        dist_path = os.path.join(DATA_DIR, dist_file)
        
        if os.path.exists(dist_path):
            print(f"Solving {dist_file}...")
            tour = solve_and_get_tour(dist_path)
            results.append({
                'dist_file': dist_file,
                'tour_indices': str(tour) # Save as string representation of list
            })
        else:
            print(f"File not found: {dist_path}")
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved optimal tours to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()