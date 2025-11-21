"""
TSP Experimental Dataset Generator
Generates distance matrices with controlled coefficient of variation (CV)
for studying formulation-structure interactions in TSP duality gaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os

class TSPDataGenerator:
    """
    Generates TSP instances with four distance structures:
    - Grid City (CV ≈ 0.42): Low variation, regular structure
    - Random Euclidean (CV ≈ 0.46): Medium-low variation, natural baseline
    - Clustered (CV ≈ 0.63): Medium-high variation, bimodal distribution
    - Hub-and-Spoke (CV ≈ 0.75): High variation, systematic structure
    """
    
    def __init__(self, output_dir='tsp_dataset'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Experimental parameters
        self.n_cities_options = [15, 18, 20]
        self.n_instances = 10
        self.structures = ['grid', 'random', 'clustered', 'hub_spoke']
        
        # Realistic CV targets
        self.cv_targets = {
            'grid': 0.42,
            'random': 0.46,
            'clustered': 0.63,
            'hub_spoke': 0.75
        }
    
    def compute_cv(self, dist_matrix):
        """Calculate coefficient of variation"""
        upper_tri = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
        return np.std(upper_tri) / np.mean(upper_tri)
    
    # ==================== STRUCTURE 1: GRID CITY ====================
    def generate_grid(self, n, seed):
        """
        Grid City: Regular grid with Euclidean distance
        Target CV: 0.42
        """
        np.random.seed(seed)
        
        grid_size = int(np.ceil(np.sqrt(n)))
        spacing = 50
        
        x = np.arange(grid_size) * spacing
        y = np.arange(grid_size) * spacing
        xx, yy = np.meshgrid(x, y)
        
        points = np.column_stack([xx.flatten(), yy.flatten()])[:n]
        
        # Convert to float before adding jitter
        points = points.astype(float)
        
        # Minimal jitter to avoid perfect symmetry
        jitter = np.random.normal(0, spacing * 0.001, points.shape)
        points += jitter
        
        dist_matrix = cdist(points, points, metric='euclidean')
        return points, dist_matrix
    
    # ==================== STRUCTURE 2: RANDOM EUCLIDEAN ====================
    def generate_random(self, n, seed):
        """
        Random Euclidean: Uniformly distributed points
        Target CV: 0.46
        """
        np.random.seed(seed)
        points = np.random.uniform(0, 100, (n, 2))
        dist_matrix = cdist(points, points, metric='euclidean')
        return points, dist_matrix
    
    # ==================== STRUCTURE 3: CLUSTERED ====================
    def generate_clustered(self, n, seed, n_clusters=4):
        """
        Clustered: Multiple tight clusters with large inter-cluster distances
        Target CV: 0.63
        """
        np.random.seed(seed)
        
        cities_per_cluster = n // n_clusters
        remainder = n % n_clusters
        
        # Cluster centers far apart
        cluster_centers = np.random.uniform(0, 100, (n_clusters, 2))
        
        points = []
        for i in range(n_clusters):
            n_in_cluster = cities_per_cluster + (1 if i < remainder else 0)
            
            # Tight distribution around center (radius 5)
            angles = np.random.uniform(0, 2*np.pi, n_in_cluster)
            radii = np.random.uniform(0, 5, n_in_cluster)
            
            cluster_points = cluster_centers[i] + np.column_stack([
                radii * np.cos(angles),
                radii * np.sin(angles)
            ])
            points.append(cluster_points)
        
        points = np.vstack(points)
        dist_matrix = cdist(points, points, metric='euclidean')
        return points, dist_matrix
    
    # ==================== STRUCTURE 4: HUB-AND-SPOKE ====================
    def generate_hub_spoke(self, n, seed):
        """
        Hub-and-Spoke: Central hub with radiating spokes
        Spoke-to-spoke distances amplified to create high CV
        Target CV: 0.75
        """
        np.random.seed(seed)
        
        # Central hub
        hub = np.array([[50, 50]])
        n_spokes = n - 1
        
        # Distribute spokes on circle with varying radii
        angles = np.linspace(0, 2*np.pi, n_spokes, endpoint=False)
        
        # Two groups: close and far spokes for maximum variation
        n_close = n_spokes // 2
        radii = np.zeros(n_spokes)
        radii[:n_close] = np.random.uniform(5, 15, n_close)
        radii[n_close:] = np.random.uniform(70, 90, n_spokes - n_close)
        
        spoke_cities = hub + np.column_stack([
            radii * np.cos(angles),
            radii * np.sin(angles)
        ])
        
        points = np.vstack([hub, spoke_cities])
        dist_matrix = cdist(points, points, metric='euclidean')
        
        # Amplify spoke-to-spoke distances (7x)
        for i in range(1, n):
            for j in range(i+1, n):
                dist_matrix[i, j] *= 7.0
                dist_matrix[j, i] *= 7.0
        
        return points, dist_matrix
    
    # ==================== MAIN GENERATION ====================
    def generate_all(self):
        """Generate complete dataset"""
        
        metadata = []
        
        print("="*70)
        print("TSP Dataset Generation")
        print("="*70)
        print(f"Sizes: {self.n_cities_options}")
        print(f"Instances per configuration: {self.n_instances}")
        print(f"Structures: {self.structures}")
        print(f"Total instances: {len(self.n_cities_options) * self.n_instances * len(self.structures)}")
        print("="*70)
        
        for n in self.n_cities_options:
            print(f"\nGenerating n={n}...")
            
            for inst_id in range(self.n_instances):
                seed = n * 1000 + inst_id
                
                for structure in self.structures:
                    # Generate
                    if structure == 'grid':
                        points, dist = self.generate_grid(n, seed)
                    elif structure == 'random':
                        points, dist = self.generate_random(n, seed)
                    elif structure == 'clustered':
                        points, dist = self.generate_clustered(n, seed)
                    elif structure == 'hub_spoke':
                        points, dist = self.generate_hub_spoke(n, seed)
                    
                    cv = self.compute_cv(dist)
                    
                    # Save files
                    dist_file = f"dist_n{n}_{structure}_i{inst_id}.csv"
                    coord_file = f"coord_n{n}_{structure}_i{inst_id}.csv"
                    
                    pd.DataFrame(dist).to_csv(
                        os.path.join(self.output_dir, dist_file),
                        index=False, header=False
                    )
                    
                    pd.DataFrame(points, columns=['x', 'y']).to_csv(
                        os.path.join(self.output_dir, coord_file),
                        index=False
                    )
                    
                    # Record metadata
                    upper_tri = dist[np.triu_indices_from(dist, k=1)]
                    metadata.append({
                        'n_cities': n,
                        'structure': structure,
                        'instance': inst_id,
                        'seed': seed,
                        'cv': cv,
                        'cv_target': self.cv_targets[structure],
                        'mean_dist': np.mean(upper_tri),
                        'std_dist': np.std(upper_tri),
                        'min_dist': np.min(upper_tri),
                        'max_dist': np.max(upper_tri),
                        'dist_file': dist_file,
                        'coord_file': coord_file
                    })
                
                if (inst_id + 1) % 5 == 0:
                    print(f"  Completed {inst_id + 1}/{self.n_instances} instances")
        
        # Save metadata
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(os.path.join(self.output_dir, 'metadata.csv'), index=False)
        
        print("\n" + "="*70)
        print(f"Complete! Files saved to: {self.output_dir}/")
        print("="*70)
        
        return metadata_df
    
    # ==================== VALIDATION ====================
    def validate_cv(self, metadata_df):
        """Check CV achievement"""
        
        print("\n" + "="*70)
        print("CV Validation")
        print("="*70)
        
        cv_summary = metadata_df.groupby('structure')['cv'].agg(['mean', 'std', 'min', 'max'])
        
        for structure in self.structures:
            target = self.cv_targets[structure]
            actual = cv_summary.loc[structure, 'mean']
            std = cv_summary.loc[structure, 'std']
            
            print(f"\n{structure.upper()}")
            print(f"  Target:  {target:.2f}")
            print(f"  Actual:  {actual:.3f} ± {std:.3f}")
            print(f"  Range:   [{cv_summary.loc[structure, 'min']:.3f}, {cv_summary.loc[structure, 'max']:.3f}]")
            
            if abs(actual - target) / target <= 0.15:
                print(f"  ✓ Achieved")
            else:
                print(f"  ⚠ Deviation: {((actual-target)/target)*100:.1f}%")
        
        return cv_summary
    
    # ==================== VISUALIZATION ====================
    def plot_cv_distribution(self, metadata_df):
        """Visualize CV distribution"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Boxplot
        ax1 = axes[0]
        structure_order = ['grid', 'random', 'clustered', 'hub_spoke']
        labels = [f'{s}\n(target {self.cv_targets[s]:.2f})' for s in structure_order]
        
        cv_data = [metadata_df[metadata_df['structure'] == s]['cv'].values 
                   for s in structure_order]
        
        bp = ax1.boxplot(cv_data, tick_labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add target lines
        for i, target in enumerate([self.cv_targets[s] for s in structure_order]):
            ax1.hlines(target, i+0.7, i+1.3, colors='red', linestyles='--', linewidth=2)
        
        ax1.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
        ax1.set_title('CV Distribution by Structure', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Summary table
        ax2 = axes[1]
        ax2.axis('off')
        
        table_data = []
        for s in structure_order:
            data = metadata_df[metadata_df['structure'] == s]
            table_data.append([
                s.replace('_', '-').title(),
                f"{self.cv_targets[s]:.2f}",
                f"{data['cv'].mean():.3f}",
                f"{data['cv'].std():.3f}",
                f"{data['cv'].min():.3f}-{data['cv'].max():.3f}"
            ])
        
        table = ax2.table(
            cellText=table_data,
            colLabels=['Structure', 'Target', 'Mean', 'Std', 'Range'],
            cellLoc='center',
            loc='center',
            colWidths=[0.25, 0.15, 0.15, 0.15, 0.30]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, 5):
            for j in range(5):
                table[(i, j)].set_facecolor(colors[i-1])
        
        ax2.set_title('CV Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.output_dir, 'cv_validation.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {plot_path}")
        plt.show()
    
    def plot_samples(self, metadata_df, n_samples=2):
        """Visualize sample instances"""
        
        print("\nGenerating sample visualizations...")
        
        fig, axes = plt.subplots(4, n_samples, figsize=(12, 16))
        
        for row, structure in enumerate(self.structures):
            samples = metadata_df[
                (metadata_df['structure'] == structure) & 
                (metadata_df['n_cities'] == 20)
            ].head(n_samples)
            
            for col, (_, sample) in enumerate(samples.iterrows()):
                ax = axes[row, col]
                
                coords = pd.read_csv(os.path.join(self.output_dir, sample['coord_file']))
                
                ax.scatter(coords['x'], coords['y'], c='blue', s=100, 
                          alpha=0.6, edgecolors='black', linewidth=1.5)
                
                if structure == 'hub_spoke':
                    ax.scatter(coords.iloc[0]['x'], coords.iloc[0]['y'], 
                             c='red', s=300, marker='*', 
                             edgecolors='black', linewidth=2, label='Hub', zorder=5)
                    ax.legend(loc='upper right')
                
                for i, (x, y) in enumerate(coords.values):
                    ax.annotate(str(i), (x, y), fontsize=8, 
                              ha='center', va='center', color='white', weight='bold')
                
                title = f"{structure.replace('_', '-').title()}\nCV={sample['cv']:.3f}"
                if col == 0:
                    title = f"Instance {sample['instance']}\n" + title
                
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                if col == 0:
                    ax.set_ylabel('Y coordinate', fontsize=10)
                if row == 3:
                    ax.set_xlabel('X coordinate', fontsize=10)
        
        plt.tight_layout()
        
        samples_path = os.path.join(self.output_dir, 'sample_instances.png')
        plt.savefig(samples_path, dpi=300, bbox_inches='tight')
        print(f"Samples saved: {samples_path}")
        plt.show()


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    # Create generator
    generator = TSPDataGenerator(output_dir='tsp_dataset')
    
    # Generate all data
    metadata = generator.generate_all()
    
    # Validate CV targets
    generator.validate_cv(metadata)
    
    # Create visualizations
    generator.plot_cv_distribution(metadata)
    generator.plot_samples(metadata, n_samples=2)
    
    print("\n" + "="*70)
    print("Dataset ready for experimentation!")
    print("="*70)