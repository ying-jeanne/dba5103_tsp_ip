import os
import pandas as pd
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

DATA_DIR = 'data'
RESULTS_DIR = 'results'

# Load optimal tours
tours_df = pd.DataFrame()
try:
    tours_path = os.path.join(RESULTS_DIR, 'optimal_tours.csv')
    if os.path.exists(tours_path):
        tours_df = pd.read_csv(tours_path)
except Exception as e:
    print(f"Error loading optimal tours: {e}")

# Load metadata
try:
    metadata_path = os.path.join(DATA_DIR, 'metadata.csv')
    metadata_df = pd.read_csv(metadata_path)
except Exception as e:
    print(f"Error loading metadata: {e}")
    metadata_df = pd.DataFrame()

# Load results
results_data = {}
result_files = {
    'DFJ': 'dfj_results.csv',
    'MTZ': 'mtz_results.csv',
    'Gavish-Graves': 'gg_results.csv',
    'Assignment': 'assignment_results.csv'
}

for algo, filename in result_files.items():
    try:
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            results_data[algo] = pd.read_csv(path)
    except Exception as e:
        print(f"Error loading {algo} results: {e}")

@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-eval' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/structures')
def get_structures():
    if metadata_df.empty:
        return jsonify([])
    structures = metadata_df['structure'].unique().tolist()
    return jsonify(structures)

@app.route('/api/random_instance/<structure_type>')
def get_random_instance(structure_type):
    if metadata_df.empty:
        return jsonify({'error': 'No data available'}), 404
    
    filtered = metadata_df[metadata_df['structure'] == structure_type]
    if filtered.empty:
        return jsonify({'error': 'Structure not found'}), 404
    
    # Randomly select one
    instance = filtered.sample(1).iloc[0]
    
    # Construct a unique ID (e.g., using the index or a combination of fields)
    # The metadata has 'instance' column which is the index (0-9)
    # We can use the row index or create a composite ID.
    # Let's use the row index from the original dataframe to easily retrieve it later,
    # or just pass the necessary info.
    # Actually, let's pass the file names directly or a composite ID.
    # Let's use: n{n}_structure_{structure}_i{instance}
    # But wait, the file names are already in metadata: coord_file and dist_file.
    # Let's return the full row as dict.
    
    return jsonify(instance.to_dict())

@app.route('/api/instance_data')
def get_instance_data():
    # Expects query params: coord_file, dist_file, n, structure, instance_idx
    coord_file = request.args.get('coord_file')
    dist_file = request.args.get('dist_file')
    n = request.args.get('n', type=int)
    structure = request.args.get('structure')
    instance_idx = request.args.get('instance_idx', type=int)
    
    if not coord_file:
        return jsonify({'error': 'Missing coord_file parameter'}), 400
        
    # Load coordinates
    try:
        coord_path = os.path.join(DATA_DIR, coord_file)
        coords_df = pd.read_csv(coord_path)
        coords = coords_df.to_dict(orient='records')
    except Exception as e:
        return jsonify({'error': f"Error loading coordinates: {e}"}), 500

    # Get optimal tour
    tour_indices = []
    if not tours_df.empty and dist_file:
        tour_match = tours_df[tours_df['dist_file'] == dist_file]
        if not tour_match.empty:
            # Parse string representation of list
            import ast
            try:
                tour_indices = ast.literal_eval(tour_match.iloc[0]['tour_indices'])
            except:
                pass

    instance_results = []
    
    for algo, df in results_data.items():
        match = pd.DataFrame()
        
        try:
            if algo in ['DFJ', 'Gavish-Graves']:
                # These have 'instance' column matching the dist_file name
                if 'instance' in df.columns and dist_file:
                    match = df[df['instance'] == dist_file]
            
            elif algo == 'MTZ':
                # Matches on n, structure, instance_idx
                if all(col in df.columns for col in ['n', 'structure', 'instance_idx']):
                    match = df[
                        (df['n'] == n) & 
                        (df['structure'] == structure) & 
                        (df['instance_idx'] == instance_idx)
                    ]
            
            elif algo == 'Assignment':
                # Matches on n_cities, structure, instance
                if all(col in df.columns for col in ['n_cities', 'structure', 'instance']):
                    match = df[
                        (df['n_cities'] == n) & 
                        (df['structure'] == structure) & 
                        (df['instance'] == instance_idx)
                    ]
            
            if not match.empty:
                row = match.iloc[0]
                
                # Normalize result fields
                ip_obj = row.get('IP_obj') or row.get('Z_IP')
                lp_obj = row.get('LP_obj') or row.get('Z_AP')
                
                gap_abs = row.get('gap_absolute')
                gap_pct = row.get('gap_percent') or row.get('Gap_Percent')
                
                # Special handling for MTZ: 'gap_absolute' in CSV is actually percentage
                if algo == 'MTZ':
                    gap_pct = row.get('gap_absolute')
                    if ip_obj is not None and lp_obj is not None:
                        gap_abs = ip_obj - lp_obj
                    else:
                        gap_abs = None

                res = {
                    'algorithm': algo,
                    'ip_obj': ip_obj,
                    'lp_obj': lp_obj,
                    'gap_absolute': gap_abs,
                    'gap_percent': gap_pct,
                    'solve_time': row.get('total_solve_time') or row.get('lp_solve_time')
                }
                instance_results.append(res)
                
        except Exception as e:
            print(f"Error processing {algo}: {e}")
            continue
    
    return jsonify({
        'coordinates': coords,
        'results': instance_results,
        'tour': tour_indices
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
