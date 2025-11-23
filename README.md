# TSP Integrality Gap Analysis Project

## Overview
This project investigates the cost of the convexity problem in the Traveling Salesman Problem (TSP) by analyzing the interaction between mathematical formulations and cost structures.  
Test data is generated using `data_generation.py`.

The main focus is on the **integrality gap**: the difference between the optimal integer programming (IP) solution and the optimal linear programming (LP) relaxation solution for various TSP formulations and cost structures.

We use 4 types of structures: Grid City, Random Euclidean, Clustered and Hub-and-Spoke. You can see what does this mean in the graph below.

**Example destination distributions:**
![Sample Instances](graphs/sample_instances.png)

Each structure may imply different CV (Coefficient of Variation), you can see the distribution of CV of each structure type we use during the test.

**CV for each data type:**
![CV Validation](graphs/cv_validation.png)

## Play with it online
Visualize it here: [Go to the website](https://dba5103.pythonanywhere.com/)

You can also run the server on your own computer, see [How to start the toy website](#how-to-start-the-toy-website)

## How to run the program
1. **Create a Python virtual environment (recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate test data and graphs:**
   ```bash
   python data_generation.py
   ```

After running, you will find:
- **CSV files** containing test data in the `data/` directory.
- **Graphs** showing CV for each structure type and 2 examples of each structure in the `graphs/` directory.

## How to run the models and generate benchmark data for IP and LP

Run the following scripts to generate results for each TSP formulation:

```bash
python run_dfj_with_testdata.py   # DFJ formulation
python run_gg_with_testdata.py    # Gavish-Graves formulation
python run_mtz_with_testdata.py   # MTZ formulation
python run_ap_with_testdata.py    # Assignment Problem relaxation
```

**Note:** For MTZ and Assignment Problem, we use the DFJ formulation with lazy constraints to compute the IP optimal value (true TSP optimum). This is mathematically equivalent to solving the native IP for each formulation, but significantly faster. The LP relaxation is computed using each formulation's own relaxation.

Results will be saved in the `results/` folder for further analysis and visualization.

## How to analyse the results with notebook
You should now be able to use `analyze_results.ipynb` to see the final result. The analysis covers problem sizes n=15, 18, and 20 across all formulations.

## How to start the toy website
This is very easy, don't get scared by the word `web app`

First run the python script to precompute the TSP solutions, since it would require gurabi license, it is better to run in local and store the result for showing on the UI

```bash
python precompute_tours.py 
```

This would generate the best TSP solution in `results/optimal_tours.csv`

Now you can spin up the flask website by simply running

```bash
python app.py
```
