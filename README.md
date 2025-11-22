# TSP Integrality Gap Analysis Project

## Overview
This project investigates the cost of the convexity problem in the Traveling Salesman Problem (TSP) by analyzing the interaction between mathematical formulations and cost structures.  
Test data is generated using `data_generation.py`.

The main focus is on the **integrality gap**: the difference between the optimal integer programming (IP) solution and the optimal linear programming (LP) relaxation solution for various TSP formulations and cost structures.

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
- **Graphs** showing CV for each data type and example destination distributions in the `graphs/` directory.