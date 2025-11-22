
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

def build_ap_model(distance_matrix):
    """
    Builds the Assignment Problem (AP) model.
    AP is a relaxation of TSP where subtour elimination constraints are removed.
    """
    n = distance_matrix.shape[0]
    model = gp.Model("Assignment_Problem")
    model.Params.OutputFlag = 0
    
    # Decision variables: x[i,j]
    # AP has the integrality property, so solving as LP (Continuous) 
    # yields integer solutions (0 or 1) automatically.
    x = model.addVars(n, n, vtype=GRB.CONTINUOUS, lb=0, name="x")
    
    # Objective: minimize total distance
    obj = quicksum(distance_matrix[i, j] * x[i, j]
                      for i in range(n) for j in range(n) if i != j)
    model.setObjective(obj, GRB.MINIMIZE)
    
    # Constraint (1): Each node has exactly one outgoing edge
    model.addConstrs((quicksum(x[i, j] for j in range(n) if j != i) == 1 
                      for i in range(n)), name="out")

    # Constraint (2): Each node has exactly one incoming edge
    model.addConstrs((quicksum(x[i, j] for i in range(n) if j != i) == 1 
                      for j in range(n)), name="in")
    
    # Prevent self-loops
    for i in range(n):
        x[i, i].ub = 0

    return model, x

def solve_ap(distance_matrix):
    """
    Solves the Assignment Problem (AP) relaxation.
    Returns the optimal objective value.
    """
    model, x = build_ap_model(distance_matrix)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return model.objVal
    else:
        return None
