from gurobipy import Model, GRB, quicksum
import pandas as pd

def build_mtz_model(distance_matrix, is_ip=True):
    N = distance_matrix.shape[0]
    model = Model("MTZ_TSP")
    model.Params.OutputFlag = 0
    vtype = GRB.BINARY if is_ip else GRB.CONTINUOUS

    x = model.addVars(N, N, lb=0, ub=1, vtype=vtype, name="x")
    u = model.addVars(N, lb=0, ub=N-1, name="u")

    model.addConstrs((quicksum(x[i, j] for j in range(N)) == 1 for i in range(N)))
    model.addConstrs((quicksum(x[i, j] for i in range(N)) == 1 for j in range(N)))
    model.addConstrs((x[i, i] == 0 for i in range(N)))
    model.addConstr(u[0] == 0)

    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                model.addConstr(u[i] - u[j] + 1 <= (N-1)*(1 - x[i, j]))

    return model, x

def solve_mtz(distance_matrix, is_ip=True, time_limit=None):
    model, x = build_mtz_model(distance_matrix, is_ip=is_ip)
    N = distance_matrix.shape[0]
    model.setObjective(quicksum(distance_matrix[i, j] * x[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)
    if time_limit:
        model.Params.TimeLimit = time_limit
    if model.ModelName == "MTZ_TSP":
        model.Params.MIPFocus = 1
    model.optimize()
    if model.Status == GRB.OPTIMAL:
        return model.objVal, False
    elif model.Status == GRB.TIME_LIMIT:
        if model.SolCount > 0:
            return model.objVal, True
        else:
            return None, True
    else:
        return None, False