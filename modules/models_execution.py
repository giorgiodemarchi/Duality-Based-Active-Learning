from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from gurobipy import Model, GRB

def classificationModel(labelled_df, seed):

    X = labelled_df.drop('Suitable', axis=1)
    y = labelled_df['Suitable']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(random_state=seed)
    log_reg.fit(X_train_scaled, y_train)

    y_pred = log_reg.predict(X_test_scaled)

    sample_accuracy = accuracy_score(y_test, y_pred)

    return sample_accuracy, log_reg, scaler

def solve_opti_model(z, supply, tr_capacity, des_capacity, cost_1, cost_2, df):
    # Define z_capacity
    z_capacity = z * des_capacity

    # Create a new model
    m = Model("optimization")
    m.setParam('OutputFlag', 0)
    # Add variables
    X = m.addVars(len(supply), len(tr_capacity), lb=0, name="X")
    Y = m.addVars(len(tr_capacity), len(des_capacity), lb=0, name="Y")
    # Set objective
    m.setObjective(sum(cost_1[i, j] * X[i, j] for i in range(len(supply)) for j in range(len(tr_capacity))) +
                sum(cost_2[j, k] * Y[j, k] for j in range(len(tr_capacity)) for k in range(len(des_capacity))),GRB.MINIMIZE)
    # Add constraints 
    for i in range(len(supply)):  # 6 constraints
        m.addConstr(sum(X[i, j] for j in range(len(tr_capacity))) == supply[i], "Supply_%d" % i)
    for j in range(len(tr_capacity)): # 8 constraints
        m.addConstr(sum(X[i, j] for i in range(len(supply))) <= tr_capacity[j], "TransCapacity_%d" % j)
    for k in range(len(des_capacity)): # 10000 constraints
        m.addConstr(sum(Y[j, k] for j in range(len(tr_capacity))) <= z_capacity[k], "Link_%d" % k)
    for j in range(len(tr_capacity)): # 8 constraints - What goes in comes out
        m.addConstr(sum(X[i, j] for i in range(len(supply))) == sum(Y[j, k] for k in range(len(des_capacity))), f"TransshipmentBalance_{j}")

    # Optimize model
    m.optimize()

    if m.status == GRB.OPTIMAL:
        opt_objective = m.ObjVal
        # For X variables
        opt_X = [(f"X[{i},{j}]", X[i, j].X) for i in range(len(supply)) for j in range(len(tr_capacity))]
        df_X = pd.DataFrame(opt_X, columns=["Variable", "Value"])
        # For Y variables
        opt_Y = [(f"Y[{j},{k}]", Y[j, k].X) for j in range(len(tr_capacity)) for k in range(len(des_capacity))]
        df_Y = pd.DataFrame(opt_Y, columns=["Variable", "Value"])
        df_Y['j'] = df_Y['Variable'].apply(lambda x: int(x.split(',')[0].split('[')[1]))
        df_Y['k'] = df_Y['Variable'].apply(lambda x: int(x.split(',')[1].split(']')[0]))
        df_Y.drop('Variable', axis=1, inplace=True)
        link_duals = [(f"Link_{k}", m.getConstrByName(f"Link_{k}").Pi) for k in range(len(des_capacity))]
        link_slacks = [(f"Link_{k}", m.getConstrByName(f"Link_{k}").Slack) for k in range(len(des_capacity))]
        df_link_duals = pd.DataFrame(link_duals, columns=["Constraint", "Dual Value"])
        df_slacks = pd.DataFrame(link_slacks, columns=["Constraint", "Slack"])
        df_link_duals = pd.merge(df_slacks, df_link_duals, on="Constraint")
        dual_df = pd.merge(df_link_duals, df[['Suitable']], left_index=True, right_index=True)
        capacity_df = pd.DataFrame(des_capacity).rename(columns={0:'Capacity'})
        dual_df = pd.merge(dual_df, capacity_df, left_index=True, right_index=True)
        return opt_objective, df_X, df_Y, dual_df
    
    else:
        print("Infeasible")
        return 0, 0, 0, 0
