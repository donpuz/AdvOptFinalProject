#!/usr/bin/env python
# coding: utf-8

# # Final Project

# In[1]:


import random
import pandas as pd
from gurobipy import *


# In[2]:


def get_d(e, s, F):
    coverage = sum(F.loc[e,s])
    return min(1, coverage)


# In[3]:


def get_s(nodes, sensors):
    sigma_s = random.sample(nodes, sensors)
    return(sigma_s)


# In[4]:


def get_random_I(I_length, n_nodes, n_sensors):
    nodes_indexes = [i for i in range(n_nodes)]
    I = []

    for _ in range(I_length):
        sigma_s = get_s(nodes_indexes, n_sensors)
        if sigma_s not in I:
            I.append(sigma_s)
        
    return I


# In[5]:


def run_RMP(params):
    model = Model("Model")
    model.Params.LogToConsole = 0

    z = model.addVar( name="z")
    sigmas = [model.addVar(name = "s_" + str(s)) for s in range(params["I"]["length"])]

    model.setObjective(z, GRB.MAXIMIZE)

    for e in range(params["n_pipes"]):
        model.addConstr(
            z <= sum(
                get_d(e, params["I"]["set"][s], params["F"]["matrix"])*sigmas[s]
                for s in range(params["I"]["length"])
            ),
            name  = "y_"+str(e)
        )

    model.addConstr(sum(sigmas) == 1, name = "w")
    model.optimize();
    
    return model


# In[6]:


def get_params(n_sensors, n_nodes, n_pipes, I, F):
    return {
        "F" : {
            "n_rows" : F.shape[1],
            "n_columns" : F.shape[0],
            "matrix" : F
        },
        "I" : {
            "length" : len(I),
            "set" : I
        },
        "n_nodes": n_nodes,
        "n_sensors": n_sensors,
        "n_pipes" : n_pipes
    }


# In[7]:


def update_I(params, s):
    params["I"]["set"].append(s)
    params["I"]["length"] += 1


# In[8]:


def get_dual_variables(model):
    dual_vars = model.getConstrs()
    len_dual = len(dual_vars)
    return {
        "Y" : [y.Pi for y in dual_vars[0:len_dual-1]],
        "w" : dual_vars[len_dual-1].Pi
    }


# In[9]:


def run_PrincingP(dual_vars, params):
    model = Model("Model")
    model.Params.LogToConsole = 0

    P = [model.addVar(name = "P_" + str(e)) for e in range(params["n_pipes"])]
    X = [model.addVar(vtype=GRB.BINARY, name = "X_" + str(v)) for v in range(params["n_nodes"])]

    model.setObjective(sum(P[e]*dual_vars["Y"][e] for e in range(params["n_pipes"])), GRB.MAXIMIZE)
    model.addConstr(sum(X) == params["n_sensors"])
    
    for e in range(params["n_pipes"]):
        model.addConstr(
            P[e] <= sum(
                params["F"]["matrix"].loc[e, v]*X[v]
                for v in range(params["n_nodes"])
            ),
            name = "y_"+str(e)
        )
        model.addConstr(P[e] <= 1)

    model.optimize();
    X_values = [x.x for x in X]
    new_s = [i for i in range(len(X_values)) if X_values[i] > 0]

    return model.ObjVal, new_s


# In[10]:


def get_dataframe():
    return pd.DataFrame(data = {
        'ObjValue' : [],
        'ReducedCost' : []
    })


# In[11]:


def update_results(RMproblem, cost, results):
    results.loc[len(results)] = [
        RMproblem.ObjVal,
        cost
    ]
    return results


# In[12]:


def get_prob_distribution(params, RMproblem, results):
    I = pd.DataFrame(data = params["I"]["set"])
    I.columns = ["Sensor_"+str(i) for i in range(1,params["n_sensors"]+1)]

    dist = RMproblem.getVars()
    I["Prob"] = [x.x for x in dist[1:len(dist)]]

    I.to_csv("ProbDist"+str(params["n_sensors"])+".csv", sep=',')


# In[ ]:





# -----------------------------
# ## Main

# In[13]:


def run(params, results):
    is_optimal = False
    counter = 0

    while is_optimal == False:
        counter +=1
        RMproblem = run_RMP(params)

        dual_vars = get_dual_variables(RMproblem)
        cost, s = run_PrincingP(dual_vars, params)

        results = update_results(RMproblem, cost, results)

        if cost - dual_vars["w"] <= 0:
            get_prob_distribution(params, RMproblem, results)
            results.to_csv("ObjVal"+str(params["n_sensors"])+".csv", sep = ",")
            is_optimal = True
        else:
            update_I(params, s)

        print("Iteration number " +str(counter)+ " complete")
        print("Objective Value: "+str(RMproblem.ObjVal))


# In[16]:


n_nodes = 811
n_pipes = 1123
n_sensors = 7

I = [[i for i in range(n_sensors)]]

F = pd.read_csv(
    'Detection_Matrix.csv',
    names = [x for x in range(n_nodes)]
)


# In[ ]:


params = get_params(n_sensors, n_nodes, n_pipes, I, F)
results = get_dataframe()

results = run(params, results)


# In[ ]:




