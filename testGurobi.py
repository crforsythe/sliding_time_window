
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

m = gp.Model('test')
m.setParam('OutputFlag', 0)
m.update()

x = m.addVars(1, vtype=GRB.CONTINUOUS, name='x')

m.update()

obj = gp.LinExpr()

obj += x[0]

m.addConstr(x[0]>=3)

m.setObjective(obj, GRB.MINIMIZE)

m.update()


a = m.optimize()

t = m.status, m.getObjective().getValue()

