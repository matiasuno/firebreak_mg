#!/usr/bin/env python
# coding: utf-8

# In[2]:


import networkx as nx
from matplotlib import pyplot as plt
from random import randint
from pyomo.environ import *
from  matplotlib.colors import LinearSegmentedColormap
import time


# In[9]:



dim = 8
simulaciones = 30


G = nx.grid_2d_graph(dim,dim, create_using=nx.DiGraph)

lista =[]
for node in G.nodes():
    lista.append(node[0]*dim+node[1])

mapping = dict(zip(G.nodes(),lista))
G = nx.relabel_nodes(G, mapping)

    
nodes = G.nodes()
F = G.copy()

#---------- REALIZACIONES --------------------------

wind = randint(1,4)
wind =[]
direc = simulaciones
i = 0
while i < direc:
    dire = randint(1,4)
    wind.append(dire)
    i=i+1

st_points=[]
puntos = simulaciones
i = 0
while i < puntos:
    punto = randint(0,dim**2-1)
    st_points.append(punto)
    i=i+1


for dire in range(1,5):
    G = F.copy()
    edgelist = list(G.edges())
    if dire == 1:
        for i in edgelist:    
            if i[0] > i[1]:
                G.remove_edge(i[0],i[1])
        g1 = G.copy()
        
    elif dire == 2:
        for i in edgelist:    
            if i[0] - i[1] == 1:
                G.remove_edge(i[0],i[1])
            elif i[0] - i[1] == -dim:
                G.remove_edge(i[0],i[1])            
        g2 = G.copy()
        
    elif dire == 3:
        for i in edgelist:    
            if i[0] - i[1] == -1:
                G.remove_edge(i[0],i[1])
            elif i[0] - i[1] == dim:
                G.remove_edge(i[0],i[1])
        g3 = G.copy()
                
    else:
        for i in edgelist:    
            if i[0] < i[1]:
                G.remove_edge(i[0],i[1])
        g4 = G.copy()


# In[10]:


t1 = time.time()

nodes = list(range(0,dim**2))
cortafuegos = 4

simulations = list(range(0,simulaciones))


model = ConcreteModel()

model.x = Var(nodes,simulations, within=Binary)
model.y = Var(nodes, within=Binary)

model.fo = Objective(expr= sum(model.x[node,sim] for node in nodes for sim in simulations), sense=minimize)

model.r1 = ConstraintList()
for sim in simulations:
    model.r1.add(expr= (model.x[st_points[sim],sim] >= 1))

model.r2 = Constraint(expr=sum(model.y[node] for node in nodes) <= cortafuegos)

model.r3 = ConstraintList()
for sim in simulations:
    if wind[sim] == 1:
        G = g1.copy()
    elif wind[sim] == 2:
        G = g2.copy()
    elif wind[sim] == 3:
        G = g3.copy()
    else:
        G = g4.copy()
    for node in nodes:
            nbrs = list(G.neighbors(node))
            for nbr in nbrs:
                model.r3.add(model.x[node,sim] <= model.x[nbr,sim]+model.y[nbr])
            
opt = SolverFactory("glpk")
opt.solve(model)


for node in nodes:
    if value(model.y[node]):
        print('se aplicó cortafuego en el nodo',node)
        
print(t1)        
print(time.time()-t1,'segundos')


# In[11]:


burned_cells=[]
for node in nodes:
    burned_cells.append(0)

for sim in simulations:
    for node in nodes:
        if value(model.x[node,sim]) == 1:
            burned_cells[node] = burned_cells[node]+1
print(burned_cells)

for node in nodes:
    if value(model.y[node]) == 1:
        burned_cells[node] = -simulaciones/6
        

G.remove_edges_from(list(G.edges))
plt.figure(figsize=(dim*2,dim*2))
pos = {(x):(x//dim,-(x%dim)) for x in G.nodes()}
nx.draw(G, 
        pos = pos,
        node_color=burned_cells, 
        with_labels=True,
        node_size=600,
        cmap=LinearSegmentedColormap.from_list('rg',["b","g", "y", "r"], N=256) )


# In[ ]:




