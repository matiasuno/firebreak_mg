import ReadDataPrometheus
import os
from matplotlib.pylab import *
#import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pyomo.environ import *
import time

#%%

Folder = os.getcwd()

#archivo que contiene el significado de cada codigo del archivo forest
#FBPlookup = Folder + 'fbp_lookup_table.csv' # Diccionario
FBPlookup = Folder + '/fbp_lookup_table.csv' # Diccionario

#archivo con el contenido de cada celda del bosque
ForestFile = Folder + '/Forest.asc'   # Combustible

# Se lee lookup_table
FBPDict, ColorsDict = ReadDataPrometheus.Dictionary(FBPlookup)
# Se lee la capa de combustibles Forest.asc
CellsGrid3, CellsGrid4, Rows, Cols, AdjCells, CoordCells, CellSize = ReadDataPrometheus.ForestGrid(ForestFile,FBPDict)


NCells = Rows * Cols
m = Rows
n = Cols
Colors = []

# Aqui construimos un set de colores, para posteriormente graficar
for i in range(NCells):
    if str(CellsGrid3[i]) not in ColorsDict.keys():
        Colors.append((1.0,1.0,1.0,1.0))
    if str(CellsGrid3[i]) in ColorsDict.keys():
        Colors.append(ColorsDict[str(CellsGrid3[i])])

AvailSet = set()
NonAvailSet = set()

for i in range(NCells):
    if CellsGrid4[i] != 'NF':
        AvailSet.add(i+1)
    else:
        NonAvailSet.add(i+1)

setDir = ['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW']
aux = set([])

Adjacents = {} # diccionario de celdas adyacentes disponibles (no condidera las NF)
for k in range(NCells):
    aux = set([])
    for i in setDir:
        if AdjCells[k][i] != None:
            if AdjCells[k][i][0] in AvailSet :
                aux = aux | set(AdjCells[k][i])
    Adjacents[k + 1] = aux & AvailSet

FAG = nx.Graph() # Fuel Avail Graph
FAG.add_nodes_from(list(AvailSet))

coord_pos = dict()
for i in AvailSet:
    coord_pos[i] = CoordCells[i-1]

ColorsAvail = {}
for i in AvailSet:
    FAG.add_edges_from([(i,j) for j in Adjacents[i]])
    ColorsAvail[i] = Colors[i-1]

Folder = 'C:/Users/Matias/Desktop/proyecto_magister/simulador/Cell2Fire-main/results/Sub40x40'
cmdoutput1 = os.listdir(Folder + '/Messages')
if ".DS_Store" in cmdoutput1:
    idx = cmdoutput1.index('.DS_Store') # Busca el indice que hay que eliminar
    del cmdoutput1[idx]
if "desktop.ini" in cmdoutput1:
    idx = cmdoutput1.index('desktop.ini') # Busca el indice que hay que eliminar
    del cmdoutput1[idx]
pathmessages = Folder + '/Messages/'

# Construimos un objeto DiGraph (grafo dirigido)
MDG = nx.MultiDiGraph()
# Agregamos nodos desde lista "nodos"
#MDG.add_nodes_from(list(AvailSet))


st_points = []
# Se cargan los incendios y se agregan iterativamente a MDG
for k in cmdoutput1:
    # Leemos la marca
    H = nx.read_edgelist(path = pathmessages + k,
                         delimiter=',',
                         create_using = nx.DiGraph(),
                         nodetype = int,
                         data = [('time', float), ('ros', float)])
    #print(pathmessages+k)
    try:
        st_points.append(list(H.edges)[0][0])
    except IndexError:
        pass
    # Se carga el MultiDigrafo
    MDG.add_weighted_edges_from(H.edges(data='time'), weight='ros')

#crea un diccionario con la cantidad de veces que se quema cada nodo
edges = MDG.edges
burned = dict.fromkeys(AvailSet,0)
for e in edges:
        burned[e[1]]=burned[e[1]]+1


# Se calculan las metricas de centralidad u otras (puede ser a nivel de nodos o aristas)
# Betweenness centrality:
metric_values = nx.betweenness_centrality(MDG, normalized = True, weight = 'ros') # dictionary


#%%------------------------------------- MODELO ------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------
sims = list(range(len(st_points)))
n_sims = len(sims)
nodos = list(MDG.nodes())

for i in range(n_sims):
    sims[i] = sims[i]+1

cortafuegos = 10

#rho probabilidad de ocurrencia del punto de ignicion
#w preponderancia del nodo

#warmstart
#contador = 0
#for key in burned:
#    if burned[key] > n_sims*0.5:
#        contador = contador +1
#        model.y[key] = 1
#print(contador)

#%% ---------------------------- MODELO 1 -------------------------------------------------------------------------
#minimizacion esperanza con maximo de cortafuegos


model = ConcreteModel()
model.x = Var(nodos, sims, within = Binary)
model.y = Var(nodos, within = Binary)

model.fo = Objective(expr = sum(sum(model.x[node,s] for node in nodos) for s in sims), sense = minimize)

model.firebreaks = Constraint(expr = sum(model.y[node] for node in nodos) <= cortafuegos)

model.spread = ConstraintList()
simulacion = 0
for k in cmdoutput1:
    simulacion = simulacion +1
    # Leemos la marca
    H = nx.read_edgelist(path = pathmessages + k,
                         delimiter=',',
                         create_using = nx.DiGraph(),
                         nodetype = int,
                         data = [('time', float), ('ros', float)])        
    for node in list(H.nodes):
        nbrs = list(H.neighbors(node))
        for nbr in nbrs:
            model.spread.add(model.x[node,simulacion] <= model.x[nbr,simulacion]+model.y[nbr])
            
model.st_point = ConstraintList()
for s in sims:
    point = st_points[s-1]
    model.st_point.add(model.x[point,s] == 1)

titulo = 'modelo 1'

#%% ---------------------------- MODELO 2 -------------------------------------------------------------------------
#minimizar la probabilidad de quema de cada nodo

model = ConcreteModel()
model.x = Var(nodos, sims, within = Binary)
model.y = Var(nodos, within = Binary)
model.eta = Var(within=Reals)

model.fo = Objective(expr = model.eta, sense = minimize)

model.firebreaks = Constraint(expr = sum(model.y[node] for node in nodos) <= cortafuegos)

model.spread = ConstraintList()
simulacion = 0
for k in cmdoutput1:
    simulacion = simulacion +1
    # Leemos la marca
    H = nx.read_edgelist(path = pathmessages + k,
                         delimiter=',',
                         create_using = nx.DiGraph(),
                         nodetype = int,
                         data = [('time', float), ('ros', float)])        
    for node in list(H.nodes):
        nbrs = list(H.neighbors(node))
        for nbr in nbrs:
            model.spread.add(model.x[node,simulacion] <= model.x[nbr,simulacion]+model.y[nbr])
            
model.st_point = ConstraintList()
for s in sims:
    point = st_points[s-1]
    model.st_point.add(model.x[point,s] == 1)

model.eta_con = ConstraintList()
for n in nodos:
    model.eta_con.add(sum(model.x[n,s] for s in sims) <= model.eta)

titulo = 'modelo 2'

#%% ---------------------------- MODELO 3 -------------------------------------------------------------------------
#minimizar el numero de celdas que se queman por escenario

model = ConcreteModel()
model.x = Var(nodos, sims, within = Binary)
model.y = Var(nodos, within = Binary ,initialize = 0)
model.eta = Var(within=Reals)

model.fo = Objective(expr = model.eta, sense = minimize)

model.firebreaks = Constraint(expr = sum(model.y[node] for node in nodos) <= cortafuegos)

model.spread = ConstraintList()
model.eta_con = ConstraintList()
simulacion = 0
for k in cmdoutput1:
    simulacion = simulacion +1
    # Leemos la marca
    H = nx.read_edgelist(path = pathmessages + k,
                         delimiter=',',
                         create_using = nx.DiGraph(),
                         nodetype = int,
                         data = [('time', float), ('ros', float)])
    nodes_sim = list(H.nodes)
    model.eta_con.add(sum(model.x[nodo,simulacion] for nodo in nodes_sim) <= model.eta)
    
    for node in nodes_sim:
        nbrs = list(H.neighbors(node))
        for nbr in nbrs:
            model.spread.add(model.x[node,simulacion] <= model.x[nbr,simulacion]+model.y[nbr])
            
model.st_point = ConstraintList()
for s in sims:
    point = st_points[s-1]
    model.st_point.add(model.x[point,s] == 1)

titulo = 'modelo 3'

#%% ---------------------------- MODELO 4 -------------------------------------------------------------------------
#minimizar la cantidad de cortafuegos dado un maximo de bosque quemado (nivel de servicio)
alfa = 0.4*n_sims

model = ConcreteModel()
model.x = Var(nodos, sims, within = Binary, initialize = 0)
model.y = Var(nodos, within = Binary, initialize = 1)

model.fo = Objective(expr = sum(model.y[node] for node in nodos), sense = minimize)

model.alfa = ConstraintList()
for n in nodos:
    model.alfa.add(sum(model.x[n,s] for s in sims) <= alfa)

model.spread = ConstraintList()
simulacion = 0
for k in cmdoutput1:
    simulacion = simulacion +1
    # Leemos la marca
    H = nx.read_edgelist(path = pathmessages + k,
                         delimiter=',',
                         create_using = nx.DiGraph(),
                         nodetype = int,
                         data = [('time', float), ('ros', float)])        
    for node in list(H.nodes):
        nbrs = list(H.neighbors(node))
        for nbr in nbrs:
            model.spread.add(model.x[node,simulacion] <= model.x[nbr,simulacion]+model.y[nbr])

model.st_point = ConstraintList()
for s in sims:
    point = st_points[s-1]
    model.st_point.add(model.x[point,s] == 1)

titulo = 'modelo4'

#%% SOLVER------------------------------------------------------------------------------------------------------------
t1 = time.time()
solver = SolverFactory('glpk')
solver.options["tmlim"] = 3600
solver.options["mipgap"] = 0.1
results = solver.solve(model, tee=True)
t2 = time.time()

data = results.Problem._list
LB = data[0].lower_bound
UB = data[0].upper_bound
gap = abs(UB - LB) * 100/max(0.1, abs(UB))
print(gap)


contador_cfuegos = 0
for node in nodos:
    if value(model.y[node]) > 0:
        contador_cfuegos = contador_cfuegos +1
print('la cantidad de cortafuegos puestos es de: ',contador_cfuegos)

lista_cf = []
for node in nodos:
    if value(model.y[node]) > 0:
        print('se aplicó cortafuego en el nodo',node)
        lista_cf.append(node)


#%%------------------------------------- GRAFICAR MODELO ------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

burned_rslt = dict.fromkeys(AvailSet,0)
for s in sims:
    for n in nodos:
        if value(model.x[n,s]) > 0:
            burned_rslt[n] = burned_rslt[n]+1

burned_rslt[13] = max(burned.values())


figure(1, figsize=(11,8))
nx.draw_networkx(FAG, pos = coord_pos,
                         node_size = 50,
                         with_labels = False,
                         node_shape = 's',
                         node_color = 'k')

nx.draw_networkx_nodes(FAG, pos = coord_pos,
                   node_size = 50,
                   nodelist = list(FAG.nodes),
                   node_shape='s',
                   node_color = list(ColorsAvail.values()))

nc = nx.draw_networkx_nodes(FAG, pos = coord_pos,
                               # node_size = nodesize,  # array
                               # nodelist=nodelist,  # list
                            linewidths = 2.0,
                            node_size = 20,
                            cmap = 'Reds',
                            node_shape = 's',
                            node_color = list(burned_rslt.values())
                            #node_color = list(metric_values.values())
                            )

cf = nx.Graph()
nx.draw_networkx_nodes(cf, pos = coord_pos,
                   node_size = 10,
                   nodelist = lista_cf,
                   node_shape='s',
                   node_color = 'b')

tiempo = round(t2-t1)

plt.title(titulo+' '+str(contador_cfuegos)+' cortafuegos '+str(tiempo)+'seg '+'alfa='+str(alfa/n_sims))

plt.colorbar(nc)
plt.show()

#%%------------------------------------- GRAFICAR BOSQUE INICIAL -------------------------------------------------------

figure(1, figsize=(11,8))
nx.draw_networkx(FAG, pos = coord_pos,
                         node_size = 50,
                         with_labels = False,
                         node_shape = 's',
                         node_color = 'k')

nx.draw_networkx_nodes(FAG, pos = coord_pos,
                   node_size = 50,
                   nodelist = list(FAG.nodes),
                   node_shape='s',
                   node_color = list(ColorsAvail.values()))

nc = nx.draw_networkx_nodes(FAG, pos = coord_pos,
                               # node_size = nodesize,  # array
                               # nodelist=nodelist,  # list
                            linewidths = 2.0,
                            node_size = 20,
                            cmap = 'Reds',
                            node_shape = 's',
                            node_color = list(burned.values())
                            )

plt.title('Bosque inicial')

plt.colorbar(nc)
plt.show() 