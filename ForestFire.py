from operator import itemgetter
import ReadDataPrometheus
from matplotlib.pylab import *
import networkx as nx
import matplotlib.pyplot as plt
from pyomo.environ import *
import os

Folder = os.getcwd()
FBPlookup = Folder + '/fbp_lookup_table.csv' # Diccionario
ForestFile = Folder + '/Forest.asc'   # Combustible

# Se lee lookup_table
FBPDict, ColorsDict = ReadDataPrometheus.Dictionary(FBPlookup)
# Se lee la capa de combustibles Forest.asc
CellsGrid3, CellsGrid4, Rows, Cols, AdjCells, CoordCells, CellSide = ReadDataPrometheus.ForestGrid(ForestFile,FBPDict)


NCells = Rows * Cols
m = Rows
n = Cols
Colors = []
#firebreak=[224,225,226,227,228,229,230,231,232,233,234 235 236]
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

Adjacents = {} # diccionario de celdas adyacentes disponibles (no considera las NF)
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

#plt.axis([-1, m, -1, n])


# Leemos el Fire digraph desde Messages
FolderMessages = Folder
H = nx.read_edgelist(path = FolderMessages+'\Messages\MessagesFile01.csv',
                     delimiter=',',
                     create_using = nx.DiGraph(), # Es de tipo grafo dirigido
                     nodetype = int,
                     data = [('time',float), ('ros',float)]) #


# Imprimimos el grafo
fig = figure(1,figsize=(8,8))

nx.draw_networkx(FAG, pos = coord_pos,
             node_size = 50,
             with_labels = False,
             node_shape='s',
             node_color = list(ColorsAvail.values())
             )

nx.draw_networkx_nodes(FAG, pos = coord_pos,
                   node_size = 50,
                   nodelist = list(FAG.nodes),
                   node_shape='s',
                   node_color = list(ColorsAvail.values()))




#nx.draw_networkx_edges(H, pos = coord_pos, edge_color = 'r', width = 1.0, arrowsize=10,arrows=True)



nodos = H.nodes()

cortafuegos = 4
model = ConcreteModel()

model.x = Var(nodos, within = Binary)
model.y = Var(nodos, within = Binary)

model.fo = Objective(expr = sum(model.x[node] for node in nodos), sense = minimize)

model.r2 = Constraint(expr=sum(model.y[node] for node in nodos) <= cortafuegos)

model.r3 = ConstraintList()
for node in nodos:
        nbrs = list(H.neighbors(node))
        for nbr in nbrs:
            model.r3.add(model.x[node] <= model.x[nbr]+model.y[nbr])

st_point = list(H.edges)[0][0]
model.r1 = Constraint(expr = model.x[st_point] == 1)

glpk = SolverFactory('glpk')
glpk.solve(model, tee=False)

lista_cf = []
for node in nodos:
    if value(model.y[node]) > 0:
        print('se aplicó cortafuego en el nodo',node)
        lista_cf.append(node)

quemados = []        
for node in nodos:
    if value(model.x[node]) > 0:
        quemados.append(node)
        
cf = nx.Graph()
nx.draw_networkx_nodes(cf, pos = coord_pos,
                   node_size = 10,
                   nodelist = lista_cf,
                   node_shape='s',
                   node_color = 'b')

quem = nx.Graph()
nx.draw_networkx_nodes(quem, pos = coord_pos,
                   node_size = 10,
                   nodelist = quemados,
                   node_shape='s',
                   node_color = 'r')
"""
burned_cells=[]
for node in nodes:
    burned_cells.append(0)

for sim in simulations:
    for node in nodes:
        if value(model.x[node,sim]) == 1:
            burned_cells[node] = burned_cells[node]+1
print(burned_cells)
"""
plt.show()


contador = 0
results = []
for node in nodos:
    results.append([node,value(model.x[node])])
    if value(model.x[node]) > 0:
        contador = contador +1

print('cantidad de nodos: ',len(nodos),'\n nodos quemados: ',contador)
#print(results)

