#%% IMPORTACIONES
import ReadDataPrometheus
import os
from matplotlib.pylab import *
import matplotlib.pyplot as plt
import networkx as nx
import random as rd
import gurobipy as gp
from gurobipy import GRB
from winsound import Beep
import heapq
from operator import itemgetter
import pandas as pd

def alarm(f,t):
    return(Beep(f,t),Beep(f+100,t),Beep(f+300,t*2))

#%% IMPORTAR DATOS C2F
Folder = os.getcwd()

#archivo que contiene el significado de cada codigo del archivo forest
#FBPlookup = Folder + 'fbp_lookup_table.csv' # Diccionario

FBPlookup = Folder + '/Sub40x40/fbp_lookup_table.csv' # Diccionario
#FBPlookup = Folder + '/Sub20x20/fbp_lookup_table.csv'

#archivo con el contenido de cada celda del bosque
ForestFile = Folder + '/Sub40x40/Forest.asc'   # Combustible
#ForestFile = Folder + '/Sub20x20/Forest.asc'

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

Folder = 'C:/Users/Matias/Documents/proyecto_magister/simulador/Cell2Fire-main/results/Sub40x40'
#Folder = 'C:/Users/matia/proyecto_magister/simulador/Cell2Fire-main/results/Sub40x40'
#Folder = 'C:/Users/jaime/OneDrive/Documentos/Matias/simulaciones/300sims'

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

avail = list(AvailSet)
st_points = []
sim_edges = []
burned = dict.fromkeys(AvailSet,0)
#eigenvector = dict.fromkeys(AvailSet,0)
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
        st_points.append(avail[rd.randint(0, len(avail)-1)])
        print(k)
    
    sim_edges.append(list(H.edges))
    
    # Se carga el MultiDigrafo
    MDG.add_weighted_edges_from(H.edges(data='time'), weight='ros')
    contador = dict.fromkeys(list(H.nodes),0)
    for e in H.edges:
        for n in list(H.nodes):
            if n == e[1] and contador[n] == 0:
                contador[n]=contador[n]+1
                burned[e[1]]=burned[e[1]]+1
        
#crea un diccionario con la cantidad de veces que se quema cada nodo
# =============================================================================
# edges = MDG.edges
# burned = dict.fromkeys(AvailSet,0)
# for e in edges:
#         burned[e[1]]=burned[e[1]]+1
# =============================================================================

for point in st_points:
    burned[point] = burned[point]+1
edges=list(MDG.edges)
# MEDIDAS DE CENTRALIDAD
#betweenness = nx.betweenness_centrality(MDG, normalized = True, weight = 'ros')
# PREPARACION MODELO 
# =============================================================================
# sims = list(range(len(st_points)))
# n_sims = len(sims)
# 
# 
# nodos = list(MDG.nodes())
# for i in range(n_sims):
#     sims[i] = sims[i]+1
# 
# intensidad = 0.05
# cortafuegos = int(NCells*intensidad)
# cortafuegos = 20
# topitems = heapq.nlargest(cortafuegos, betweenness.items(), key=itemgetter(1))
# warmstart = dict(topitems)
# 
# n = cantidad de cortafuegos a dar en el warm start
# n = cortafuegos
# n = 100
# topitems2 = heapq.nlargest(n, burned.items(), key=itemgetter(1))
# warmstart2 = dict(topitems2)
# 
# rho probabilidad de ocurrencia del punto de ignicion
# w preponderancia del nodo
# =============================================================================

sims = list(range(len(st_points)))
n_sims = len(sims)

nodos = list(MDG.nodes())
for i in range(len(sims)):
    sims[i] = sims[i]+1
#%% MODELOS 
#MODELO 1 minimize expected burned cells

def modelo1(NCells,sims,nodos,cmdoutput1,intensidad,st_points,warmstart,gap,tlimit):
    
    n_sims = len(sims)
    cortafuegos = int(NCells*intensidad)
    
    model = gp.Model()
    
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    
    model.setObjective(gp.quicksum(x[n,s]*w[n] for n in nodos for s in sims)/n_sims, GRB.MINIMIZE)
    model.addConstr(gp.quicksum(y[n] for n in nodos) <= cortafuegos)
    
    simulacion = 0
    for k in cmdoutput1:
        simulacion = simulacion +1
        # Leemos la marca
        H = nx.read_edgelist(path = pathmessages + k,
                             delimiter=',',
                             create_using = nx.DiGraph(),
                             nodetype = int,
                             data = [('time', float), ('ros', float)])        
        for n in list(H.nodes):
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,simulacion] <= x[nbr,simulacion]+y[nbr])
                
    for s in sims:
        point = st_points[s-1]
        model.addConstr(x[point,s] == 1)
            
    for n in warmstart:
        y[n].Start = 1
            
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    model.optimize()
    
    gap = model.MIPGap
    gap = round(gap,3)
    tiempo = round(model.Runtime)
    
    #modelo 1
    fo1 = sum(x[n,s].x for n in nodos for s in sims)/n_sims
      
    #modelo 2
    eta_aux=[]
    for n in nodos:
        eta_aux.append(sum(x[n,s].x for s in sims))
    fo2 = max(eta_aux)/n_sims
      
    #modelo 3
    eta_aux2=[]
    for s in sims:
        eta_aux2.append(sum(x[n,s].x for n in nodos))
    fo3 = max(eta_aux2)/NCells
      
    #modelo 4
    fo4 = sum(y[n].x for n in nodos)
    
    b = 0
    lmbda = 0
    prom_pc = 0
    
   
    
    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0:
            contador_cfuegos = contador_cfuegos+1
            fb_list.append(n)
            
    burned_map = dict.fromkeys(AvailSet,0)
    for s in sims:
        for n in nodos:
            if x[n,s].x > 0:
                burned_map[n] = burned_map[n]+1
                
    titulo = 'm1_intensidad'+str(intensidad)
    results = ['modelo1',n_sims,intensidad,cortafuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pc,0] 
    titulo2 = 'm1'+' '+str(contador_cfuegos)+'cf '+str(n_sims)+'sims'+' fo:'+str(round(model.ObjVal,2))
    titulos = [titulo,titulo2]
    
    return results, fb_list, burned_map,titulos
# MODELO 2 minimize worst case by node ---------------------------------------------------------------------------
def modelo2(NCells,sims,nodos,cmdoutput1,intensidad,st_points,warmstart,gap,tlimit):

    n_sims = len(sims)
    cortafuegos = int(NCells*intensidad)
    
    model = gp.Model()
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    eta = model.addVar()
    
    model.setObjective(eta, GRB.MINIMIZE)
    
    #firebreaks intensity constraint
    model.addConstr(gp.quicksum(y[n] for n in nodos) <= cortafuegos)
    
    #restriccion que genera la cicatriz
    simulacion = 0
    for k in cmdoutput1:
        simulacion = simulacion +1
        # Leemos la marca
        H = nx.read_edgelist(path = pathmessages + k,
                             delimiter=',',
                             create_using = nx.DiGraph(),
                             nodetype = int,
                             data = [('time', float), ('ros', float)])        
        for n in list(H.nodes):
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,simulacion] <= x[nbr,simulacion]+y[nbr])
    
    #restriccion starting points            
    for s in sims:
        point = st_points[s-1]
        model.addConstr(x[point,s] == 1)
    
    #restriccion de burn probability
    for n in nodos:
        model.addConstr(gp.quicksum(x[n,s] for s in sims) <= eta)
    
    for n in warmstart:
        y[n].Start = 1
            
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    model.optimize()
    
    gap = model.MIPGap
    gap = round(gap,3)
    tiempo = round(model.Runtime)
    
    #modelo 1
    fo1 = sum(x[n,s].x for n in nodos for s in sims)/n_sims
      
    #modelo 2
    eta_aux=[]
    for n in nodos:
        eta_aux.append(sum(x[n,s].x for s in sims))
    fo2 = max(eta_aux)/n_sims
      
    #modelo 3
    eta_aux2=[]
    for s in sims:
        eta_aux2.append(sum(x[n,s].x for n in nodos))
    fo3 = max(eta_aux2)/NCells
      
    #modelo 4
    fo4 = sum(y[n].x for n in nodos)
    
    b = 0
    lmbda = 0
    prom_pc = 0
    
    
    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0:
            contador_cfuegos = contador_cfuegos+1
            fb_list.append(n)
            
    burned_map = dict.fromkeys(AvailSet,0)
    for s in sims:
        for n in nodos:
            if x[n,s].x > 0:
                burned_map[n] = burned_map[n]+1
                
    titulo = 'm2_intensidad'+str(intensidad)
    results = ['modelo2',n_sims,intensidad,cortafuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pc] 
    titulo2 = 'm2'+' '+str(contador_cfuegos)+'cf '+str(n_sims)+'sims'+' fo:'+str(round(model.ObjVal,2))
    titulos = [titulo,titulo2]
    
    return results, fb_list, burned_map,titulos

# MODELO 3 minimize worst case by scenary -------------------------------------------------------------------------------------------

def modelo3(NCells,sims,nodos,cmdoutput1,intensidad,st_points,warmstart,gap,tlimit):
    
    n_sims = len(sims)
    cortafuegos = int(NCells*intensidad)

    model = gp.Model()
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    eta = model.addVar()
    
    model.setObjective(eta, GRB.MINIMIZE)
    
    model.addConstr(gp.quicksum(y[n] for n in nodos) <= cortafuegos)
    
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
        #model.addConstr(gp.quicksum(x[n,simulacion] for n in nodes_sim) <= eta)
        for n in nodes_sim:
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,simulacion] <= x[nbr,simulacion]+y[nbr])
    
    for s in sims:
        point = st_points[s-1]
        model.addConstr(x[point,s] == 1)
        model.addConstr(gp.quicksum(x[n,s] for n in nodos) <= eta)
    
    for n in warmstart:
        y[n].Start = 1
            
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    model.optimize()
    
    gap = model.MIPGap
    gap = round(gap,3)
    tiempo = round(model.Runtime)
    
    #modelo 1
    fo1 = sum(x[n,s].x for n in nodos for s in sims)/n_sims
      
    #modelo 2
    eta_aux=[]
    for n in nodos:
        eta_aux.append(sum(x[n,s].x for s in sims))
    fo2 = max(eta_aux)/n_sims
      
    #modelo 3
    eta_aux2=[]
    for s in sims:
        eta_aux2.append(sum(x[n,s].x for n in nodos))
    fo3 = max(eta_aux2)/NCells
      
    #modelo 4
    fo4 = sum(y[n].x for n in nodos)
    
    b = 0
    lmbda = 0
    prom_pc = 0
        
    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0:
            contador_cfuegos = contador_cfuegos+1
            fb_list.append(n)
            
    burned_map = dict.fromkeys(AvailSet,0)
    for s in sims:
        for n in nodos:
            if x[n,s].x > 0:
                burned_map[n] = burned_map[n]+1
                
    titulo = 'm3_intensidad'+str(intensidad)
    results = ['modelo3',n_sims,intensidad,cortafuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pc] 
    titulo2 = 'm3'+' '+str(contador_cfuegos)+'cf '+str(n_sims)+'sims'+' fo:'+str(round(model.ObjVal,2))
    titulos = [titulo,titulo2]
    
    return results, fb_list, burned_map,titulos
        
# MODELO 4 minimize firebreaks intensity given worst case value by node burn probability (nivel de servicio) ------------------------------------

def modelo4(NCells,sims,nodos,cmdoutput1,alfa,st_points,warmstart,gap,tlimit):
    
    n_sims = len(sims)
    #cortafuegos = int(NCells*intensidad)
    
    alfa = alfa*n_sims
    
    model = gp.Model()
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    
    model.setObjective(gp.quicksum(y[n] for n in nodos), GRB.MINIMIZE)
    
    simulacion = 0
    for k in cmdoutput1:
        simulacion = simulacion +1
        # Leemos la marca
        H = nx.read_edgelist(path = pathmessages + k,
                             delimiter=',',
                             create_using = nx.DiGraph(),
                             nodetype = int,
                             data = [('time', float), ('ros', float)])        
        for n in list(H.nodes):
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,simulacion] <= x[nbr,simulacion]+y[nbr])
                
    for s in sims:
        point = st_points[s-1]
        model.addConstr(x[point,s] == 1)
        
    for n in nodos:
        model.addConstr(gp.quicksum(x[n,s] for s in sims) <= alfa)
    
    for n in warmstart:
        y[n].Start = 1
            
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    model.optimize()
    
    gap = model.MIPGap
    gap = round(gap,3)
    tiempo = round(model.Runtime)
    
    #modelo 1
    fo1 = sum(x[n,s].x for n in nodos for s in sims)/n_sims
      
    #modelo 2
    eta_aux=[]
    for n in nodos:
        eta_aux.append(sum(x[n,s].x for s in sims))
    fo2 = max(eta_aux)/n_sims
      
    #modelo 3
    eta_aux2=[]
    for s in sims:
        eta_aux2.append(sum(x[n,s].x for n in nodos))
    fo3 = max(eta_aux2)/NCells
      
    #modelo 4
    fo4 = sum(y[n].x for n in nodos)
    
    b = 0
    lmbda = 0
    prom_pc = 0

    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0:
            contador_cfuegos = contador_cfuegos+1
            fb_list.append(n)
    titulo = 'modelo4 '+str(alfa)        
    results = [titulo,n_sims,intensidad,contador_cfuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pc]
            
    burned_map = dict.fromkeys(AvailSet,0)
    for s in sims:
        for n in nodos:
            if x[n,s].x > 0:
                burned_map[n] = burned_map[n]+1
                
    titulo = titulo+' '+str(contador_cfuegos)+'cortafuegos '+str(n_sims)+'sims'+' fo'+str(round(model.ObjVal,2))
    
    return results, fb_list, burned_map,titulo

# MODELO 5 --------------------------------------------------------------------------------------

def modelo5(NCells,sims,nodos,cmdoutput1,intensidad,st_points,warmstart,gap,tlimit,b,lmbda):
    
    n_sims = len(sims)
    cortafuegos = int(NCells*intensidad)

    model = gp.Model()
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    eta = model.addVars(sims, vtype=GRB.CONTINUOUS, lb=0)
    phi = model.addVar(vtype=GRB.CONTINUOUS, lb =0)
    
    f_esperanza = gp.quicksum(x[n,s]*w[n] for n in nodos for s in sims)/n_sims
    f_cvar = phi+(1/(1-b))*gp.quicksum(eta[s] for s in sims)/n_sims
    
    model.setObjective(lmbda*f_esperanza+(1-lmbda)*f_cvar, GRB.MINIMIZE)
    
    #firebreak intensity constraint
    model.addConstr(gp.quicksum(y[n] for n in nodos) <= cortafuegos)
    
    #fire spread constraint
    simulacion = 0
    for k in cmdoutput1:
        simulacion = simulacion +1
        # Leemos la marca
        H = nx.read_edgelist(path = pathmessages + k,
                             delimiter=',',
                             create_using = nx.DiGraph(),
                             nodetype = int,
                             data = [('time', float), ('ros', float)])        
        for n in list(H.nodes):
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,simulacion] <= x[nbr,simulacion]+y[nbr])
    
    #starting points constraint            
    for s in sims:
        point = st_points[s-1]
        model.addConstr(x[point,s] == 1)
    
    for s in sims:
        model.addConstr(gp.quicksum(x[n,s] for n in nodos)-phi <= eta[s])
    
    for n in warmstart:
        y[n].Start = 1
            
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    model.optimize()
    
    lista_aux=[] 
    for s in sims:
        lista_aux.append(sum(x[n,s].x for n in nodos))
    
    lista_aux = sort(lista_aux)
    #print(lista_aux)
    
    peores_casos1 = int(round((1-0.9)*n_sims,0))
    peores_casos2 = int(round((1-0.95)*n_sims,0))
    
    prom_pcb1= 0
    prom_pcb2= 0
    for i in range(peores_casos1):
        prom_pcb1 = prom_pcb1 + lista_aux[-i-1]
    prom_pcb1 = prom_pcb1/peores_casos1
    
    for i in range(peores_casos2):
        prom_pcb2 = prom_pcb2 + lista_aux[-i-1]
    prom_pcb2 = prom_pcb2/peores_casos2
       
    gap = model.MIPGap
    gap = round(gap,3)
    tiempo = round(model.Runtime)
    
    #modelo 1
    fo1 = sum(x[n,s].x for n in nodos for s in sims)/n_sims
      
    #modelo 2
    eta_aux=[]
    for n in nodos:
        eta_aux.append(sum(x[n,s].x for s in sims))
    fo2 = max(eta_aux)/n_sims
      
    #modelo 3
    eta_aux2=[]
    for s in sims:
        eta_aux2.append(sum(x[n,s].x for n in nodos))
    fo3 = max(eta_aux2)/NCells
      
    #modelo 4
    fo4 = sum(y[n].x for n in nodos)
    
    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0:
            contador_cfuegos = contador_cfuegos+1
            fb_list.append(n)
    
    titulo = 'modelo4'       
    results = [titulo,n_sims,intensidad,cortafuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pcb1,prom_pcb2]
            
    burned_map = dict.fromkeys(AvailSet,0)
    for s in sims:
        for n in nodos:
            if x[n,s].x > 0:
                burned_map[n] = burned_map[n]+1
                
    
    titulo = 'm4_beta'+str(b)+'_lambda'+str(lmbda)
    results = ['modelo4',n_sims,intensidad,cortafuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pcb1,prom_pcb2] 
    titulo2 = 'm4 '+str(contador_cfuegos)+'cf '+str(n_sims)+'sims beta'+str(b)+' lambda'+str(lmbda)
    titulos = [titulo,titulo2]
    
    return results, fb_list, burned_map,titulos

#-------------------------------------------------------------- MODELO 6 -----------------------------------------------------------

def modelo6(NCells,sims,nodos,cmdoutput1,alfa,st_points,warmstart,gap,tlimit):
    
    n_sims = len(sims)
    #cortafuegos = int(NCells*intensidad)
    
    alfa = alfa*NCells
    
    model = gp.Model()
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    
    model.setObjective(gp.quicksum(y[n] for n in nodos), GRB.MINIMIZE)
    
    simulacion = 0
    for k in cmdoutput1:
        simulacion = simulacion +1
        # Leemos la marca
        H = nx.read_edgelist(path = pathmessages + k,
                             delimiter=',',
                             create_using = nx.DiGraph(),
                             nodetype = int,
                             data = [('time', float), ('ros', float)])        
        for n in list(H.nodes):
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,simulacion] <= x[nbr,simulacion]+y[nbr])
                
    for s in sims:
        point = st_points[s-1]
        model.addConstr(x[point,s] == 1)
        
    for s in sims:
        model.addConstr(gp.quicksum(x[n,s] for n in nodos) <= alfa)
    
    for n in warmstart:
        y[n].Start = 1
            
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    model.optimize()
    
    gap = model.MIPGap
    gap = round(gap,3)
    tiempo = round(model.Runtime)
    
    #modelo 1
    fo1 = sum(x[n,s].x for n in nodos for s in sims)/n_sims
      
    #modelo 2
    eta_aux=[]
    for n in nodos:
        eta_aux.append(sum(x[n,s].x for s in sims))
    fo2 = max(eta_aux)/n_sims
      
    #modelo 3
    eta_aux2=[]
    for s in sims:
        eta_aux2.append(sum(x[n,s].x for n in nodos))
    fo3 = max(eta_aux2)/NCells
      
    #modelo 4
    fo4 = sum(y[n].x for n in nodos)
    
    b = 0
    lmbda = 0
    prom_pc = 0

    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0:
            contador_cfuegos = contador_cfuegos+1
            fb_list.append(n)
    titulo = 'modelo6 '+str(alfa)        
    results = [titulo,n_sims,intensidad,contador_cfuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pc]
            
    burned_map = dict.fromkeys(AvailSet,0)
    for s in sims:
        for n in nodos:
            if x[n,s].x > 0:
                burned_map[n] = burned_map[n]+1
                
    titulo = 'm4_intensidad'+str(intensidad)
    results = [titulo,n_sims,intensidad,cortafuegos,fo1,fo2,fo3,fo4,tiempo,gap,lmbda,b,prom_pc] 
    titulo2 = 'm4'+' '+str(contador_cfuegos)+'cf '+str(n_sims)+'sims'+' fo:'+str(round(model.ObjVal,2))
    titulos = [titulo,titulo2]
    
    return results, fb_list, burned_map,titulo

#%% RESULTADOS 
#GRAFICAR MODELO
#--------------------------------------------------------------------------------------------------------------------
def model_map(lista_cf,burned_results,burned_map,FAG,titulos,b,lmbda):

    min_node = min(burned_results, key=burned_results.get)
    burned_results[min_node] = max(burned_map.values())
    
    
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
                                node_color = list(burned_results.values())
                                #node_color = list(metric_values.values())
                                )
    
    cf = nx.Graph()
    nx.draw_networkx_nodes(cf, pos = coord_pos,
                       node_size = 10,
                       nodelist = lista_cf,
                       node_shape='s',
                       node_color = 'b')
    
    """
    sp = nx.Graph()
    nx.draw_networkx_nodes(cf, pos = coord_pos,
                       node_size = 10,
                       nodelist = st_points,
                       node_shape='s',
                       node_color = 'purple')
    """
    titulo = titulos[0]
    folder = 'plots/'+titulo[0:2]+'/'
    
    plt.title(titulos[1])
    plt.colorbar(nc)
    plt.savefig(folder+titulo+'.png', bbox_inches='tight')
    plt.clf()
    #plt.show()
    

# EVALUAR SOL EN FO --------------------------------------------------------------------------------------------------------
def save_results(lista):
    
    df_results = pd.DataFrame()
    cols = ['modelo','nsims','intensidad','cf','fo1','fo2','fo3','fo4','ptime','gap','lambda','beta','10 wc avg','5 wc avg']
    dic = dict.fromkeys(cols,0)
    
    i = 0
    for key in dic:
        dic[key] = lista[i]
        i = i+1
    df_results = df_results.append(dic,ignore_index = True)
    
    with pd.ExcelWriter('resultados.xlsx',engine = 'openpyxl',mode='a',if_sheet_exists='overlay') as writer:
        df_results.to_excel(writer, sheet_name='Sheet1',startrow=writer.sheets['Sheet1'].max_row, header=None)

# EXPORTAR SOLUCIÓN  --------------------------------------------------------------------------------------------------------

def harvested(l,titulos): #funcion que pasa una lista de elementos a un archivo .csv que contiene a los cortafuegos
    datos=[np.insert(l,0,1)] #inserto el elemento 1 que corresponde al ano que necesita el archivo 
    if len(l)==0: #si no hay cortafuegos
        cols=['Year Number'] #creo solamente una columna correspondiente al ano
    else: #si hay cortafuegos
        colu=['Year Number',"Cell Numbers"] #creo 2 columnas
        col2=[""]*(len(l)-1) #creo el resto de columnas correspondientes a los otros nodos
        cols=colu+col2 #junto ambas columnas
    titulo = titulos[0]
    titulo2 = titulos[1]
    folder = 'plots/'+titulo[0:2]+'/'
    df = pd.DataFrame(datos,columns=cols) #creo el dataframe
    df.to_csv(folder+"harvested_"+titulo2+".csv",index=False) #guardo el dataframe
    
#bin_to_nod(lista_cf)
#%% GRAFICAR BOSQUE INICIAL

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

"""
cf = nx.Graph()
nx.draw_networkx_nodes(cf, pos = coord_pos,
                        node_size = 10,
                        nodelist = warmstart.keys(),
                        node_shape='s',
                        node_color = 'b')
"""

#titulo = 'Bosque inicial'
titulo = 'Bosque quemado '+str(n_sims)+'sims'
plt.title(titulo)
plt.colorbar(nc)
#plt.savefig('plots/'+titulo+'.png', bbox_inches='tight')
plt.show() 

fo1 = 0
for key in burned:
    fo1 = fo1 + burned[key]
    
fo1 = fo1/n_sims

fo2 = max(burned.values())
fo2 = fo2/NCells



print('fo1: ',fo1,' fo2: ',fo2)

#%% main

w = dict.fromkeys(AvailSet,1)



intensidad = 0.04
gap = 0.01
tlimit = 3600
fb_list = rd.sample(nodos,int(intensidad*NCells))
b = 0
lmbda = 0

for i in range(0,1):
    intensidad = i/100
    warmstart = fb_list
    print(f'{intensidad:-^100}')
    results = modelo1(NCells, sims, nodos, cmdoutput1, intensidad, st_points, warmstart, gap, tlimit)
    evaluation, fb_list, burned_results, titulo = results[0],results[1],results[2],results[3]
    save_results(evaluation)
    #model_map(fb_list, burned_results, burned, FAG, titulo, b, lmbda)
    #harvested(fb_list,titulo)
    
# =============================================================================
# for i in range(0,10):
#     alfa = 0.1-i/100
#     warmstart = fb_list
#     print(alfa)
#     results = modelo6(NCells, sims, nodos, cmdoutput1, alfa, st_points, warmstart, gap, tlimit)
#     evaluation, fb_list, burned_results, titulo = results[0],results[1],results[2],results[3]
#     save_results(evaluation)
# =============================================================================

# =============================================================================
# for i in range(4,15):
#     alfa = i/50
#     warmstart = fb_list
#     print(alfa)
#     results = modelo4(NCells, sims, nodos, cmdoutput1, alfa, st_points, warmstart, gap, tlimit)
#     evaluation, fb_list, burned_results, titulo = results[0],results[1],results[2],results[3]
#     save_results(evaluation)
# =============================================================================

# =============================================================================
# betas = [0.90,0.95]
# #betas=[0.9]
# lambdas = [0.75,0.5,0.25,0]
# for b in betas:
#     #for i in range(0,11):
#     for l in lambdas:
#         #b = 0.9
#         #lmbda = round(1-i/10,2)
#         lmbda = l
#         print(f'{lmbda:-^100}')
#         print(f'{b:-^100}')
#         intensidad = 0.04
#         warmstart = fb_list
#         results = modelo5(NCells, sims, nodos, cmdoutput1, intensidad, st_points, warmstart, gap, tlimit, b, lmbda)
#         evaluation, fb_list, burned_results, titulo = results[0],results[1],results[2],results[3]
#         save_results(evaluation)
#         model_map(fb_list, burned_results, burned, FAG, titulo, b, lmbda)
#         harvested(fb_list,titulo)
# =============================================================================

alarm(1000,150)

#%% random solution
intensidad = list(range(0,11))

nodos = list(MDG.nodes())
fb_list = []
for i in intensidad:
    fi = i/100
    print(fi)
    print(fb_list)
    harvested(fb_list,['rd_solutions',str(fi)])
    elements = rd.sample(nodos,int(0.01*NCells))
    for e in elements:
        nodos.remove(e)
    print(elements)
    fb_list = fb_list + elements

len(fb_list)
