#%%
"""
Created on Mon Nov 28 12:33:33 2022

@author: Matias
"""

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

#%% save results
def save_results(lista,path):
    
    df_results = pd.DataFrame()
    cols = ['modelo','nsims','intensidad','fo1','fo2','fo3','10 wc avg','5 wc avg','beta','lambda']
    dic = dict.fromkeys(cols,0)
    
    i = 0
    for key in dic:
        dic[key] = lista[i]
        i = i+1
    df_results = df_results.append(dic,ignore_index = True)
    
    with pd.ExcelWriter(path,engine = 'openpyxl',mode='a',if_sheet_exists='overlay') as writer:
        df_results.to_excel(writer, sheet_name='Sheet1',startrow=writer.sheets['Sheet1'].max_row, header=None)
#%% read sims
def read_sims(n_sims,ruta):
    
    
    Folder = os.getcwd()
    
    FBPlookup = Folder + '/Sub40x40/fbp_lookup_table.csv'
    
    ForestFile = Folder + '/Sub40x40/Forest.asc'
    
    FBPDict, ColorsDict = ReadDataPrometheus.Dictionary(FBPlookup)
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
    
    Adjacents = {}
    for k in range(NCells):
        aux = set([])
        for i in setDir:
            if AdjCells[k][i] != None:
                if AdjCells[k][i][0] in AvailSet :
                    aux = aux | set(AdjCells[k][i])
        Adjacents[k + 1] = aux & AvailSet
    
    FAG = nx.Graph()
    FAG.add_nodes_from(list(AvailSet))
    
    coord_pos = dict()
    for i in AvailSet:
        coord_pos[i] = CoordCells[i-1]
    
    ColorsAvail = {}
    for i in AvailSet:
        FAG.add_edges_from([(i,j) for j in Adjacents[i]])
        ColorsAvail[i] = Colors[i-1]
    
    Folder = 'C:/Users/Matias/Documents/proyecto_magister/simulador/Cell2Fire-main/results/'+str(ruta)
    #Folder = 'C:/Users/Matias/Documents/proyecto_magister/simulador/Cell2Fire-main/results/Sub40x40'
    
    cmdoutput1 = os.listdir(Folder + '/Messages')
    if ".DS_Store" in cmdoutput1:
        idx = cmdoutput1.index('.DS_Store') # Busca el indice que hay que eliminar
        del cmdoutput1[idx]
    if "desktop.ini" in cmdoutput1:
        idx = cmdoutput1.index('desktop.ini') # Busca el indice que hay que eliminar
        del cmdoutput1[idx]
    pathmessages = Folder + '/Messages/'
    
    MDG = nx.MultiDiGraph()
    
    avail = list(AvailSet)
    st_points = []
    sim_edges = []
    
    sims_list = list(range(1,n_sims+1))
    f3_dic = dict.fromkeys(sims_list,0)
    burned = dict.fromkeys(AvailSet,0)
    
    n_sims = 0
    # Se cargan los incendios y se agregan iterativamente a MDG
    for k in cmdoutput1:
        n_sims = n_sims+1
        # Leemos la marca
        H = nx.read_edgelist(path = pathmessages + k,
                             delimiter=',',
                             create_using = nx.DiGraph(),
                             nodetype = int,
                             data = [('time', float), ('ros', float)])
        
        
        
        sim_edges.append(list(H.edges))
        
        MDG.add_weighted_edges_from(H.edges(data='time'), weight='ros')
        
        try:
            st_points.append(list(H.edges)[0][0])
        except IndexError:
            st_points.append(avail[rd.randint(0, len(avail)-1)])
            #print(k)
        
        contador = dict.fromkeys(list(H.nodes),0)
        for e in H.edges:
            for n in list(H.nodes):
                if n == e[1] and contador[n] == 0:
                    contador[n]=contador[n]+1
                    burned[e[1]]=burned[e[1]]+1
                    
        f3_dic[n_sims] = sum(contador[n] for n in contador)+1
    
    #crea un diccionario con la cantidad de veces que se quema cada nodo
    
    for point in st_points:
        burned[point] = burned[point]+1
        
    for key in burned:
        burned[key] = burned[key]
        
    fo1 = 0
    for key in burned:
        fo1 = fo1 + burned[key]
        
    fo1 = fo1/n_sims

    fo2 = max(burned.values())
    fo2 = fo2/n_sims

    fo3 = max(f3_dic.values())
    fo3 = fo3/NCells
    
    lista_aux = list(f3_dic.values())
    lista_aux = sort(lista_aux)
    
    wc10 = 0
    for i in range(10):
        wc10 = wc10 + lista_aux[-i-1]
    wc10 = wc10/10
    
    wc5 = 0
    for i in range(5):
        wc5 = wc5 + lista_aux[-i-1]
    wc5 = wc5/5
    
    modelo = ruta[0:2]
    if modelo == 'm4':
        intensidad = 0.04
        txt = Folder+'/LogFile.txt'
        with open(txt) as f:
            for line in f:
                if 'beta' in line:
                    pos = line.find('beta')
                    pos2 = line.find('lambda')
                    pos3 = line.find('.csv')
                    b = float(line[pos+4:pos2-1])
                    l = float(line[pos2+6:pos3])
    else:
        intensidad = int(ruta.partition('/')[2])/100
        b = 0
        l = 0

    resultados = [modelo,n_sims,intensidad,fo1,fo2,fo3,wc10,wc5,b,l]
    
    return resultados
#%% plot bosque inicial   
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

#%%

folders = list(range(0,11))
folders2 = list(range(1,9))

models = ['m1','m2','m3','m4']
models2 = ['m3','m4']
random_sol = ['rd']

for m in random_sol:
    if m == 'm4':
        f = folders2
        
    else:
        f = folders
    
    for i in f:
        ruta = m+'/'+str(i)
        print('*'*30,ruta,'*'*30)
        results = read_sims(10000, ruta)
        save_results(results,'final_results.xlsx')

#%%
