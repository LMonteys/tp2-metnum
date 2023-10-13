import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
pt = __import__("Potencia+Deflacion")

file = open("Data\\karateclub_matriz.txt", "r")
filas = file.readlines()
file.close()

n = len(filas)

matriz_adyacencia = np.zeros((n,n))
for i in range(n):
    matriz_adyacencia[i] = filas[i].split()



'''
file = open("Data\\matriz_adyacencia.txt", "w")
for f in matriz_adyacencia.astype(int):
    for e in f:
        file.write(f'{e}')
        file.write(' ')
    file.write('\n')
file.close
'''


a, v = pt.power_iteration(matriz_adyacencia, eps=1e-12)


centralidad = v
centralidad_normalizada = v / sum(v)


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(centralidad_normalizada[node], 2))) for node in G.nodes()}
node_colors = [centralidad_normalizada[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
plt.show()






diagonal = np.zeros(n)
for i in range(n):
    diagonal[i] = sum(matriz_adyacencia[i])
matriz_diagonal = np.diag(diagonal)

matriz_laplaciana = matriz_diagonal - matriz_adyacencia

a, v = pt.eigen(matriz_laplaciana, num=n, eps=1e-12)


file = open("Data\\karateclub_labels.txt", "r")
grupo = np.array(file.readlines()).astype(np.float64)
file.close()

correlaciones = np.zeros(n)
for i in range(n):
    correlaciones[i] = np.dot(v[i], grupo) / np.sqrt(np.dot(v[i], v[i])*np.dot(grupo, grupo))

