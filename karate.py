import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
pt = __import__("Potencia+Deflacion")

file = open("./data/karateclub_matriz.txt", "r")
filas = file.readlines()
file.close()

n = len(filas)

matriz_adyacencia = np.zeros((n,n))
for i in range(n):
    
    matriz_adyacencia[i] = filas[i].split()


a, v = pt.power_iteration(matriz_adyacencia, eps=1e-12)


centralidad = v
centralidad_normalizada = v / sum(v)


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(centralidad[node], 2))) for node in G.nodes()}
node_colors = [centralidad[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
plt.title("CENTRALIDAD")
plt.show()


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(centralidad_normalizada[node], 2))) for node in G.nodes()}
node_colors = [centralidad_normalizada[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
plt.title("CENTRALIDAD NORMALIZADA")
plt.show()







diagonal = np.zeros(n)
for i in range(n):
    diagonal[i] = sum(matriz_adyacencia[i])
matriz_diagonal = np.diag(diagonal)

matriz_laplaciana = matriz_diagonal - matriz_adyacencia


a1, v1 = pt.eigen(matriz_laplaciana, num=n, eps=1e-12)

a2, v2 = np.linalg.eig(matriz_laplaciana)

file = open("./data/karateclub_labels.txt", "r")
grupo = np.array(file.readlines()).astype(np.float64)
file.close()

correlaciones = np.zeros(n)
max_v = 0;
max_v_np = 0;
for i in range(n):
    correlaciones[i] = np.dot(v1[:,i], grupo) / np.sqrt(np.dot(v1[:,i], v1[:,i])*np.dot(grupo, grupo))
    if abs(correlaciones[max_v]) < abs(correlaciones[i]):
        max_v = i

correlaciones_numpy = np.zeros(n)
for i in range(n):
    correlaciones_numpy[i] = np.dot(v2[:,i], grupo) / np.sqrt(np.dot(v2[:,i], v2[:,i])*np.dot(grupo, grupo))
    if abs(correlaciones_numpy[max_v_np]) < abs(correlaciones_numpy[i]):
        max_v_np = i







G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(v1[node, max_v], 2))) for node in G.nodes()}
node_colors = [v1[node, max_v] / abs(v1[node, max_v]) for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
plt.title("MEJOR AUTOVECTOR")
plt.show()


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(v2[node, max_v_np], 2))) for node in G.nodes()}
node_colors = [v2[node, max_v_np] / abs(v2[node, max_v_np]) for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1100, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
plt.title("MEJOR AUTOVECTOR NUMPY")
plt.show()