import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import interfaz as iz
import numpy as np
import subprocess


matriz_adyacencia = np.loadtxt("./data/karateclub_matriz.txt")
n = len(matriz_adyacencia)



a, v = iz.potenciadeflacion(matriz_adyacencia)


centralidad = v[:,0]
if sum(centralidad) < 0:
    centralidad = centralidad * -1
centralidad_normalizada = centralidad / sum(centralidad)


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(centralidad[node], 2))) for node in G.nodes()}
node_colors = [centralidad[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1300, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#plt.title("CENTRALIDAD")
plt.show()


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(centralidad_normalizada[node], 2))) for node in G.nodes()}
node_colors = [centralidad_normalizada[node] for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=1300, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#plt.title("CENTRALIDAD NORMALIZADA")
plt.show()







diagonal = np.zeros(n)
for i in range(n):
    diagonal[i] = sum(matriz_adyacencia[i])
matriz_diagonal = np.diag(diagonal)

matriz_laplaciana = matriz_diagonal - matriz_adyacencia


a1, v1 = iz.potenciadeflacion(matriz_laplaciana)

a2, v2 = np.linalg.eig(matriz_laplaciana)

file = open("./data/karateclub_labels.txt", "r")
grupo = np.array(file.readlines()).astype(np.float64)
file.close()

correlaciones = np.zeros(n)
max_v = 0;
max_v_np = 0;
for i in range(n):
    correlaciones[i] = np.dot(v1[:,i], grupo) / np.sqrt(np.dot(v1[:,i], v1[:,i])*np.dot(grupo, grupo))
    if abs(correlaciones[max_v]) < abs(correlaciones[i]) and abs(a1[i]) > 1e-12:
        max_v = i

correlaciones_numpy = np.zeros(n)
for i in range(n):
    correlaciones_numpy[i] = np.dot(v2[:,i], grupo) / np.sqrt(np.dot(v2[:,i], v2[:,i])*np.dot(grupo, grupo))
    if abs(correlaciones_numpy[max_v_np]) < abs(correlaciones_numpy[i]) and abs(a2[i]) > 1e-12:
        max_v_np = i




#print(np.sort(abs(a1)), '\n', np.sort(abs(a2)))


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(v1[node, max_v], 2))) for node in G.nodes()}
node_colors = ['g' if v1[node, max_v] > 0 else 'y' for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_size=1300, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#plt.title(f"MEJOR AUTOVECTOR ASOCIADO AL AUTOVALOR: {a1[max_v]}")
plt.show()


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(round(v2[node, max_v_np], 2))) for node in G.nodes()}
node_colors = ['y' if v2[node, max_v_np] > 0 else 'g' for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=1300, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#plt.title(f"MEJOR AUTOVECTOR NUMPY ASOCIADO AL AUTOVALOR: {a2[max_v_np]}")
plt.show()


G = nx.from_numpy_array(matriz_adyacencia)
node_labels = {node:(str(node+1)+"|"+str(grupo[node])) for node in G.nodes()}
node_colors = ['y' if grupo[node] > 0 else 'g' for node in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=1300, node_color=node_colors, node_shape='s')
nx.draw_networkx_edges(G, pos)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
#plt.title("SEPARACION DEL CLUB")
plt.show()