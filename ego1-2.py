import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import interfaz as it

plt.style.use('seaborn-v0_8-darkgrid')
naranja = '#F08228'
azul = '#1F77B4'
amarillo = '#F0E14A'
rojo = '#D62728'

def correlacion(x, y):
    mediaX = np.mean(x)
    mediaY = np.mean(y)
    return (np.dot((x-mediaX),(y-mediaY))) / np.sqrt(np.dot((x-mediaX),(x-mediaX)) * np.dot((y-mediaY),(y-mediaY)))

feat = np.loadtxt("./data/ego-facebook.feat")[:, 1:]  # Sin el primer atributo porque es el id del nodo
feat_with_first = np.loadtxt("./data/ego-facebook.feat")
edges = np.loadtxt("./data/ego-facebook.edges", dtype=int)

# Mapear ids de nodo a índices de matriz
edge_map = {}
for i, line in enumerate(feat_with_first):
    edge_map[line[0]] = i


similarity_matrix = feat @ feat.T

# Hisograma de similitudes
plt.hist(similarity_matrix.flatten(), bins=100)
plt.xlabel('Similitud')
plt.ylabel('Cantidad de pares de nodos')
plt.show()


# Acá tomé el percentil 90 por tomar uno cualquiera
percentile_similarity = np.percentile(similarity_matrix, 90)

G = nx.Graph()
for i in range(feat.shape[0]):
    G.add_node(i)
for i in range(similarity_matrix.shape[0]):
    for j in range(i+1, similarity_matrix.shape[1]):
        if similarity_matrix[i,j] > percentile_similarity:
            G.add_edge(i, j)
nx.draw(G, node_size=10)
plt.show()

# Correlación de matrices de adyacencia aplanadas
adjacency_matrix = np.zeros((feat.shape[0], feat.shape[0]))
for i, j in edges:
    i = edge_map[i]
    j = edge_map[j]
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1

original_flattened = adjacency_matrix.flatten()
nuestra_flattened = nx.adjacency_matrix(G).todense().flatten()

correlation = correlacion(original_flattened, nuestra_flattened)
print(f'Correlación de matrices de adyacencia aplanadas: {correlation}')





#Calculo con numpy optimizado (Rápido)
# autovalores_originales, _ = np.linalg.eig(adjacency_matrix)
# autovalores_nuestros, _ = np.linalg.eig(nx.adjacency_matrix(G).todense())

# Calculo con c++ (Muy lento)
autovalores_originales, _ = it.potenciadeflacion(adjacency_matrix)
autovalores_nuestros, _ = it.potenciadeflacion(nx.adjacency_matrix(G).todense())

correlation = abs(correlacion(autovalores_originales, autovalores_nuestros)) # Le puse abs porque me estaba dando un número complejo
print(f'Correlación de listas de autovalores: {correlation}')