import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

feat = np.loadtxt("..//ego-facebook.feat")[:, 1:]  # Sin el primer atributo porque es el id del nodo
feat_with_first = np.loadtxt("..//ego-facebook.feat")
edges = np.loadtxt("..//ego-facebook.edges", dtype=int)


# Mapear ids de nodo a índices de matriz
edge_map = {}
for i, line in enumerate(feat_with_first):
    edge_map[line[0]] = i


similarity_matrix = feat @ feat.T

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

correlation = np.corrcoef(original_flattened, nuestra_flattened)[0,1] # ¿¿¿A esto se refieren con "correlación"??? ¿O a producto interno?
print(f'Correlación de matrices de adyacencia aplanadas: {correlation}')



# Correlación de listas de autovalores
eigenvalues, _ = np.linalg.eig(adjacency_matrix) # ¿Usar matriz de adyacencia o laplaciana?
eigenvalues_original = np.linalg.eigvals(adjacency_matrix) # ¿Usar matriz de adyacencia o laplaciana?
correlation = abs(np.corrcoef(eigenvalues, eigenvalues_original)[0,1])  # ¿¿¿A esto se refieren con "correlación"??? ¿O a producto interno?
                                                                        # Le puse abs porque me estaba dando un número complejo
print(f'Correlación de listas de autovalores: {correlation}')