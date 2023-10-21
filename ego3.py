import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

# Vamos a hacer el calculo de correlacion con los dos métodos: aplanando la matriz de adyacencia y con los autovalores
correlations_flat = []
correlations_eigenvalues = []

# Creamos la matriz de adyacencia original
adjacency_matrix = np.zeros((feat.shape[0], feat.shape[0]))
for i, j in edges:
    i = edge_map[i]
    j = edge_map[j]
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1
original_flattened = adjacency_matrix.flatten()

umbrales = np.arange(-2, 12, 0.5)

for umbral in umbrales:
    G = nx.Graph()
    for i in range(feat.shape[0]):
        G.add_node(i)
    for i in range(similarity_matrix.shape[0]):
        for j in range(i+1, similarity_matrix.shape[1]):
            if similarity_matrix[i,j] > umbral:
                G.add_edge(i, j)

    eigenvalues, eigenvectors = np.linalg.eigh(nx.adjacency_matrix(G).todense())
    
    # Correlación de listas de autovalores
    eigenvalues_original, _ = np.linalg.eigh(adjacency_matrix)
    correlation_eigenvalues = (correlacion(eigenvalues, eigenvalues_original))
    correlations_eigenvalues.append(correlation_eigenvalues)

    # Correlación de matrices de adyacencia aplanadas
    nuestra_flattened = nx.adjacency_matrix(G).todense().flatten()
    correlation_flat = correlacion(original_flattened, nuestra_flattened)
    correlations_flat.append(correlation_flat)
    print(f'Umbral: {umbral}, correlación flat: {correlation_flat}, correlación eigenvalues: {correlation_eigenvalues}')

# Printear maximos
for correlations in [correlations_flat, correlations_eigenvalues]:
    max_correlation = max(correlations)
    max_percentile = np.argmax(correlations)
    print(f'Max correlation: {max_correlation}')
    print(f'Max percentile: {max_percentile}')

# Graficar maximos
for correlations in [correlations_flat, correlations_eigenvalues]:
    plt.plot(umbrales, correlations)
    plt.axvline(x=umbrales[np.argmax(correlations)], color='r', linestyle='--', label=f'Percentil {np.argmax(correlations)}')    
    plt.xlabel('Umbral de similaridad')
    plt.ylabel('Correlación')
    plt.legend()
    plt.show()
