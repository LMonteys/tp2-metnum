import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

plt.style.use('seaborn-darkgrid')

def correlacion(x, y):
    mediaX = np.mean(x)
    mediaY = np.mean(y)
    # if not(any(x) and any(y)):
    #     return 0
    return (np.dot((x-mediaX),(y-mediaY))) / np.sqrt(np.dot((x-mediaX),(x-mediaX)) * np.dot((y-mediaY),(y-mediaY)))

feat = np.loadtxt("./data/ego-facebook.feat")[:, 1:]  # Sin el primer atributo porque es el id del nodo
feat_with_first = np.loadtxt("./data/ego-facebook.feat")
edges = np.loadtxt("./data/ego-facebook.edges", dtype=int)

# Mapear ids de nodo a índices de matriz
edge_map = {}
for i, line in enumerate(feat_with_first):
    edge_map[line[0]] = i


# Creamos la matriz de adyacencia original
adjacency_matrix = np.zeros((feat.shape[0], feat.shape[0]))
for i, j in edges:
    i = edge_map[i]
    j = edge_map[j]
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1
original_flattened = adjacency_matrix.flatten()

umbrales = np.arange(-0.5, 11, 0.5)

#ks = [2, 4, 6, 8, 10, 25, 50, 100, 200, 319]
ks = [2, 3, 10, 25, 50, 100, 319]
correlations_k = {}

feat_original = feat
for k in ks:
    correlations_flat = []
    correlations_eigenvalues = []
    featCentered = feat - np.mean(feat, axis=0)
    matrizCovarianza = np.cov(featCentered, rowvar=False)
    _, autovectores = np.linalg.eigh(matrizCovarianza) # reemplazar por nuestra implementación en C++
    feat = np.matmul(feat, autovectores[:, -k:])
    similarity_matrix = feat @ feat.T
    for umbral in umbrales:
        G = nx.Graph()
        for i in range(feat.shape[0]):
            G.add_node(i)
        for i in range(similarity_matrix.shape[0]):
            for j in range(i+1, similarity_matrix.shape[1]):
                if similarity_matrix[i,j] > umbral:
                    G.add_edge(i, j)

        eigenvalues, eigenvectors = np.linalg.eigh(nx.adjacency_matrix(G).todense()) # Reemplazar por nuestra implementación en C++
        
        # Correlación de listas de autovalores
        eigenvalues_original, _ = np.linalg.eigh(adjacency_matrix) # Reemplazar por nuestra implementación en C++
        correlation_eigenvalues = (correlacion(eigenvalues, eigenvalues_original))
        correlations_eigenvalues.append(correlation_eigenvalues)

        # Correlación de matrices de adyacencia aplanadas
        nuestra_flattened = nx.adjacency_matrix(G).todense().flatten()
        correlation_flat = correlacion(original_flattened, nuestra_flattened)
        correlations_flat.append(correlation_flat)
        print(f'K: {k}, Umbral: {umbral}, correlacion flat: {correlation_flat}, correlacion autovalores: {correlation_eigenvalues}')


    correlations_k[k] = (correlations_flat, correlations_eigenvalues)
    feat = feat_original

# Printear maximos
# for correlations in [correlations_flat, correlations_eigenvalues]:
#     max_correlation = max(correlations)
#     max_percentile = np.argmax(correlations)
#     print(f'Max correlation: {max_correlation}')
#     print(f'Max percentile: {max_percentile}')

# Graficar maximos
for k in ks:
    # correlación flat
    plt.plot(umbrales, correlations_k[k][0], marker='o', label=f'k={k}')
    #plt.axvline(x=umbrales[np.argmax(correlations)], color='r', linestyle='--', label=f'Percentil {np.argmax(correlations)}')    
plt.xlabel('Umbral de similaridad')
plt.ylabel('Correlación de las matrices de adyacencia')
plt.legend()
plt.show()

# correlación eigenvalues
for k in ks:
    plt.plot(umbrales, correlations_k[k][1], marker='o', label=f'k={k}')
#plt.axvline(x=umbrales[np.argmax(correlations)], color='r', linestyle='--', label=f'Percentil {np.argmax(correlations)}')
plt.xlabel('Umbral de similaridad')
plt.ylabel('Correlación de los autovalores')
plt.legend()
plt.show()
