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

umbrales = np.arange(-1, 12, 1)

for umbral in umbrales:
    G = nx.Graph()
    for i in range(feat.shape[0]):
        G.add_node(i)
    for i in range(similarity_matrix.shape[0]):
        for j in range(i+1, similarity_matrix.shape[1]):
            if similarity_matrix[i,j] > umbral:
                G.add_edge(i, j)

    # eigenvalues, eigenvectors = np.linalg.eig(nx.adjacency_matrix(G).todense())
    eigenvalues, _ = it.potenciadeflacion(nx.adjacency_matrix(G).todense())
    
    # Correlación de listas de autovalores
    # eigenvalues_original, _ = np.linalg.eig(adjacency_matrix)
    eigenvalues_original, _ = it.potenciadeflacion(adjacency_matrix)
    correlation_eigenvalues = abs(correlacion(eigenvalues, eigenvalues_original))
    correlations_eigenvalues.append(correlation_eigenvalues)

    # Correlación de matrices de adyacencia aplanadas
    nuestra_flattened = nx.adjacency_matrix(G).todense().flatten()
    correlation_flat = correlacion(original_flattened, nuestra_flattened)
    correlations_flat.append(correlation_flat)
    print(f'Umbral: {umbral}, correlacion flat: {correlation_flat}, correlacion eigenvalues: {correlation_eigenvalues}')

# Printear maximos
for correlations in [correlations_flat, correlations_eigenvalues]:
    print(f'Max correlation: {np.max(correlations)}')
    print(f'Umbral: {umbrales[np.argmax(correlations)]}')

# Graficar
fig, ax1 = plt.subplots()

color = azul
ax1.set_xlabel('Umbral de similaridad')
ax1.set_ylabel('Correlación de autovalores', color=color)
ax1.plot(umbrales, correlations_eigenvalues, color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # create second y-axis

color = naranja
ax2.set_ylabel('Correlación de matrices de adyacencia', color=color)
ax2.plot(umbrales, correlations_flat, color=color, marker='o')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()