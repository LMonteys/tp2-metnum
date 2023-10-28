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


# Creamos la matriz de adyacencia original
adjacency_matrix = np.zeros((feat.shape[0], feat.shape[0]))
for i, j in edges:
    i = edge_map[i]
    j = edge_map[j]
    adjacency_matrix[i, j] = 1
    adjacency_matrix[j, i] = 1
original_flattened = adjacency_matrix.flatten()

umbrales = np.arange(-1, 11, 1)

ks = [2, 5, 50, 100, 319]
correlations_k = {}

feat_original = feat
for k in ks:
    correlations_flat = []
    correlations_eigenvalues = []
    featCentered = feat - np.mean(feat, axis=0)
    matrizCovarianza = np.cov(featCentered, rowvar=False)
    
    # Cálculo de autovectores con numpy
    #autovalores, autovectores = np.linalg.eig(matrizCovarianza)
    ## Ordenar autovectores según autovalores de mayor a menor
    #indices = np.argsort(autovalores)[::-1]
    #autovectores = autovectores[:, indices]
    #autovalores = autovalores[indices]

    # Cálculo de autovectores con C++
    _, autovectores = it.potenciadeflacion(matrizCovarianza)

    feat = np.matmul(feat, autovectores[:, :k])
    #feat = np.matmul(feat, autovectores[:, -k:])
    similarity_matrix = feat @ feat.T
    for umbral in umbrales:
        G = nx.Graph()
        for i in range(feat.shape[0]):
            G.add_node(i)
        for i in range(similarity_matrix.shape[0]):
            for j in range(i+1, similarity_matrix.shape[1]):
                if similarity_matrix[i,j] > umbral:
                    G.add_edge(i, j)

        # eigenvalues, eigenvectors = np.linalg.eig(nx.adjacency_matrix(G).todense())  # Usando numpy
        eigenvalues, eigenvectores = it.potenciadeflacion(nx.adjacency_matrix(G).todense()) # Usando nuestra implementación en C++:
        
        # Correlación de listas de autovalores
        # eigenvalues_original, _ = np.linalg.eig(adjacency_matrix) # Usando numpy
        eigenvalues_original, _ = it.potenciadeflacion(adjacency_matrix) # Usando nuestra implementación en C++:
        correlation_eigenvalues = abs(correlacion(eigenvalues, eigenvalues_original))
        correlations_eigenvalues.append(correlation_eigenvalues)

        # Correlación de matrices de adyacencia aplanadas
        nuestra_flattened = nx.adjacency_matrix(G).todense().flatten()
        correlation_flat = correlacion(original_flattened, nuestra_flattened)
        correlations_flat.append(correlation_flat)
        print(f'K: {k}, Umbral: {umbral}, correlacion flat: {correlation_flat}, correlacion autovalores: {correlation_eigenvalues}')


    correlations_k[k] = (correlations_flat, correlations_eigenvalues)
    feat = feat_original


# Graficar maximos
for k in ks:
    # correlación flat
    plt.plot(umbrales, correlations_k[k][0], marker='o', label=f'k={k}')
plt.axhline(y=0.10987884773143827, color='r', linestyle='--')  # para marcar el maximo encontrado en ego3.py
plt.xlabel('Umbral de similaridad', fontsize=19)
plt.ylabel('Correlación de las matrices de adyacencia', fontsize=19)
plt.legend(fontsize=23)
plt.tick_params(labelsize=19)
plt.show()

# correlación eigenvalues
for k in ks:
    plt.plot(umbrales, correlations_k[k][1], marker='o', label=f'k={k}')
plt.axhline(y=0.592470078611196, color='r', linestyle='--')  # para marcar el maximo encontrado en ego3.py
plt.xlabel('Umbral de similaridad', fontsize=19)
plt.ylabel('Correlación de los autovalores', fontsize=19)
plt.legend(fontsize=23)
plt.tick_params(labelsize=19)
plt.show()
