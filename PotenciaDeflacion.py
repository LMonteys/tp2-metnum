import numpy as np

def check_criterio(v, v_viejo, eps):
    return np.linalg.norm(v - v_viejo) < eps


def power_iteration(A, niter=10_000, eps=1e-6):
    a = 1
    v = np.random.rand(A.shape[0])
    v = v / np.linalg.norm(v)

    for i in range(niter):
      v_viejo = v
      v = A @ v
      v = v / np.linalg.norm(v)
      if(check_criterio(v, v_viejo, eps)):
        print(i)
        break
    a = v.T @ A @ v  
    return a, v

def eigen(A, num=3, niter=10000, eps=1e-6):
    J = A.copy()
    eigenvalues = []
    eigenvectors = np.zeros((A.shape[0], num))

    for i in range(num):
      l, v = power_iteration(J, eps=1e-12)
      eigenvalues.append(l)
      eigenvectors[:, i] = v
      J = J.astype(float)
      J -= l * np.outer(v, v)
    return np.array(eigenvalues), eigenvectors
