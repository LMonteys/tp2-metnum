import subprocess
import numpy as np

def call(A):
   f = open('input_data.txt','w')
   f.write(f"{A.shape[0]} {A.shape[1]}\n")
   np.savetxt(f,A, newline="\n")
   f.write(f"{1e-12}")
   f.close()

   argv = ["g++", "Potencia+Deflacion.c++", "-o", "out", "input_data.txt", "output_data.txt"]
   subprocess.run(argv)

   f = open('output_data.txt', 'r')
   lineas = f.readlines()
   f.close()

   a = np.array((lineas[1]).split())

   n = len(A)
   v = np.zeros((n,n))

   for i in range(n):
      v[i] = np.array(lineas[i+3].split())

   return a, v
