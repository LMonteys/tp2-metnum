import numpy as np
import subprocess


def potenciadeflacion(A):
  # Crear el archivo input_data.txt y escribir los datos
  with open('input_data.txt', 'w') as f:
      f.write(f"{A.shape[0]} {A.shape[1]}\n")
      np.savetxt(f, A, newline="\n")
      f.write(f"{1e-12}")

  # Crear el archivo autovalores.txt y autovectores.txt como un archivo vacío
  with open('autovalores.txt', 'w') as f:
      pass
  with open('autovectores.txt', 'w') as f:
      pass


  # Compila el archivo C++ utilizando g++
  compilation_command = f'g++ eigen_types_iofile_test.cpp -o my_cpp_program'
  subprocess.run(compilation_command, shell=True, check=True)

  # Ejecuta el programa C++
  cpp_program = './my_cpp_program'
  subprocess.run(cpp_program, shell=True)

  autovalores = np.loadtxt('autovalores.txt')
  autovectores = np.loadtxt('autovectores.txt')

A = np.array([[2, 1, 4], [1, 3, 0], [4, 0, 4]], dtype=np.float64)
potenciadeflacion(A)
