import numpy as np
import subprocess


def potenciadeflacion(A, eps=1e-12):
  # Crear el archivo input_data.txt y escribir los datos
  with open('input_data.txt', 'w') as f:
      f.write(f"{A.shape[0]} {A.shape[1]}\n")
      np.savetxt(f, A, newline="\n")
      f.write(f"{eps}")

  # Crear el archivo autovalores.txt y autovectores.txt como un archivo vacío
  with open('autovalores.txt', 'w') as f:
      pass
  with open('autovectores.txt', 'w') as f:
      pass


  # Compila el archivo C++ utilizando g++
  compilation_command = f'g++ PotenciaDeflacion.cpp -o potenciaDeflacion'
  subprocess.run(compilation_command, shell=True, check=True)

  # Ejecuta el programa C++
  cpp_program = 'potenciaDeflacion'
  subprocess.run(cpp_program, shell=True)

  autovalores = np.loadtxt('autovalores.txt')
  autovectores = np.loadtxt('autovectores.txt')

  return autovalores, autovectores