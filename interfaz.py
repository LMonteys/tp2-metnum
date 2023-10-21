from IPython.core.hooks import subprocess
import subprocess
import numpy as np

A = np.array([[1, 2, 3], [2, 5, 6], [3, 6, 9]], dtype=np.float64)
!rm input_data.txt
with open('input_data.txt','a') as f:
   f.write(f"{A.shape[0]} {A.shape[1]}\n")
   np.savetxt(f,A, newline="\n")
   f.write(f"{1e-12}")


!rm eigen_ctypes_test.so #borramos por las dudas
!g++ -shared -fPIC -Llibdl -o eigen_ctypes_test.so Potencia+Deflacion.c++
!ls #chequeamos que este la lib
!pwd #nos fijamos en que carpeta esta

argv = ["/content/eigen_ctypes_test.so", "./out", "input_data.txt", "output_data.txt"]
subprocess.run(argv)

