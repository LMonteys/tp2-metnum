//%%file eigen_types_iofile_test.cpp

#include <iostream>
#include <fstream>
#include "./libs/eigen3/Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

VectorXd matrix_vector_multiplication(const MatrixXd& matrix, const VectorXd& vector) {
    return matrix * vector;
}

bool check_criterio(const Eigen::VectorXd& v, const Eigen::VectorXd& v_viejo, double eps) {
    // Calcula la norma de la diferencia entre los vectores v y v_viejo
    double norm_diff = (v - v_viejo).norm();
    // Compara la norma con el valor de eps y devuelve true si es menor
    return norm_diff < eps;
}

pair<double, Eigen::VectorXd> power_iteration(const Eigen::MatrixXd& A, int niter, double eps) {
Eigen::VectorXd v = Eigen::VectorXd::Random(A.rows());
v.normalize();

    for (int i = 0; i < niter; i++) {
        VectorXd v_viejo = v;
        v = A * v;
        v.normalize();
        
        
        // Verificar convergencia
        if (check_criterio(v, v_viejo, eps)) {
            break;
        }
        if(i == niter - 1){
        }
    }
            

    double a_calculado = v.dot(matrix_vector_multiplication(A, v));
    return make_pair(a_calculado, v);
}

pair<Eigen::VectorXd, Eigen::MatrixXd> eigen(const Eigen::MatrixXd& A, int num, int niter, double eps) {
    Eigen::MatrixXd J = A;
    Eigen::VectorXd eigenvalues(num);
    Eigen::MatrixXd eigenvectors(A.rows(), num);
    int progress = 0;  // Inicializar barra de progreso
    cout << "Calculo de autovalores y autovectores: [" << string(50, ' ') << "] 0%" << flush; 
    for (int i = 0; i < num; i++) {
        pair<double, Eigen::VectorXd> result = power_iteration(J, niter, eps);
        eigenvalues(i) = result.first;
        eigenvectors.col(i) = result.second;    
        J = J - (eigenvalues(i) * eigenvectors.col(i) * result.second.transpose());
        
        // actualizar barra de progreso: 
        int new_progress = (i + 1) * 50 / num;
        if (new_progress > progress) {
            progress = new_progress;
            cout << "\rCalculo de autovalores y autovectores: [" << string(progress, '=') << string(50 - progress, ' ') << "] " << progress * 2 << "%" << flush;
        }
    }
    cout << "\rCalculo de autovalores y autovectores: [" << string(50, '=') << "] 100%" << endl;  // completar barra de progreso
    return make_pair(eigenvalues, eigenvectors);
}


int main(int argc, char** argv) {
    const char* input_file = "input_data.txt";
    const char* output_autovalores = "autovalores.txt";
    const char* output_autovectores = "autovectores.txt";


    ifstream fin(input_file);
    if (!fin.is_open()) {
        cerr << "Error: could not open input file " << input_file << endl;
        return 1;
    }

    int nrows, ncols;
    double eps;
    fin >> nrows >> ncols;

    MatrixXd A(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> A(i, j);
        }
    }
    fin >> eps;
    fin.close();
    int niter = 5000;
    cout << "Calculando autovalores y autovectores..." << endl;
    pair<Eigen::VectorXd, Eigen::MatrixXd> result = eigen(A, nrows, niter, eps);
    cout << "Done!" << endl;
    ofstream fout(output_autovalores);
    fout << result.first;
    fout.close();

    ofstream pout(output_autovectores);
    pout << result.second;
    pout.close();
          
    return 0;
}
