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
        v = matrix_vector_multiplication(A, v);
        v.normalize();
        
        // Verificar convergencia
        if (check_criterio(v, v_viejo, eps)) {
            break;
        }
    }
               

    double a_calculado = v.dot(matrix_vector_multiplication(A, v));
    return make_pair(a_calculado, v);
}

    pair<Eigen::VectorXd, Eigen::MatrixXd> eigen(const Eigen::MatrixXd& A, int num, int niter, double eps) {
    Eigen::MatrixXd J = A;
    Eigen::VectorXd eigenvalues(num);
    Eigen::MatrixXd eigenvectors(A.rows(), num);

    for (int i = 0; i < num; i++) {
        pair<double, Eigen::VectorXd> result = power_iteration(J, niter, eps);
        eigenvalues(i) = result.first;
        eigenvectors.col(i) = result.second;    
        J = J - (eigenvalues(i) * eigenvectors.col(i) * result.second.transpose());
    }
    return make_pair(eigenvalues, eigenvectors);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " input_file output_file" << endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    ifstream fin(input_file);
    if (!fin.is_open()) {
        cerr << "Error: could not open input file " << input_file << endl;
        return 1;
    }

    // Read matrix from file
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
    int niter = 10000;

    pair<Eigen::VectorXd, Eigen::MatrixXd> result = eigen(A, nrows, niter, eps);

    // Write eigenvalues and eigenvectors to the output file
    ofstream fout(output_file);
    if (!fout.is_open()) {
        cerr << "Error: could not open output file " << output_file << endl;
        return 1;
    }

    fout << "Eigenvalues:\n" << result.first << "\n";
    fout << "Eigenvectors:\n" << result.second << "\n";
            

    fout.close();

    return 0;
}
