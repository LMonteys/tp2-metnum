%%file eigen_types_iofile_test.cpp

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>

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

Eigen::VectorXd power_iteration(const Eigen::MatrixXd& A, int niter, double eps) {
    double a = 1.0;
    Eigen::VectorXd v = Eigen::VectorXd::Random(A.rows());
    v.normalize();
    
    for (int i = 0; i < niter; i++) {
        Eigen::VectorXd v_viejo = v;
        v = matrix_vector_multiplication(A, v);
        v.normalize();
        
        // Verificar convergencia
        if (check_criterio(v, v_viejo, eps)) {
            break;
        }
    }
    
    double a_calculado = v.dot(matrix_vector_multiplication(A,v));
    return std::make_pair(a_calculado, v);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> eigen(const Eigen::MatrixXd& A, int num, int niter, double eps) {
    Eigen::MatrixXd J = A;
    Eigen::VectorXd eigenvalues(num);
    Eigen::MatrixXd eigenvectors(A.rows(), num);

    for (int i = 0; i < num; i++) {
        std::pair<double, Eigen::VectorXd> result = power_iteration(J, niter, eps);
        eigenvalues(i) = result.first;
        eigenvectors.col(i) = result.second;
        J -= eigenvalues(i) * result.second * result.second.transpose();
    }

    return std::make_pair(eigenvalues, eigenvectors);
}


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_file output_file" << std::endl;
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];

    std::ifstream fin(input_file);
    if (!fin.is_open()) {
        std::cerr << "Error: could not open input file " << input_file << std::endl;
        return 1;
    }

    // Read matrix and vector from file
    int nrows, ncols;
    fin >> nrows >> ncols;

    MatrixXd A(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            fin >> A(i, j);
        }
    }
    
    int eps = 0;
    fin >> eps;
    fin.close();

    // Perform matrix-vector multiplication
    int niter = 10000
    std::pair<double, Eigen::VectorXd> result = eigen(A, nrows, niter, eps);

    // Write result to output file
    std::ofstream fout(output_file);
    if (!fout.is_open()) {
        std::cerr << "Error: could not open output file " << output_file << std::endl;
        return 1;
    }

    fout << result << std::endl;

    fout.close();

    return 0;
}
