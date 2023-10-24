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
        //cout << "Eigen: " << i << endl;
        VectorXd v_viejo = v;
        v = A * v;
        v.normalize();
        
        
        // Verificar convergencia
        if (check_criterio(v, v_viejo, eps)) {
            //cout << "Convergencia en la iteracion " << i << endl;
            break;
        }
        if(i == niter - 1){
         //cout << "No convergio" << endl;
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
        cout << "Autovalor " << i << ": " << result.first << endl;
        eigenvalues(i) = result.first;
        eigenvectors.col(i) = result.second;    
        J = J - (eigenvalues(i) * eigenvectors.col(i) * result.second.transpose());
    }
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
    eps = pow(10, -8);
    fin.close();
    int niter = 5000;
    cout << "Matriz A: " << A.size() << endl;
    cout << "Calculating eigenvalues and eigenvectors..." << endl;
    pair<Eigen::VectorXd, Eigen::MatrixXd> result = eigen(A, nrows, niter, eps);
    cout << "Done!" << endl;
    // Write eigenvalues and eigenvectors to the output file
    ofstream fout(output_autovalores);
    fout << result.first;
    fout.close();

    ofstream pout(output_autovectores);
    pout << result.second;
    pout.close();
          
    return 0;
}
