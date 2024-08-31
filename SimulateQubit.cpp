#include "..\..\..\local\eigen-3.4.0\Eigen\Core"
#include <complex>
#include <iostream>
#include <vector>

using namespace Eigen;
using namespace std;

// パラメータ設定
const int n_qubits = 5;
const double hbar = 1.0;
const double T = 10.0;
const double dt = 0.01;
const int num_steps = static_cast<int>(T / dt);

// パウリ行列の定義
Matrix2cd sigma_x, sigma_z, identity;
void initialize_pauli_matrices() {
    sigma_x << 0, 1, 1, 0;
    sigma_z << 1, 0, 0, -1;
    identity = Matrix2cd::Identity();
}

MatrixXcd kroneckerProduct(const MatrixXcd& A, const MatrixXcd& B) {
    MatrixXcd result = MatrixXcd::Zero(A.rows() * B.rows(), A.cols() * B.cols());

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            result.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
        }
    }

    return result;
}

// テンソル積の計算
MatrixXcd kron_n(const vector<Matrix2cd>& matrices) {
    MatrixXcd result = matrices[0];
    for (size_t i = 1; i < matrices.size(); ++i) {
        result = kroneckerProduct(result, matrices[i]).eval();
    }
    return result;
}

// ハミルトニアンの設定
MatrixXcd H0, Hp;
void initialize_hamiltonians(VectorXd& J, VectorXd& h) {
    H0 = MatrixXcd::Zero(pow(2, n_qubits), pow(2, n_qubits));
    Hp = MatrixXcd::Zero(pow(2, n_qubits), pow(2, n_qubits));

    for (int i = 0; i < n_qubits; ++i) {
        vector<Matrix2cd> matrices(n_qubits, identity);
        matrices[i] = sigma_x;
        H0 -= kron_n(matrices);
    }

    for (int i = 0; i < n_qubits; ++i) {
        vector<Matrix2cd> matrices(n_qubits, identity);
        matrices[i] = sigma_z;
        Hp -= h(i) * kron_n(matrices);
    }

    for (int i = 0; i < n_qubits - 1; ++i) {
        vector<Matrix2cd> matrices(n_qubits, identity);
        matrices[i] = sigma_z;
        matrices[i + 1] = sigma_z;
        Hp -= J(i) * kron_n(matrices);
    }
}

// 時間依存ハミルトニアンの計算
MatrixXcd hamiltonian(double t) {
    double A = 1 - t / T;
    double B = t / T;
    return A * H0 + B * Hp;
}

// ルンゲ＝クッタ法での時間発展
VectorXcd runge_kutta_step(double t, const VectorXcd& psi, double dt) {
    auto time_derivative = [&](double t, const VectorXcd& psi) -> VectorXcd {
        MatrixXcd H_t = hamiltonian(t);
        return -std::complex<double>(0, 1) / hbar * H_t * psi;
    };

    VectorXcd k1 = time_derivative(t, psi);
    VectorXcd k2 = time_derivative(t + dt / 2, psi + dt / 2 * k1);
    VectorXcd k3 = time_derivative(t + dt / 2, psi + dt / 2 * k2);
    VectorXcd k4 = time_derivative(t + dt, psi + dt * k3);

    return psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

// メイン関数
int main() {
    initialize_pauli_matrices();

    // 相互作用の強度と磁場の強度をランダムに設定
    VectorXd J = VectorXd::Random(n_qubits - 1) * 20;
    VectorXd h = VectorXd::Random(n_qubits) * 20;

    initialize_hamiltonians(J, h);

    // 初期状態 (重ね合わせ状態)
    VectorXcd psi_0 = VectorXcd::Constant(pow(2, n_qubits), 1.0 / sqrt(pow(2, n_qubits)));

    // シミュレーションの実行
    vector<VectorXcd> psi_values(num_steps, VectorXcd::Zero(pow(2, n_qubits)));
    psi_values[0] = psi_0;

    for (int i = 1; i < num_steps; ++i) {
        double t = i * dt;
        psi_values[i] = runge_kutta_step(t, psi_values[i - 1], dt);
    }

    // 結果の出力
    for (int j = 0; j < pow(2, n_qubits); ++j) {
        cout << "Probability for state " << j << ": ";
        for (int i = 0; i < num_steps; ++i) {
            cout << abs(psi_values[i](j)) * abs(psi_values[i](j)) << " ";
        }
        cout << endl;
    }
    
    return 0;
}
