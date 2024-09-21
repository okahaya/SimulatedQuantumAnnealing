#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

thread_local std::mt19937 rng(std::random_device{}());

double qubo_energy(const vector<int>& bits, const vector<vector<double>>& Q) {
    int N = bits.size();
    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        if (bits[j] == 0) continue;
        for (int k = j; k < N; ++k) {
            if (bits[k] == 0) continue;
            energy += Q[j][k];
        }
    }
    return energy;
}

double driver_energy(const vector<vector<int>>& bits, const int layer) {
    int N = bits[0].size();
    int L = bits.size();

    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;

    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        if(bits[layer][j] == 0)continue;
        energy += bits[next_layer][j] + bits[prev_layer][j];
    }
    return energy;
}

int randint(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

double rand_real() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}
    
double init_coolingrate(int anneal_steps){
    if (anneal_steps <= 1) return 1.0;
    double cool = pow(0.1, 10.0 /(double(anneal_steps)-1));
    return cool;
}

double init_gamma(int anneal_steps){
    if (anneal_steps <= 1) return 1.0;
    double gamma = pow(0.1, 10.0 /(double(anneal_steps)-1));
    return gamma;
}

void showProgressBar(int current, int total, const std::string& label) {
    int bar_width = 50;
    float progress = (float)current / total;

    std::cout << label << " [";
    int pos = bar_width * progress;
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

double calculate_delta_E(const vector<vector<int>>& bits, const vector<vector<double>>& Q, int layer, int bit_index, int new_bit_value, double At, double Bt) {
    double delta_E = 0.0;
    int N = bits[0].size();
    int L = bits.size();
    const vector<int>& bits_layer = bits[layer];
    int delta_bit = 2*new_bit_value - 1;

    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0) {
            delta_E += Q[bit_index][j] * bits_layer[j];
        }
    }
    delta_E *= delta_bit;
    delta_E *= At;

    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;

    delta_E += (Bt / L) * delta_bit * ((2 * bits[next_layer][bit_index] - 1) + (2 * bits[prev_layer][bit_index] - 1));

    return delta_E;
}

double calculate_delta_E_classical(const vector<int>& bits, const vector<vector<double>>& Q, int bit_index, int new_bit_value) {
    double delta_E = 0.0;
    int N = bits.size();
    int old_bit_value = bits[bit_index];
    int delta_bit = new_bit_value - old_bit_value;
    if (delta_bit == 0) return 0.0;

    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0) {
            delta_E += Q[bit_index][j] * bits[j];
        }
    }
    delta_E *= delta_bit;


    return delta_E;
}

vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    if (matrix.empty()) return {};

    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<double>> transposed(cols, vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,const std::vector<std::pair<std::vector<int>,int>> nhot_memo) {
    int k = 1;
    for (int q = 0; q < nhot_memo.size(); ++q) {
        int n = nhot_memo[q].second;
        int start = nhot_memo[q].first[0];
        int end = nhot_memo[q].first[nhot_memo[q].first.size()];
        for (int i = start; i < end; ++i) {
            Q[i][i] = k*(1 - 2 * n);
            for (int j = i + 1; j < end; ++j) {
                Q[i][j] = k*2;
                Q[j][i] = k*2;
            }
        }
    }
}