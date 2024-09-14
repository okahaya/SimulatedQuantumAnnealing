#include <iostream>
#include <vector>
#include <random>

using namespace std;

thread_local std::mt19937 rng(std::random_device{}());

double qubo_energy(const vector<int>& bits, const vector<vector<double>>& Q) {
    int N = bits.size();
    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int k = j; k < N; ++k) {
            energy += Q[j][k] * bits[j] * bits[k];
        }
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
    double cool = pow(10,-10/(double(anneal_steps)-1));
    return cool;
}

double init_gamma(int mc_steps){
    double gamma = pow(10,10/(double(mc_steps)-1));
    return gamma;
}

void showProgressBar(int progress, int total) {
    int barWidth = 50; // 進捗バーの幅

    float progressRatio = (float)progress / total;
    int pos = barWidth * progressRatio;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progressRatio * 100.0) << " %\r";
    std::cout.flush();
}

// エネルギー差分の計算関数
double calculate_delta_E(const vector<vector<int>>& bits, const vector<vector<double>>& Q, int layer, int bit_index, int new_bit_value, double Bt) {
    double delta_E = 0.0;
    int N = bits[0].size();
    int L = bits.size();
    const vector<int>& bits_layer = bits[layer];
    int old_bit_value = bits_layer[bit_index];
    int delta_bit = new_bit_value - old_bit_value;
    if (delta_bit == 0) return 0.0;

    // 古典的な相互作用のエネルギー差分を計算
    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0) {
            delta_E += Q[bit_index][j] * bits_layer[j];
        }
    }
    delta_E *= delta_bit;

    // トロッター層間の相互作用のエネルギー差分
    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;

    delta_E += (Bt / L) * delta_bit * ((2 * bits[next_layer][bit_index] - 1) + (2 * bits[prev_layer][bit_index] - 1));

    return delta_E;
}

// エネルギー差分の計算関数
double calculate_delta_E_classical(const vector<int>& bits, const vector<vector<double>>& Q, int bit_index, int new_bit_value) {
    double delta_E = 0.0;
    int N = bits.size();
    int old_bit_value = bits[bit_index];
    int delta_bit = new_bit_value - old_bit_value;
    if (delta_bit == 0) return 0.0;

    // 古典的な相互作用のエネルギー差分を計算
    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0) {
            delta_E += Q[bit_index][j] * bits[j];
        }
    }
    delta_E *= delta_bit;


    return delta_E;
}