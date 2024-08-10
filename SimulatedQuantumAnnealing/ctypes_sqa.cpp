#include "pch.h"
#include "ctypes_sqa.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>

using namespace std;
random_device rd;
mt19937 gen(rd());


vector<vector<double>> SimulatedQuantumAnnealing::init_jij()
{
    return vector<vector<double>>(N, vector<double>(N, 0.0));
}

int randint(int low, int high)
{
    uniform_int_distribution<> dist(low, high);
    return dist(gen);
}

double qubo_energy(const vector<vector<int>>& bits, const vector<vector<double>>& Q) {
    int L = bits.size();
    int N = bits[0].size();
    double energy = 0.0;

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                energy += Q[j][k] * bits[i][j] * bits[i][k];
            }
        }
    }

    return energy;
}

void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double max_dE = 1000.0) {
    int L = bits.size();
    int N = bits[0].size();
    for (int i = 0; i < L * N; ++i) {
        int layer = randint(0, L - 1);
        int bit = randint(0, N - 1);

        int current_bit = bits[layer][bit];
        bits[layer][bit] = 1 - bits[layer][bit];

        double dE = (1 - 2 * current_bit) * (inner_product(Q[bit].begin(), Q[bit].end(), bits[layer].begin(), 0.0));
        dE = max(-max_dE, min(dE, max_dE));
        if (static_cast<double>(randint(1, 1e8) / 1e8) >= exp(-dE / T)) {
            bits[layer][bit] = current_bit;
        }
    }
}

pair<vector<int>, double> SimulatedQuantumAnnealing::simulated_quantum_annealing(vector<vector<double>> Q)
{
    vector<vector<int>> bits(L, vector<int>(N));
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            bits[i][j] = randint(0, 1);
        }
    }
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < anneal_steps; ++i) {
        for (int j = 0; j < mc_steps; ++j) {
            monte_carlo_step(bits, Q, T);
        }
        T *= 0.95;
    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start).count();
    cout << "Execution time: " << duration << " microseconds" << endl;

    double min_energy = numeric_limits<double>::infinity();
    vector<int> best_bits;

    for (int layer = 0; layer < L; ++layer) {
        double layer_energy = qubo_energy({ bits[layer] }, Q);
        if (layer_energy <= min_energy) {
            min_energy = layer_energy;
            best_bits = bits[layer];
            // for(int bit=0;bit<N;++bit)cout << bits[layer][bit] << " ";
            // cout <<endl;
        }
    }

    min_energy = qubo_energy({ best_bits }, Q);
    return { best_bits, min_energy };
}