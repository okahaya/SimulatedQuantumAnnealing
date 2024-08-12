#include "SimulatedQuantumAnnealing.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <random>
#include <chrono>
#include <numeric>
#include <omp.h>

using namespace std;

random_device seed_gen;
mt19937 engine(seed_gen());


vector<vector<double>> SimulatedQuantumAnnealing::init_jij()
{
    return vector<vector<double>>(N, vector<double>(N, 0.0));
}

int randint(int low, int high)
{
    uniform_int_distribution<> dist(low, high);
    return dist(engine);
}

double qubo_energy(const vector<int>& bits, const vector<vector<double>>& Q) {
    int N = bits.size();
    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int k = 0; k < N; ++k) {
            energy += Q[j][k] * bits[j] * bits[k];
        }
    }
    return energy;
}
// double qubo_diff(vector<int>& bits,int bit, const vector<vector<double>>& Q){
//     double diff = 0;
//     int N = bits.size();
//     for(int i=0;i<N;++i){
//         diff += (Q[bit][i]+Q[i][bit])*bits[i];
//     }
//     diff -= Q[bit][bit]*bits[bit];
//     return diff;
// }//怪しい

void monte_carlo_step(vector<int>& bits, const vector<vector<double>>& Q, double T, double max_dE = 1000.0) {
    int N = bits.size();

    int bit = randint(0,N-1);
    // double diff = qubo_diff(bits,bit,Q);
    int current_bit = bits[bit];
    bits[bit] = 1 - bits[bit];

    // double dE = (1 - 2 * current_bit) * diff;
    double dE = (1 - 2* current_bit) * (inner_product(Q[bit].begin(), Q[bit].end(), bits.begin(), 0.0));
    dE = max(-max_dE, min(dE, max_dE));
    if (static_cast<double>(randint(1,1e8) / 1e8) >= exp(-dE / T)) {
        bits[bit] = current_bit;
    }
}

pair<vector<int>, double> SimulatedQuantumAnnealing::simulated_quantum_annealing(vector<vector<double>> Q) 
{
    vector<vector<int>> bits(L, vector<int>(N));
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            bits[i][j] = randint(0,1);
        }
    }
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(L)
    for(int layer=0;layer<L;++layer){
        for (int i = 0; i < anneal_steps; ++i){
            for (int j = 0; j < mc_steps; ++j)
            {
                monte_carlo_step(bits[layer], Q, T);
            }
        T *= 0.9;
        }

    }
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    cout << "Execution time: " << duration << " ms" << endl;

    double min_energy = numeric_limits<double>::infinity();
    vector<int> best_bits;
    #pragma omp parallel for num_threads(L)
    for (int layer = 0; layer < L; ++layer) {
        double layer_energy = qubo_energy(bits[layer], Q); 
        if (layer_energy <= min_energy) {
            min_energy = layer_energy;
            best_bits = bits[layer];
        }
    }

    min_energy = qubo_energy(best_bits, Q);
    return {best_bits, min_energy};
}