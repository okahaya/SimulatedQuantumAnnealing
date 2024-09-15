#include "functions.cpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <random>
#include <numeric>
#include <omp.h>
#include <chrono>
using namespace std;


void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e6) {
    int N = bits[0].size();
    int L = bits.size();
    double Bt = T / 2 * log(tanh(Gamma / (L * T)));

    #pragma omp parallel
    {
        thread_local mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist_real(0.0, 1.0);
        uniform_int_distribution<int> dist_layer(0, L - 1);
        uniform_int_distribution<int> dist_bit(0, N - 1);
        
        vector<int> local_bits;

        #pragma omp for
        for(int i = 0; i < L; ++i){
            int bit = dist_bit(rng);
            int layer = dist_layer(rng);

            int before_bit = bits[layer][bit];

            bits[layer][bit] = 1 - bits[layer][bit];
            
            double delta_E = 0.0;
            delta_E += calculate_delta_E(bits, Q, layer, bit, bits[layer][bit], Bt);
        
            delta_E = max(-max_dE, min(delta_E, max_dE));

            if (dist_real(rng) >= exp(-delta_E / T)) {
            bits[layer][bit] = before_bit;
            }
        }
    }
}

void execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,double T, double Gamma,int anneal_steps,int mc_steps,double& duration,vector<pair<vector<int>,int>>nhot_memo){
    
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            bits[i][j] = randint(0,1);
        }
    }

    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(mc_steps);
    vector<int>energies;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < anneal_steps; ++i){
        for (int j = 0; j < mc_steps; ++j){
            monte_carlo_step(bits, Q, T, Gamma, nhot_memo);
            Gamma *= gamma;
        }
        T *= coolingrate;
    }
    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
}
