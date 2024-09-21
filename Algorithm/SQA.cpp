// #include "functions.cpp"
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


void saq_monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e6) {
    int N = bits[0].size();
    int L = bits.size();
    double Bt = -1.0 / 2.0 * log(tanh(Gamma / (L * T)));
    double At = 1/ (L * T);

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
            delta_E += calculate_delta_E(bits, Q, layer, bit, bits[layer][bit], At, Bt);
        
            delta_E = max(-max_dE, min(delta_E, max_dE));

            if (dist_real(rng) >= exp(-delta_E / T)) {
            bits[layer][bit] = before_bit;
            }
        }
    }
}

void saq_execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,double T, double Gamma,int anneal_steps,int mc_steps,double& duration,vector<pair<vector<int>,int>>nhot_memo){
    
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            bits[i][j] = randint(0,1);
        }
    }

    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(mc_steps);
    vector<vector<double>>energies(anneal_steps,vector<double>(L,0));

    showProgressBar(0, anneal_steps,"annealing step");
    omp_set_num_threads(1);
    #pragma omp parallel
    for (int i = 0; i < anneal_steps; ++i){
        for (int j = 0; j < mc_steps; ++j){
            saq_monte_carlo_step(bits, Q, T, Gamma, nhot_memo);
        }
        for (int k = 0; k < L; ++ k){
            energies[i][k] = qubo_energy(bits[k], Q);
        }

        int tid = omp_get_thread_num();
        if(tid == 0){
            showProgressBar(i+1, anneal_steps,"annealing step");
            T *= coolingrate;
            Gamma *= gamma;
        }
    }
    
    energies = transpose(energies);
    ofstream file1("preannealing_energies.csv");

    for (const auto& row : energies) {
        for (size_t i = 0; i < row.size(); ++i) {
            file1 << row[i];
            if (i < row.size() - 1) {
                file1 << ","; 
            }
        }
        file1 << "\n";  
    }

    file1.close();
}
