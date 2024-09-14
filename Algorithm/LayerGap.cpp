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


void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, double max_dE = 10000.0) {
    int N = bits[0].size();
    int L = bits.size();
    double dE = 0;
    double Bt = T/2*log(tanh(Gamma/(L*T)));
    vector<vector<int>> changed_bits(L,vector(2,0));
    vector<vector<int>> current_bits = bits;

    for(int i=0;i<L;++i){
        int bit = randint(0,N-1);
        int layer = randint(0,L-1);
        int before_bit = bits[layer][bit];
        changed_bits.push_back({layer,bit});
        bits[layer][bit] = 1 - bits[layer][bit];
        dE += (1 - 2* before_bit) * (inner_product(Q[bit].begin(), Q[bit].end(), bits[layer].begin(), 0.0));
    }
    for(int i=0;i<L;++i){
        int layer = changed_bits[i][0];
        int next_layer = layer % L;
        int bit = changed_bits[i][1];
        if (bits[layer][bit] == current_bits[layer][bit] && bits[next_layer][bit] != current_bits[next_layer][bit]){
            dE += (Bt/L)*bits[layer][bit]*(1-2*bits[next_layer][bit]);
        }
        else if (bits[layer][bit] != current_bits[layer][bit] && bits[next_layer][bit] == current_bits[next_layer][bit]){
            dE += (Bt/L)*bits[next_layer][bit]*(1-2*bits[layer][bit]);
        }
    }
    dE = max(-max_dE, min(dE, max_dE));
    if (((double)rand() / RAND_MAX) >= exp(-dE / T)) {
        bits = current_bits;
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
            monte_carlo_step(bits, Q, T, Gamma);
            Gamma *= gamma;
        }
        T *= coolingrate;
    }
    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
}
