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

random_device seed_gen;
mt19937 engine(seed_gen());

int randint(int low, int high)
{
    uniform_int_distribution<> dist(low, high);
    return dist(engine);
}

void monte_carlo_step(vector<int>& bits, const vector<vector<double>>& Q, double T, double max_dE = 1000.0) {
    int N = bits.size();

    int bit = randint(0,N-1);
    int current_bit = bits[bit];
    bits[bit] = 1 - bits[bit];
    double dE = (1 - 2* current_bit) * (inner_product(Q[bit].begin(), Q[bit].end(), bits.begin(), 0.0));
    dE = max(-max_dE, min(dE, max_dE));
    if (static_cast<double>(randint(1,1e8) / 1e8) >= exp(-dE / T)) {
        bits[bit] = current_bit;
    }
}

void execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,int T,int anneal_steps,int mc_steps,double& duration){
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
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
}


void monte_carlo_step_layer_metropolis(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double max_dE = 1000.0) {
    int N = bits[0].size();
    int L = bits.size();
    double dE = 0;
    vector<vector<int>> changed_bits(L,vector(3,0));
    vector<vector<int>> current_bits = bits;
    for(int i=0;i<L;++i){
        int bit = randint(0,N-1);
        int layer = randint(0,L-1);
        int before_bit = bits[layer][bit];
        changed_bits.push_back({layer,bit,before_bit});
        bits[layer][bit] = 1 - bits[layer][bit];
        dE += (1 - 2* before_bit) * (inner_product(Q[bit].begin(), Q[bit].end(), bits[layer].begin(), 0.0));
    }
    for(int i=0;i<L;++i){
        int layer = changed_bits[i][0];
        int next_layer = layer % L;
        int bit = changed_bits[i][1];
        if (bits[layer][bit] == current_bits[layer][bit] && bits[next_layer][bit] != current_bits[next_layer][bit]){
            dE += T/L*Q[bit][bit]*bits[layer][bit]*(1-2*bits[next_layer][bit]);
        }
        if (bits[layer][bit] != current_bits[layer][bit] && bits[next_layer][bit] == current_bits[next_layer][bit]){
            dE += T/L*Q[bit][bit]*bits[next_layer][bit]*(1-2*bits[layer][bit]);
        }

    }
    dE = max(-max_dE, min(dE, max_dE));
    if (static_cast<double>(randint(1,1e8) / 1e8) >= exp(-dE / T)) {
        bits = current_bits;
    }
}

void execute_annealing_layer_metropolis(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,int T,int anneal_steps,int mc_steps,double& duration){
    
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            bits[i][j] = randint(0,1);
        }
    }
    
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < anneal_steps; ++i){
        for (int j = 0; j < mc_steps; ++j){
            monte_carlo_step_layer_metropolis(bits, Q, T);
        }
        T *= 0.9;
    }
    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
}