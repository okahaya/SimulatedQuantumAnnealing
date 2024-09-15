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


void monte_carlo_step_swap(vector<int>& bits, const vector<vector<double>>& Q, double T, vector<pair<vector<int>,int>>nhot_memo, double max_dE = 1000.0) {
    int N = bits.size();
    int nhotsize = nhot_memo.size();
    vector<int> selected_nhot = nhot_memo[randint(0,nhotsize-1)].first;
    int bit1 = randint(0,selected_nhot.size()-1);
    int bit2 = randint(0,selected_nhot.size()-1);

    int before_bit1 = bits[bit1];
    int before_bit2 = bits[bit2];

    bits[bit1] = 1 - bits[bit1];
    bits[bit2] = 1 - bits[bit2];
    
    double delta_E = 0.0;
    
    delta_E += calculate_delta_E_classical(bits, Q, bit1, bits[bit1]);
    delta_E += calculate_delta_E_classical(bits, Q, bit2, bits[bit2]);
    
    delta_E = max(-max_dE, min(delta_E, max_dE));
    
    // double dE = (1 - 2* before_bit1) * (inner_product(Q[bit1].begin(), Q[bit1].end(), bits.begin(), 0.0));
    // dE += (1 - 2* before_bit2) * (inner_product(Q[bit2].begin(), Q[bit2].end(), bits.begin(), 0.0));
    // dE = max(-max_dE, min(dE, max_dE));
    
    if (rand_real() >= exp(-delta_E / T)) {
        bits[bit1] = before_bit1;
        bits[bit2] = before_bit2;
    }

}



void execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,double T,double Gamma,int anneal_steps,int mc_steps,double& duration,vector<pair<vector<int>,int>>nhot_memo){
    
    bits.assign(L, vector<int>(N, 0));
    
    for (int i = 0; i < L; ++i) {
        for(int j = 0;j<nhot_memo.size();++j){
            vector<int> selected_bits = nhot_memo[j].first;
            int n = nhot_memo[j].second;
            vector<bool> Is_selected(selected_bits.size(),false);
            for(int k = 0;k<n;++k){
                bool loop = true;
                while(loop==true){
                int rand = randint(0,selected_bits.size()-1);
                if(Is_selected[rand] == true)continue;
                bits[i][selected_bits[rand]] = 1;
                Is_selected[rand] = true;
                loop = false;
                }
            }
        }
    }
    auto start = chrono::high_resolution_clock::now();
    #pragma omp parallel for private(T)
    for(int layer=0;layer<L;++layer){
        vector<int>bits_of_this_layer = bits[layer];
        for (int i = 0; i < anneal_steps; ++i){
            for (int j = 0; j < mc_steps; ++j)
            {
                monte_carlo_step_swap(bits_of_this_layer, Q, T, nhot_memo);
            }
            T *= 0.9;
        }
        bits[layer] = bits_of_this_layer;
    }
    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
}
