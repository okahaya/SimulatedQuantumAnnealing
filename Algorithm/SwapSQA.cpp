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

    
void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, vector<pair<vector<int>,int>>nhot_memo, double max_dE = 1000000.0) {
    int N = bits[0].size();
    int L = bits.size();
    double dE = 0;
    double Bt = T/2*log(tanh(Gamma/(L*T)));

    vector<vector<int>> swapped_bits(L,vector(3,0));
    const vector<vector<int>> current_bits = bits;

    #pragma omp parallel for
    for(int i=0;i<L;++i){
        int layer = randint(0,L-1);

        int nhotsize = nhot_memo.size();
        vector<int> selected_nhot = nhot_memo[randint(0,nhotsize-1)].first;
        int bit1 = randint(0,selected_nhot.size()-1);
        int bit2 = randint(0,selected_nhot.size()-1);
        //ビットが単一のnhotに含まれる場合のみ有効

        int before_bit1 = bits[layer][bit1];
        int before_bit2 = bits[layer][bit2];
        swapped_bits[i] = {layer,bit1,bit2};
        bits[layer][bit1] = 1 - bits[layer][bit1];
        bits[layer][bit2] = 1 - bits[layer][bit2];
        dE += (1 - 2* before_bit1) * (inner_product(Q[bit1].begin(), Q[bit1].end(), bits[layer].begin(), 0.0));
        dE += (1 - 2* before_bit2) * (inner_product(Q[bit2].begin(), Q[bit2].end(), bits[layer].begin(), 0.0));
    }
    for(int i=0;i<L;++i){
        int layer = swapped_bits[i][0];
        int next_layer = (layer+1) % L;
        int bit1 = swapped_bits[i][1];
        int bit2 = swapped_bits[i][2];
        dE += (Bt/L)*(2*bits[layer][bit1]-1)*(2*bits[next_layer][bit1]-1);
        dE += (Bt/L)*(2*bits[layer][bit2]-1)*(2*bits[next_layer][bit2]-1);
    }
    dE = max(-max_dE, min(dE, max_dE));

    if (((double)rand() / RAND_MAX) >= exp(-dE / T)) {
        bits = current_bits;
    }

}

void execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,double T,double Gamma,int anneal_steps,int mc_steps,double &duration,vector<pair<vector<int>,int>>nhot_memo){
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
