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
int randint(int low, int high)
{
    uniform_int_distribution<> dist(low, high);
    return dist(engine);
}

void monte_carlo_step_swap(vector<int>& bits, const vector<vector<double>>& Q, double T, vector<pair<vector<int>,int>>nhot_memo, double max_dE = 1000.0) {
    int N = bits.size();
    int bit1 = randint(0,N-1);
    int bit2 = -1;
    //ビットが単一のnhotに含まれ、さらに含まれるbitの名前が連続である場合のみ有効
    int temp;
    for(int i=0;i<nhot_memo.size();++i){
        if(nhot_memo[i].first[0] <= bit1 && bit1 <= nhot_memo[i].first[nhot_memo[i].first.size()-1]){
        temp = i;
        break;
        }
    }
    while (bit2 == -1){
        int bit = randint(nhot_memo[temp].first[0],nhot_memo[temp].first[nhot_memo[temp].first.size()-1]);
        if(bits[bit] != bits[bit1])bit2 = bit;
    }
    int current_bit1 = bits[bit1];
    int current_bit2 = bits[bit2];
    bits[bit1] = 1 - bits[bit1];
    bits[bit2] = 1 - bits[bit2];
    double dE = (1 - 2* current_bit1) * (inner_product(Q[bit1].begin(), Q[bit1].end(), bits.begin(), 0.0));
    dE += (1 - 2* current_bit2) * (inner_product(Q[bit2].begin(), Q[bit2].end(), bits.begin(), 0.0));
    dE = max(-max_dE, min(dE, max_dE));
    if (((double)rand() / RAND_MAX) >= exp(-dE / T)) {
        bits[bit1] = current_bit1;
        bits[bit2] = current_bit2;
    }

}

void execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,int T,int anneal_steps,int mc_steps,double& duration,vector<pair<vector<int>,int>>nhot_memo){
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
    #pragma omp parallel for num_threads(L)
    for(int layer=0;layer<L;++layer){
        for (int i = 0; i < anneal_steps; ++i){
            for (int j = 0; j < mc_steps; ++j)
            {
                monte_carlo_step_swap(bits[layer], Q, T, nhot_memo);
            }
        T *= 0.9;
        }
    }
    auto end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();
}
