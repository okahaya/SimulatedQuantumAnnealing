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
#include <fstream>

using namespace std;

void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e6) {
    int N = bits[0].size();
    int L = bits.size();
    double Bt = -1.0 / 2.0 * log(tanh(Gamma / (L * T)));
    double At = 1/ (L * T);
    // cout << At <<endl;
    // cout << Bt << endl;
    // #pragma omp parallel
    {
        thread_local mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist_real(0.0, 1.0);
        uniform_int_distribution<int> dist_layer(0, L - 1);
        uniform_int_distribution<int> dist_nhot(0, nhot_memo.size() - 1);

        #pragma omp for 
        for (int i = 0; i < L; ++i) {
            int layer = dist_layer(rng);

            const vector<int>& selected_nhot = nhot_memo[dist_nhot(rng)].first;
            if (selected_nhot.size() < 2) continue; 

            uniform_int_distribution<int> dist_bit(0, selected_nhot.size() - 1);
            int idx1 = dist_bit(rng);
            int idx2 = dist_bit(rng);
            while (idx1 == idx2) {
                idx2 = dist_bit(rng);
            }
            int bit1 = selected_nhot[idx1];
            int bit2 = selected_nhot[idx2];

            int before_bit1 = bits[layer][bit1];
            int before_bit2 = bits[layer][bit2];

            bits[layer][bit1] = 1 - bits[layer][bit1];
            bits[layer][bit2] = 1 - bits[layer][bit2];

            double delta_E = 0.0;
            delta_E += calculate_delta_E(bits, Q, layer, bit1, bits[layer][bit1], At, Bt);
            delta_E += calculate_delta_E(bits, Q, layer, bit2, bits[layer][bit2], At, Bt);

            delta_E = max(-max_dE, min(delta_E, max_dE));

            if (dist_real(rng) >= exp(-delta_E / T)) {
                bits[layer][bit1] = before_bit1;
                bits[layer][bit2] = before_bit2;
            }
        }
    }
}


void execute_annealing(vector<vector<int>>& bits, const vector<vector<double>>& Q, int L, int N, double T, double Gamma, int anneal_steps, int mc_steps, double& duration, const vector<pair<vector<int>, int>>& nhot_memo, bool bit_initialized) {
    bits.assign(L, vector<int>(N, 0));
    #pragma omp parallel for
    for (int i = 0; i < L; ++i) {
        for (const auto& nhot_pair : nhot_memo) {
            const vector<int>& selected_bits = nhot_pair.first;
            int n = nhot_pair.second;
            vector<bool> is_selected(selected_bits.size(), false);
            for (int k = 0; k < n; ++k) {
                int rand_index;
                do {
                    rand_index = randint(0, selected_bits.size() - 1);
                } while (is_selected[rand_index]);
                bits[i][selected_bits[rand_index]] = 1;
                is_selected[rand_index] = true;
            }
        }
    }


    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(anneal_steps);


    showProgressBar(0, anneal_steps,"annealing step");
    omp_set_num_threads(1);
    #pragma omp parallel
    {
    for (int i = 0; i < anneal_steps; ++i) {
        for (int j = 0; j < mc_steps; ++j) {
            
            monte_carlo_step(bits, Q, T, Gamma, nhot_memo);
        }
        int tid = omp_get_thread_num();
        if(tid == 0){
            showProgressBar(i+1, anneal_steps,"annealing step");
            T *= coolingrate;
            Gamma *= gamma;
        }
    }
    }

}