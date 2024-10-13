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
#include <unordered_map>

using namespace std;

std::ofstream mc_log_file("montecarlo_log.csv");

void select_bits(vector<int>& bits, vector<int> idx1, vector<int> idx2, vector<int>& flip_bits) {
    flip_bits.clear();
    for (int i=0; i<idx1.size(); ++i) {
        if (bits[idx1[i]] != bits[idx2[i]]) {
            flip_bits.push_back(idx1[i]);
            flip_bits.push_back(idx2[i]);
        }
    }
}

void flip_bits(vector<int>& bits, const vector<int>& flip_bits) {
    for (int i=0; i<flip_bits.size(); ++i) {
        bits[flip_bits[i]] = 1 - bits[flip_bits[i]];
    }
}

bool Is_contains(const vector<int>& a, int N) {
    unordered_set<int> set(a.begin(), a.end());
    return set.find(N) != set.end();
}

double calculate_delta_E_rowswap(const vector<vector<int>> &bits, const vector<vector<double>>& Q, int layer, const vector<int>& fliped_bits, double At, double Bt) {
    double delta_E = 0.0;
    int N = bits[0].size();
    int L = bits.size();
    const vector<int>& bits_layer = bits[layer];
    vector<double> delta_bits;
    for (int i = 0; i < fliped_bits.size(); ++i) {
        delta_bits.push_back(static_cast<double>(-2*bits_layer[fliped_bits[i]] + 1));
    }   

    for (int i = 0; i < fliped_bits.size(); i++)
    {
        for (int j = 0; j < fliped_bits.size(); j++)
        {
            if (fliped_bits[i] <= fliped_bits[j])
            {
                delta_E += At*delta_bits[i]*Q[fliped_bits[i]][fliped_bits[j]]/2.0;
            }
            else
            {
                delta_E += At*delta_bits[i]*Q[fliped_bits[j]][fliped_bits[i]]/2.0;
            }
        }
    }
    for (int i = 0; i < fliped_bits.size(); i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (bits_layer[j] == 1 && (Q[fliped_bits[i]][j] != 0.0 || Q[j][fliped_bits[i]] != 0.0))
            {
                if (Is_contains(fliped_bits,j) == false)   
                {
                    delta_E += At*delta_bits[i]*(Q[fliped_bits[i]][j] + Q[j][fliped_bits[i]]);                  
                }
            }
            
        }
        
    }
    
    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;
    for (int i = 0; i < fliped_bits.size(); ++i) {
        delta_E += (Bt / L) * delta_bits[i] * (bits[next_layer][fliped_bits[i]] + bits[prev_layer][fliped_bits[i]]);
    }
    return delta_E;
}

void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, int layer, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e100) {
    int N = bits[layer].size();
    int L = bits.size();
    // double At = 1 / (L * T);
    // double Bt = (-1.0 / 2.0 * log(tanh(Gamma / (L * T))));
    double Bt = 10.0*(-1.0 / 2.0 * log(tanh(Gamma / (L * T))))*(L * T);
    if (Bt < 1e-3)Bt = 0;
    double At = 1.0;

    thread_local mt19937 rng(random_device{}());
    uniform_real_distribution<double> dist_real(0.0, 1.0);
    uniform_int_distribution<int> dist_nhot(0, nhot_memo.size() - 1);

    int idx1 = dist_nhot(rng);
    int idx2 = dist_nhot(rng);
    while (idx1 % 2 != idx2 % 2) {
        idx2 = dist_nhot(rng);
    }

    vector<int> fliped_bits(1,0);
    select_bits(bits[layer], nhot_memo[idx1].first, nhot_memo[idx2].first, fliped_bits);

    double delta_E = calculate_delta_E_rowswap(bits, Q, layer, fliped_bits, At, Bt);
    
    delta_E = max(-max_dE, min(delta_E, max_dE));

    double acceptance_probability = exp(-delta_E / T);
    
    bool accept = false;
    if (dist_real(rng) < exp(-delta_E / T) && delta_E != 0) {
        flip_bits(bits[layer],fliped_bits);
        accept = true;
    }
    if (layer == 0) {
    mc_log_file << "Layer: " << layer << ", Bt :" << Bt << ", At: " << At <<", Delta_E: " << delta_E << ", Acceptance Probability: " << acceptance_probability <<", Is Accepted:" << accept << "\n";
    }
}

void execute_annealing(vector<vector<int>>& bits, const vector<vector<double>>& Q, int L, int N, double T, double Gamma, int anneal_steps, int mc_steps, double& duration, const vector<pair<vector<int>, int>>& nhot_memo, bool bit_initialized) {
    if (!bit_initialized) {
        cout << "bits should be initialized" << endl;
        return;
    }

    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(anneal_steps);

    vector<vector<double>> energies(anneal_steps, vector<double>(L, 0));
    vector<vector<double>> driver_energies(anneal_steps, vector<double>(L, 0));

    mc_log_file << "Acceptance Probability\n";
    showProgressBar(0, anneal_steps, "annealing step");
    omp_set_num_threads(4);

    #pragma omp parallel for
    for (int  layer = 0; layer < L; layer++) {
        for (int i = 0; i < anneal_steps; i++) {
            for (int j = 0; j < mc_steps; j++) {
                monte_carlo_step(bits,Q,layer,T,Gamma,nhot_memo);
            }
            energies[i][layer] = qubo_energy(bits[layer], Q);
            driver_energies[i][layer] = driver_energy(bits, layer);

            if (layer == 0) {
                showProgressBar(i + 1, anneal_steps, "annealing step");
                T *= coolingrate;
                Gamma *= gamma;        
            }
        }
    }

    energies = transpose(energies);
    driver_energies = transpose(driver_energies);
    ofstream file1("energies.csv");
    ofstream file2("driver_energies.csv");
    
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

    for (const auto& row : driver_energies) {
        for (size_t i = 0; i < row.size(); ++i) {
            file2 << row[i];
            if (i < row.size() - 1) {
                file2 << ","; 
            }
        }
        file2 << "\n";  
    }
    file2.close();
}