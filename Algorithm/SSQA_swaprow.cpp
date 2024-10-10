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

std::ofstream log_file("annealing_log.csv");
std::ofstream log_file2("montecarlo_log.csv");

void flip_bits(vector<int>& bits, vector<int> idx1, vector<int> idx2, vector<int>& fliped_bits) {
    fliped_bits.clear();
    for (int i=0; i<idx1.size(); ++i) {
        if (bits[idx1[i]] != bits[idx2[i]]) {
            fliped_bits.push_back(idx1[i]);
            fliped_bits.push_back(idx2[i]);
            bits[idx1[i]] = 1 - bits[idx1[i]]; 
            bits[idx2[i]] = 1 - bits[idx2[i]]; 
        }
    }
}

void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e100) {
    int N = bits[0].size();
    int L = bits.size();
    double Bt = -1.0 / 2.0 * log(tanh(Gamma / (L * T)));
    double At = 1 / (L * T);

    thread_local mt19937 rng(random_device{}());
    uniform_real_distribution<double> dist_real(0.0, 1.0);
    uniform_int_distribution<int> dist_layer(0, L - 1);
    uniform_int_distribution<int> dist_nhot(0, nhot_memo.size() - 1);

    #pragma omp for 
    for (int i = 0; i < L; ++i) {
        int layer = dist_layer(rng);
        int idx1 = dist_nhot(rng);
        int idx2 = dist_nhot(rng);
        while (idx1 % 2 != idx2 % 2) {
            idx2 = dist_nhot(rng);
        }
        
        vector<int> fliped_bits(1,0);
        flip_bits(bits[layer], nhot_memo[idx1].first, nhot_memo[idx2].first, fliped_bits);

        double delta_E = 0.0;
        for(int j = 0; j < fliped_bits.size(); ++j) {
            delta_E += calculate_delta_E(bits, Q, layer, fliped_bits[j], bits[layer][fliped_bits[j]], At, Bt);
        }

        delta_E = max(-max_dE, min(delta_E, max_dE));

        double acceptance_probability = exp(-delta_E / T);
    
        log_file2 << "Layer: " << layer << ", Bt :" << Bt << ", At: " << At <<", Delta_E: " << delta_E << ", Acceptance Probability: " << acceptance_probability << "\n";


        if (dist_real(rng) >= exp(-delta_E / T)) {
            for(int j = 0; j < fliped_bits.size(); ++j) {
                bits[layer][fliped_bits[j]] = 1 - bits[layer][fliped_bits[j]];
            }
        }
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

    log_file << "Step,Temperature,Energy\n";
    log_file2 << "Acceptance Probability\n";
    showProgressBar(0, anneal_steps, "annealing step");
    omp_set_num_threads(1);

    #pragma omp parallel
    {
        for (int i = 0; i < anneal_steps; ++i) {
            for (int j = 0; j < mc_steps; ++j) {
                monte_carlo_step(bits, Q, T, Gamma, nhot_memo);
            }
            for (int k = 0; k < L; ++k) {
                energies[i][k] = qubo_energy(bits[k], Q);
                driver_energies[i][k] = driver_energy(bits, k);
            }

            int tid = omp_get_thread_num();
            if (tid == 0) {
                double avg_energy = accumulate(energies[i].begin(), energies[i].end(), 0.0) / L;
                double avg_driver_energy = accumulate(driver_energies[i].begin(), driver_energies[i].end(), 0.0) / L;
                log_file << i << "," << T << "," << avg_energy << "\n";

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