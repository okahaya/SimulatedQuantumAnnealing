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

class VectorSet {
private:
    unordered_map<int, vector<int>> element_to_vectors;
    vector<vector<int>> vectors;

public:
    VectorSet(const vector<vector<int>>& vectors)  : vectors(vectors)
    {
        for (int i = 0; i < vectors.size(); ++i) {
            for (int element : vectors[i]) {
                element_to_vectors[element].push_back(i);
            }
        }
    }

    int find_common_element(int idxA, int idxB) {
        const vector<int>& A = vectors[idxA];
        const vector<int>& B = vectors[idxB];

        for (int element : A) {
            if (element_to_vectors[element].size() > 1) {
                for (int vec_idx : element_to_vectors[element]) {
                    if (vec_idx == idxB) {
                        return element;
                    }
                }
            }
        }
        return -1;
    }

    void cout_all(int idxA, int idxB) {
        const vector<int>& A = vectors[idxA];
        const vector<int>& B = vectors[idxB];
        for(int i=0;i<A.size();++i)cout << A[i] << endl;
        for(int i=0;i<B.size();++i)cout << B[i] << endl;
    }
};


void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, vector<vector<int>>& bit_nhot, VectorSet VectorSet, vector<vector<int>>& ones, double max_dE = 1e6) {
    int N = bits[0].size();
    int L = bits.size();
    // double Bt = -1.0 / 2.0 * log(tanh(Gamma / (L * T)));
    // double At = 1/ (L * T);
    double Bt = 0;
    double At = 1 - T;
    // cout << At <<endl;
    // cout << Bt/At << endl;
    // #pragma omp parallel
    {
        thread_local mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist_real(0.0, 1.0);
        uniform_int_distribution<int> dist_layer(0, L - 1);
        uniform_int_distribution<int> dist_bit_nhot(0, bit_nhot.size() - 1);

        #pragma omp for 
        for (int i = 0; i < L; ++i) {
            int layer = dist_layer(rng);

            uniform_int_distribution<int> dist_ones(0, ones[i].size() - 1); 
            int idx1 = dist_ones(rng);
            int idx2 = dist_ones(rng);
            int bit1 = ones[layer][idx1];
            int bit2 = ones[layer][idx2];
            while (bit1 == bit2) {
                idx2 = dist_ones(rng);
                bit2 = ones[layer][idx2];
            }

            int pi1 = VectorSet.find_common_element(bit_nhot[bit1][0], bit_nhot[bit2][0]);
            int pi2 = VectorSet.find_common_element(bit_nhot[bit1][1], bit_nhot[bit2][1]);
            if (pi1 == -1 || pi2 == -1) {
                pi1 = VectorSet.find_common_element(bit_nhot[bit1][0], bit_nhot[bit2][1]);
                pi2 = VectorSet.find_common_element(bit_nhot[bit1][1], bit_nhot[bit2][0]);
            }

            if (bits[layer][pi1] == 1 || bits[layer][pi2] == 1) continue;

            int before_bit1 = bits[layer][bit1];
            int before_bit2 = bits[layer][bit2];
            int before_pi1 = bits[layer][pi1];
            int before_pi2 = bits[layer][pi2];

            bits[layer][bit1] = before_pi1;
            bits[layer][bit2] = before_pi2;
            bits[layer][pi1] = before_bit1;
            bits[layer][pi2] = before_bit2;

            ones[layer][idx1] = pi1;
            ones[layer][idx2] = pi2;

            double delta_E = 0.0;
            delta_E += calculate_delta_E(bits, Q, layer, bit1, bits[layer][bit1], At, Bt);
            delta_E += calculate_delta_E(bits, Q, layer, bit2, bits[layer][bit2], At, Bt);
            delta_E += calculate_delta_E(bits, Q, layer, pi1, bits[layer][pi1], At, Bt);
            delta_E += calculate_delta_E(bits, Q, layer, pi2, bits[layer][pi2], At, Bt);
            // cout << delta_E << endl;
            delta_E = max(-max_dE, min(delta_E, max_dE));
            if (dist_real(rng) >= exp(-delta_E / T)) {
                bits[layer][bit1] = before_bit1;
                bits[layer][bit2] = before_bit2;
                bits[layer][pi1] = before_pi1;
                bits[layer][pi2] = before_pi2;

                ones[layer][idx1] = bit1;
                ones[layer][idx2] = bit2;
            }
        }
    }
}


void execute_annealing(vector<vector<int>>& bits, const vector<vector<double>>& Q, int L, int N, double T, double Gamma, int anneal_steps, int mc_steps, double& duration, const vector<pair<vector<int>, int>>& nhot_memo, bool bit_initialized) {
    if (bit_initialized == false) {cout << "bits should be initialized" << endl;}

    vector<vector<int>> ones(L);
    for (int l = 0;l < L ; ++l) {
        for (int i = 0;i < N; ++i) {
            if (bits[l][i] == 1) {
                ones[l].push_back(i);
            }
        }
    } 

    vector<vector<int>>nhot1(N);//bit_to_nhot
    for (int i = 0; i < nhot_memo.size(); ++i) {
        for (int j = 0; j < nhot_memo[i].first.size(); ++j) {
            nhot1[nhot_memo[i].first[j]].push_back(i);
        }
    }

    vector<vector<int>>nhot2(nhot_memo.size());//nhot_to_bit
    for (int i = 0; i < nhot_memo.size(); ++i) {
        nhot2[i] = nhot_memo[i].first;
        // cout << "bit_nhot[" << i <<"] : ";
        // for (int j=0;j<nhot_memo[i].first.size();j++)cout << nhot_memo[i].first[j] <<" ";
        // cout <<endl;
    }  

    VectorSet VectorSet(nhot2);
    // cout << VectorSet.find_common_element(0,2) <<endl;
    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(anneal_steps);


    vector<vector<double>>energies(anneal_steps,vector<double>(L,0));
    vector<vector<double>>driver_energies(anneal_steps,vector<double>(L,0));

    showProgressBar(0, anneal_steps,"annealing step");
    // int max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    #pragma omp parallel
    {
    for (int i = 0; i < anneal_steps; ++i) {
        for (int j = 0; j < mc_steps; ++j) {
            
            monte_carlo_step(bits, Q, T, Gamma, nhot1, VectorSet, ones);
        }
        for (int k = 0; k < L; ++ k){
            energies[i][k] = qubo_energy(bits[k], Q);
        }
        for (int k = 0; k < L; ++ k){
            driver_energies[i][k] = driver_energy(bits, k);
        }

        int tid = omp_get_thread_num();
        if(tid == 0){
            showProgressBar(i+1, anneal_steps,"annealing step");
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