#include "SimulatedQuantumAnnealing.h"
#include "Algorithm/SSQA_2nhot.cpp"
#include "Algorithm/SQA.cpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <random>
#include <omp.h>

using namespace std;


vector<vector<double>> SimulatedQuantumAnnealing::init_jij()
{
    return vector<vector<double>>(N, vector<double>(N, 0.0));
}

void SimulatedQuantumAnnealing::init_default_bit(vector<int> bit) {
    default_bit = bit;
    bit_initialized_true();
}

vector<int> SimulatedQuantumAnnealing::create_default_bit(vector<vector<double>> Q,vector<pair<vector<int>,int>>nhot_memo) {
    double duration = -1;
    double Gamma = 5.0;
    vector<vector<int>> bits(L, vector<int>(N,0));

    saq_execute_annealing(bits, Q, L, N, T, Gamma, anneal_steps, mc_steps, duration, nhot_memo);

    vector<int> best_bits;
    double min_energy = numeric_limits<double>::infinity();
    #pragma omp parallel for
    for (int layer = 0; layer < L; ++layer) {
        double layer_energy = qubo_energy(bits[layer], Q); 
        if (layer_energy <= min_energy) {
            min_energy = layer_energy;
            best_bits = bits[layer];
        }
    }
    min_energy = qubo_energy(best_bits, Q);
    // for debug
    // bit_to_csv(best_bits,4,"defaultbit");
    return best_bits;
}

void SimulatedQuantumAnnealing::bit_initialized_false() {
    bit_initialized = false;
}

void SimulatedQuantumAnnealing::bit_initialized_true() {
    bit_initialized = true;
}


pair<vector<int>, double> SimulatedQuantumAnnealing::swaq(vector<vector<double>> Q,vector<pair<vector<int>,int>>nhot_memo) 
{
    double duration = -1;
    double Gamma = 5.0;

    vector<vector<int>> bits(L, vector<int>(N,0));

    if (bit_initialized == true) {
        for (int i = 0; i < L; ++i) bits[i] = default_bit;
    } 

    execute_annealing(bits,Q,L,N,T,Gamma,anneal_steps,mc_steps,duration,nhot_memo,bit_initialized);

    // std::cout << "Execution time: " << duration << " ms" << endl;

    double min_energy = numeric_limits<double>::infinity();
    vector<int> best_bits;
    #pragma omp parallel for
    for (int layer = 0; layer < L; ++layer) {
        double layer_energy = qubo_energy(bits[layer], Q); 
        if (layer_energy <= min_energy) {
            min_energy = layer_energy;
            best_bits = bits[layer];
        }
    }
    min_energy = qubo_energy(best_bits, Q);
    return {best_bits, min_energy};
}

