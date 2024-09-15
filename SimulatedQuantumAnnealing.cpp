#include "SimulatedQuantumAnnealing.h"
#include "Algorithm/SSQA_ene.cpp"
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


pair<vector<int>, double> SimulatedQuantumAnnealing::simulated_quantum_annealing(vector<vector<double>> Q,vector<pair<vector<int>,int>>nhot_memo) 
{
    vector<vector<int>> bits(L, vector<int>(N,0));
    double duration = -1;
    double Gamma = 1;


    execute_annealing(bits,Q,L,N,T,Gamma,anneal_steps,mc_steps,duration,nhot_memo);

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