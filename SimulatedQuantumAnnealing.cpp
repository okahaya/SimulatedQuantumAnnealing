#include "SimulatedQuantumAnnealing.h"
#include "Algorithm.cpp"
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

pair<vector<int>, double> SimulatedQuantumAnnealing::simulated_quantum_annealing(vector<vector<double>> Q) 
{
    vector<vector<int>> bits(L, vector<int>(N));
    double duration = -1;
    execute_annealing(bits,Q,L,N,T,anneal_steps,mc_steps,duration);

    cout << "Execution time: " << duration << " ms" << endl;

    double min_energy = numeric_limits<double>::infinity();
    vector<int> best_bits;
    #pragma omp parallel for num_threads(L)
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