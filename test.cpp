#include <iostream>
#include "SimulatedQuantumAnnealing.cpp"

using namespace std;
void generate_n_hot_qubo(vector<vector<double>>& Q,int N, int n) {
    for (int i = 0; i < N; ++i) {
        Q[i][i] = 1 - 2 * n;
        for (int j = i + 1; j < N; ++j) {
            Q[i][j] = 0;
            Q[j][i] = 2;
        }
    }
}

int main(){
    cout << "start"<<endl;
    int N = 10; //total bits
    N = N*N;
    int num = 10; //num of selected bits
    SimulatedQuantumAnnealing SQA = SimulatedQuantumAnnealing(N=N);
    auto Q = SQA.init_jij();

    generate_n_hot_qubo(Q,N,num);

    pair<vector<int>, double> result = SQA.simulated_quantum_annealing(Q);
    cout << "Best bits: ";
    for (int bit=0;bit<N;++bit) {
        cout << result.first[bit] << " ";
    }
    cout << endl;

    cout << "Minimum energy: " << result.second + num*num << endl;

    return 0;
}