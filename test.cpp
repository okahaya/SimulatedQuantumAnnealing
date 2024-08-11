#include <iostream>
#include "SimulatedQuantumAnnealing.cpp"

using namespace std;
void generate_n_hot_qubo(vector<vector<double>>& Q,int N, int n) {
    for (int i = 0; i < N; ++i) {
        Q[i][i] = 1 - 2 * n;
        for (int j = i + 1; j < N; ++j) {
            Q[i][j] = 2;
            Q[j][i] = 0;
        }
    }
}

int main(){
    cout << "start"<<endl;
    int N = 10; //total bits
    // N = N*N;
    int L = 10;
    int mc_steps = 100;
    int anneal_steps = 1000;
    double T = 1;
    int num = 5; //num of selected bits
    SimulatedQuantumAnnealing SQA = SimulatedQuantumAnnealing(N,L,mc_steps,anneal_steps,T);
    auto Q = SQA.init_jij();

    generate_n_hot_qubo(Q,N,num);

    pair<vector<int>, double> result = SQA.simulated_quantum_annealing(Q);

    cout << "Best Solution: " << endl;
    for(int i=0;i<N;++i){
        cout << result.first[i] << " ";
    }cout << endl;
    cout << "Minimum energy: " << result.second + num*num << endl;

    return 0;
}