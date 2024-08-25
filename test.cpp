#include <iostream>
#include "SimulatedQuantumAnnealing.cpp"

using namespace std;

void generate_n_hot_qubo(vector<vector<double>>& Q,int start,int end, int n,vector<pair<vector<int>,int>>& nhot_memo) {
    for (int i = start; i < end; ++i) {
        Q[i][i] = 1 - 2 * n;
        for (int j = i + 1; j < end; ++j) {
            Q[i][j] = 2;
            Q[j][i] = 2;
        }
    }
    vector<int>temp;
    for(int i=start;i<end;++i)temp.push_back(i);
    nhot_memo.push_back(make_pair(temp,n));
}

int main(){
    cout << "start"<<endl;
    int N = 1000; //total bits
    // N = N*N;
    int L = 1;
    int mc_steps = 10;
    int anneal_steps = 10;
    double T = 1;
    int num = 5; //num of selected bits
    SimulatedQuantumAnnealing SQA = SimulatedQuantumAnnealing(N,L,mc_steps,anneal_steps,T);
    auto Q = SQA.init_jij();
    vector<pair<vector<int>,int>>nhot_memo;


    for(int i=0;i<N/10;++i){
        generate_n_hot_qubo(Q,10*i,10*(1+i),num,nhot_memo);   
    }
    pair<vector<int>, double> result = SQA.simulated_quantum_annealing(Q,nhot_memo);
    int one = 0;
    for(int i=0;i<N;i++){
        if(result.first[i]==1)one++;
    }
    cout << "Best Solution: " << endl;
    for(int i=0;i<N;++i){
        if(i%10==0 && i!= 0)cout<<endl;
        cout << result.first[i] << " ";

    }cout << endl;
    //cout << "Minimum energy: " << result.second << endl;
    cout << "number of ones: "<< one << endl;
    return 0;
}