#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include "../../SimulatedQuantumAnnealing.cpp"

void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,vector<int> bits, int n,std::vector<std::pair<std::vector<int>,int>>& nhot_memo, int k) {
    for (int i = 0; i < bits.size(); ++i) {
        int idx1 = bits[i];
        Q[i][i] = k*(1 - 2 * n);
        for (int j = i + 1; j < bits.size(); ++j) {
            int idx2 = bits[j];
            Q[i][j] = k*2;
            Q[j][i] = k*2;
        }
    }
    nhot_memo.push_back(make_pair(bits,n));
}

void TSP(vector<vector<double>>& Q,vector<vector<int>>distance, int n,vector<pair<vector<int>,int>>& nhot_memo){
    double c = 1;
    for(int i=0;i<n;++i){
        for(int k=0;k<n;++k){
            for(int j=0;j<n;++j){
                Q[i*n+j][k*n+j] += distance[i][k];
            }
        }
    }
    for(int i=0;i<n;++i){
        vector<int>bits1(n,-1);
        vector<int>bits2(n,-1);

        for(int j=0;j<n;++j)bits1[j] = i*n+j;
        for(int j=0;j<n;++j)bits2[j] = j*n+i;

        generate_n_hot_qubo(Q,bits1,1,nhot_memo,0);   
        generate_n_hot_qubo(Q,bits2,1,nhot_memo,0);
    }
}

void PreAnnealing(SimulatedQuantumAnnealing& SQA, int n) {
    vector<pair<vector<int>,int>>nhot_memo; 
    auto preQ = SQA.init_jij();

    for(int i=0;i<n;++i){
        vector<int>idx1(n,-1);
        vector<int>idx2(n,-1);

        for(int j=0;j<n;++j)idx1[j] = i*n+j;
        for(int j=0;j<n;++j)idx2[j] = j*n+i;

        generate_n_hot_qubo(preQ,idx1,1,nhot_memo,1);   
        generate_n_hot_qubo(preQ,idx2,1,nhot_memo,1);
    }
    

    SQA.init_default_bit(SQA.create_default_bit(preQ,nhot_memo));
}

int main(){
    int num_reads = 1;
    int mc_steps = 100;
    int anneal_steps = 100;  

    int n = 5; // num of sites
    vector<vector<int>>distance(n,vector<int>(n,0));

    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            distance[i][j] = 1;
        }
    }

    int L = 4; //num of trotter slices
    double T = 1.0; // initialzie templature
    SimulatedQuantumAnnealing SQA = SimulatedQuantumAnnealing(n*n,L,mc_steps,anneal_steps,T);
    auto Q = SQA.init_jij();

    vector<pair<vector<int>,int>>nhot_memo;    

    PreAnnealing(SQA,n);
    TSP(Q,distance,n,nhot_memo);

    vector<pair<vector<int>, double>> result;
    vector<double>duration;

    showProgressBar(0, num_reads,"numreads");
    for(int queue=0;queue<num_reads;++queue){
        auto start = chrono::high_resolution_clock::now();
    
        pair<vector<int>, double> res = SQA.swaq(Q,nhot_memo);
        result.push_back(res);
        showProgressBar(queue+1, num_reads,"numreads");

            
        auto end = chrono::high_resolution_clock::now();
        duration.push_back(chrono::duration_cast<chrono::milliseconds>(end - start).count());
    }cout << endl;


    double min = 1e10;
    int best_queue = -1;
    for(int queue=0;queue<num_reads;++queue){
        if(min>result[queue].second){
            min = result[queue].second;
            best_queue = queue;
        }
    }
  
    bit_to_csv(result[best_queue].first,n,"graphcolored");

    cout << duration[0];
    for(int i = 1;i < num_reads; ++i) cout <<" "<< duration[i] ;
    cout << endl;

    return 0;
}