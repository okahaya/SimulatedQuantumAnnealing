#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include "SimulatedQuantumAnnealing.cpp"

void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,int start,int end, int n,std::vector<std::pair<std::vector<int>,int>>& nhot_memo, double k) {
    for (int i = start; i < end; ++i) {
        Q[i][i] = k*(1 - 2 * n);
        for (int j = i + 1; j < end; ++j) {
            Q[i][j] = k*2;
            Q[j][i] = k*2;
        }
    }
    std::vector<int>temp;
    for(int i=start;i<end;++i)temp.push_back(i);
    nhot_memo.push_back(make_pair(temp,n));
}

void GraphColoring(std::vector<std::vector<double>>& Q, std::pair<int,int>hw,int num_colors, std::vector<std::pair<std::vector<int>,int>>& nhot_memo){
    std::vector<int>colors(num_colors);
    double c = 1;
    for(int i=0;i<num_colors;++i){
        colors[i]=i;
    }
    int h = hw.first;
    int w = hw.second;
    int size = h*w;
    for(int i=0;i<size;++i){
        for(int j=0;j<num_colors;++j){
            if(i==0){
                Q[i*num_colors+j][(i+1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i+w)*num_colors+j]+=c;
            }
            else if(i==w-1){
                Q[i*num_colors+j][(i+w)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-1)*num_colors+j]+=c;
            }
            else if(i==h*w-1){
                Q[i*num_colors+j][(i-1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-w)*num_colors+j]+=c;
            }
            else if(i==h*w-w){
                Q[i*num_colors+j][(i+1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-w)*num_colors+j]+=c;
            }
            else if(i<w-1){
                Q[i*num_colors+j][(i+1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i+w)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-1)*num_colors+j]+=c;
            }
            else if(i>h*w-w){
                Q[i*num_colors+j][(i+1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-w)*num_colors+j]+=c;
            }
            else if(i%w==0){
                Q[i*num_colors+j][(i+1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i+w)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-w)*num_colors+j]+=c;
            }
            else if(i%w==w-1){
                Q[i*num_colors+j][(i+w)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-w)*num_colors+j]+=c;
            }
            else{
                Q[i*num_colors+j][(i+1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i+w)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-1)*num_colors+j]+=c;
                Q[i*num_colors+j][(i-w)*num_colors+j]+=c;
            }
        }

    }
    for(int i=0;i<size;++i){
        generate_n_hot_qubo(Q,i*num_colors,(1+i)*num_colors,1,nhot_memo,0);   
    }
}

void PreAnnealing(SimulatedQuantumAnnealing& SQA, std::pair<int,int>hw, int num_colors) {
    // vector<pair<vector<int>,int>>nhot_memo; 
    // auto preQ = SQA.init_jij();
    int h = hw.first;
    int w = hw.second;
    int size = h*w;
    // for(int i=0;i<size;++i){
    //     generate_n_hot_qubo(preQ,i*num_colors,(1+i)*num_colors,1,nhot_memo,1);   
    // }
    
    // SQA.init_default_bit(SQA.create_default_bit(preQ,nhot_memo));
    vector<int>bit(size*num_colors,0);
    thread_local mt19937 rng(random_device{}());
    uniform_int_distribution<int> dist_col(0,num_colors-1);
    for(int i=0;i<size;++i){
        int col = dist_col(rng);
        bit[i*num_colors+col] = 1;
    }
    SQA.init_default_bit(bit);
}

int evaluate(int h, int w,std::vector<std::vector<int>> result){
    int ene = 0;
    for(int i=0;i<h*w;++i){
        for(int j=0;j<result[0].size();++j){
            if(i>=w)ene+=result[i][j]*result[i-w][j];
            if(i%w!=0)ene+=result[i][j]*result[i-1][j];
            if(i%w!=w-1)ene+=result[i][j]*result[i+1][j];
            if(i<h*w-w)ene+=result[i][j]*result[i+w][j];
        }
    }
    return ene/2;
}




int main(){
    int num_reads = 1;
    int mc_steps = 100;
    int anneal_steps = 100;  


    int h = 20;
    int w = 20;
    int colors = 4; // num of colors


    pair<int,int> hw = {h,w};
    int size = h*w;// num of nodes
    int L = 4; //num of trotter slices
    double T = 1.0; // initialzie templature
    SimulatedQuantumAnnealing SQA = SimulatedQuantumAnnealing(size*colors,L,mc_steps,anneal_steps,T);
    auto Q = SQA.init_jij();

    vector<pair<vector<int>,int>>nhot_memo;    

    PreAnnealing(SQA,hw,colors);
    GraphColoring(Q,hw,colors,nhot_memo);

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
  
    bit_to_csv(result[best_queue].first,colors,"graphcolored");

    cout << duration[0];
    for(int i = 1;i < num_reads; ++i) cout <<" "<< duration[i] ;
    cout << endl;

    return 0;
}