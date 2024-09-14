#include <iostream>
#include <vector>
#include <fstream>
#include "../SimulatedQuantumAnnealing.cpp"

void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,int start,int end, int n,std::vector<std::pair<std::vector<int>,int>>& nhot_memo) {
    // for (int i = start; i < end; ++i) {
    //     Q[i][i] = 1 - 2 * n;
    //     for (int j = i + 1; j < end; ++j) {
    //         Q[i][j] = 2;
    //         Q[j][i] = 2;
    //     }
    // }
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
        generate_n_hot_qubo(Q,i*num_colors,(1+i)*num_colors,1,nhot_memo);   
    }
}


std::vector<std::vector<int>> split_into_chunks(const std::vector<int>& arr, int n) {
    std::vector<std::vector<int>> result;
    size_t size = arr.size();
    
    // n要素ずつ分割して二次元配列に格納
    for (size_t i = 0; i < size; i += n) {
        // 配列の末尾に近づいた場合の処理
        std::vector<int> chunk(arr.begin() + i, arr.begin() + std::min(i + n, size));
        result.push_back(chunk);
    }

    return result;
}


int main(){
    int L = 16; //num of trotter slices
    int mc_steps = 10;
    int anneal_steps = 10;
    double T = 1;


    int h = 8;
    int w = 8;
    int colors = 4; // num of colors


    pair<int,int> hw = {h,w};
    int size = h*w;// num of nodes
    SimulatedQuantumAnnealing SQA = SimulatedQuantumAnnealing(size*colors,L,mc_steps,anneal_steps,T);
    auto Q = SQA.init_jij();
    vector<pair<vector<int>,int>>nhot_memo;
    GraphColoring(Q,hw,colors,nhot_memo);
    pair<vector<int>, double> result = SQA.simulated_quantum_annealing(Q,nhot_memo);


    std::vector<std::vector<int>> data = split_into_chunks(result.first,colors);

    // CSVファイルに書き込むためのファイルストリームを開く
    std::ofstream file("graphcolored.csv");

    // ファイルが正しく開けたかを確認
    if (!file.is_open()) {
        std::cerr << "ファイルを開けませんでした" << std::endl;
        return 1;
    }

    // 2次元配列のデータをCSV形式で書き込む
    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";  // カンマで区切る
            }
        }
        file << "\n";  // 行の終わりに改行を追加
    }

    // ファイルを閉じる
    file.close();

    std::cout << "saved as csv" << std::endl;
    std::cout << result.second << std::endl;
    return 0;
}