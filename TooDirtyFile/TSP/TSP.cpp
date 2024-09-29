#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <unordered_map>

using namespace std;

thread_local std::mt19937 rng(std::random_device{}());

double qubo_energy(const vector<int>& bits, const vector<vector<double>>& Q) {
    int N = bits.size();
    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        if (bits[j] == 0) continue;
        for (int k = j; k < N; ++k) {
            if (bits[k] == 0) continue;
            energy += Q[j][k];
        }
    }
    return energy;
}

double driver_energy(const vector<vector<int>>& bits, const int layer) {
    int N = bits[0].size();
    int L = bits.size();

    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;

    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        if(bits[layer][j] == 0)continue;
        energy += bits[next_layer][j] + bits[prev_layer][j];
    }
    return energy;
}

int randint(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

double rand_real() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}
    
double init_coolingrate(int anneal_steps){
    if (anneal_steps <= 1) return 1.0;
    double cool = pow(0.1, 10.0 /(double(anneal_steps)-1));
    return cool;
}

double init_gamma(int anneal_steps){
    if (anneal_steps <= 1) return 1.0;
    double gamma = pow(0.1, 10.0 /(double(anneal_steps)-1));
    return gamma;
}

void showProgressBar(int current, int total, const std::string& label) {
    if (total != 1){
        int bar_width = 50;
        float progress = (float)current / total;

        std::cout << label << " [";
        int pos = bar_width * progress;
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }
}

double calculate_delta_E(const vector<vector<int>> bits, const vector<vector<double>>& Q, int layer, int bit_index, int new_bit_value, double At, double Bt) {
    double delta_E = 0.0;
    int N = bits[0].size();
    int L = bits.size();
    const vector<int>& bits_layer = bits[layer];
    int delta_bit = 2*new_bit_value - 1;

    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0) {
            delta_E += Q[bit_index][j] * bits_layer[j];
        }
    }
    delta_E *= delta_bit;
    delta_E *= At;

    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;

    delta_E += (Bt / L) * delta_bit * ((2 * bits[next_layer][bit_index] - 1) + (2 * bits[prev_layer][bit_index] - 1));

    return delta_E;
}

double calculate_delta_E_classical(const vector<int>& bits, const vector<vector<double>>& Q, int bit_index, int new_bit_value) {
    double delta_E = 0.0;
    int N = bits.size();
    int old_bit_value = bits[bit_index];
    int delta_bit = new_bit_value - old_bit_value;
    if (delta_bit == 0) return 0.0;

    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0) {
            delta_E += Q[bit_index][j] * bits[j];
        }
    }
    delta_E *= delta_bit;


    return delta_E;
}

vector<vector<double>> transpose(const vector<vector<double>>& matrix) {
    if (matrix.empty()) return {};

    int rows = matrix.size();
    int cols = matrix[0].size();

    vector<vector<double>> transposed(cols, vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,const std::vector<std::pair<std::vector<int>,int>> nhot_memo) {
    int k = 1;
    for (int q = 0; q < nhot_memo.size(); ++q) {
        int n = nhot_memo[q].second;
        int start = nhot_memo[q].first[0];
        int end = nhot_memo[q].first[nhot_memo[q].first.size()];
        for (int i = start; i < end; ++i) {
            Q[i][i] += k*(1 - 2 * n);
            for (int j = i + 1; j < end; ++j) {
                Q[i][j] += k*2;
                Q[j][i] += k*2;
            }
        }
    }
}

std::vector<std::vector<int>> split_into_chunks(const std::vector<int>& arr, int n) {
    std::vector<std::vector<int>> result;
    size_t size = arr.size();
    
    for (size_t i = 0; i < size; i += n) {
        std::vector<int> chunk(arr.begin() + i, arr.begin() + std::min(i + n, size));
        result.push_back(chunk);
    }

    return result;
}

void bit_to_csv(vector<int> result, int colors, string filename) {
    std::vector<std::vector<int>> data = split_into_chunks(result,colors);

    std::ofstream file(filename + ".csv");

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n"; 
    }

    file.close();
    // int ene = evaluate(h,w,data);
    std::cout << "saved as " << filename << ".csv" << std::endl;
    // std::cout << ene << std::endl;

}
void all_bit_to_csv(vector<vector<vector<int>>> result, int colors, string filename) {
    std::vector<std::vector<int>> data;
    for(int i=0;i<result.size();++i)data.push_back(result[i][0]);
    filename = "bit";
    std::ofstream file(filename + ".csv");

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n"; 
    }

    file.close();
    // std::cout << ene << std::endl;

}

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
    vector<vector<vector<int>>>keep_bit;

    showProgressBar(0, anneal_steps,"annealing step");
    // int max_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    #pragma omp parallel
    {
    for (int i = 0; i < anneal_steps; ++i) {
        for (int j = 0; j < mc_steps; ++j) {
            
            monte_carlo_step(bits, Q, T, Gamma, nhot1, VectorSet, ones);
            keep_bit.push_back(bits);

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
        all_bit_to_csv(keep_bit,4,"bit");
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


void saq_monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e6) {
    int N = bits[0].size();
    int L = bits.size();
    double Bt = -1.0 / 2.0 * log(tanh(Gamma / (L * T)));
    double At = 1/ (L * T);

    {
        thread_local mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist_real(0.0, 1.0);
        uniform_int_distribution<int> dist_layer(0, L - 1);
        uniform_int_distribution<int> dist_bit(0, N - 1);
        
        vector<int> local_bits;

        #pragma omp for
        for(int i = 0; i < L; ++i){
            int bit = dist_bit(rng);
            int layer = dist_layer(rng);

            int before_bit = bits[layer][bit];

            bits[layer][bit] = 1 - bits[layer][bit];
            
            double delta_E = 0.0;
            delta_E += calculate_delta_E(bits, Q, layer, bit, bits[layer][bit], At, Bt);
        
            delta_E = max(-max_dE, min(delta_E, max_dE));

            if (dist_real(rng) >= exp(-delta_E / T)) {
            bits[layer][bit] = before_bit;
            }
        }
    }
}

void saq_execute_annealing(vector<vector<int>>& bits,vector<vector<double>> Q,int L,int N,double T, double Gamma,int anneal_steps,int mc_steps,double& duration,vector<pair<vector<int>,int>>nhot_memo){
    
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < N; ++j) {
            bits[i][j] = randint(0,1);
        }
    }

    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(mc_steps);
    vector<vector<double>>energies(anneal_steps,vector<double>(L,0));

    showProgressBar(0, anneal_steps,"annealing step");
    omp_set_num_threads(1);
    #pragma omp parallel
    for (int i = 0; i < anneal_steps; ++i){
        for (int j = 0; j < mc_steps; ++j){
            saq_monte_carlo_step(bits, Q, T, Gamma, nhot_memo);
        }
        for (int k = 0; k < L; ++ k){
            energies[i][k] = qubo_energy(bits[k], Q);
        }

        int tid = omp_get_thread_num();
        if(tid == 0){
            showProgressBar(i+1, anneal_steps,"annealing step");
            T *= coolingrate;
            Gamma *= gamma;
        }
    }
    
    energies = transpose(energies);
    ofstream file1("preannealing_energies.csv");

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
}


class SimulatedQuantumAnnealing
{
private:
  int mc_steps;  // Number of Monte Carlo steps
  int anneal_steps;  // Number of annealing steps
  int L;  // Number of layers
  int N; // Number of bits
  double T;
  vector<int> default_bit;
  bool bit_initialized;

public:
  SimulatedQuantumAnnealing(int N, int L = 10, int mc_steps = 10, int anneal_steps = 10, double T = 1.0): N(N),L(L),mc_steps(mc_steps),anneal_steps(anneal_steps),T(T) {}
  
  vector<vector<double>> init_jij();

  pair<vector<int>, double> swaq(vector<vector<double>> Q,vector<pair<vector<int>,int>>nhot_memo) ;
  
  vector<int> create_default_bit(vector<vector<double>> Q,vector<pair<vector<int>,int>>nhot_memo);

  void bit_initialized_true();

  void bit_initialized_false();

  void init_default_bit(vector<int> bit);
  
};


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


void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,vector<int> bits, int n,std::vector<std::pair<std::vector<int>,int>>& nhot_memo, double k) {
    for (int i = 0; i < bits.size(); ++i) {
        Q[i][i] += k*(1 - 2 * n);
        for (int j = i + 1; j < bits.size(); ++j) {
            Q[i][j] += k*2;
            Q[j][i] += k*2;
        }
    }
    nhot_memo.push_back(make_pair(bits,n));
}

void TSP(vector<vector<double>>& Q,vector<vector<double>>distance, int n,vector<pair<vector<int>,int>>& nhot_memo){
    for(int i=0;i<n;++i){
        vector<int>bits1(n,-1);
        vector<int>bits2(n,-1);

        for(int j=0;j<n;++j)bits1[j] = i*n+j;
        for(int j=0;j<n;++j)bits2[j] = j*n+i;

        generate_n_hot_qubo(Q,bits1,1,nhot_memo,0);   
        generate_n_hot_qubo(Q,bits2,1,nhot_memo,0);
    }
        for(int i=0;i<n;++i){
            for(int k=0;k<n;++k){
                for(int j=0;j<n-1;++j){
                    Q[i*n+j][k*n+j+1] += distance[i][k];
                }
                Q[i*n+(n-1)][k*n] += distance[i][k];
            }
        }
}

void PreAnnealing(SimulatedQuantumAnnealing& SQA, int n) {
    // vector<pair<vector<int>,int>>nhot_memo; 
    // auto preQ = SQA.init_jij();

    // for(int i=0;i<n;++i){
    //     vector<int>idx1(n,-1);
    //     vector<int>idx2(n,-1);

    //     for(int j=0;j<n;++j)idx1[j] = i*n+j;
    //     for(int j=0;j<n;++j)idx2[j] = j*n+i;

    //     generate_n_hot_qubo(preQ,idx1,1,nhot_memo,1);   
    //     generate_n_hot_qubo(preQ,idx2,1,nhot_memo,1);
    // }
    

    // SQA.init_default_bit(SQA.create_default_bit(preQ,nhot_memo));
    vector<int>def(n*n,0);
    for(int i=0;i<n;++i){
        def[i*n+i] = 1;
    }
    SQA.init_default_bit(def);

}

double calculate_distance(pair<double,double>f, pair<double,double>s){
    return pow(pow(f.first - s.first, 2.0) + pow(f.second - s.second, 2.0), 0.5);
}

vector<vector<double>> generate_sites(int n) {
    uniform_real_distribution<double> dist_real(0.0, 10.0);
    vector<vector<double>>distance(n,vector<double>(n,0.0));
    vector<pair<double,double>>sites;
    for (int i=0;i<n;++i) {
        double x = dist_real(rng);
        double y = dist_real(rng);
        sites.push_back(make_pair(x,y));
    }
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            distance[i][j] = calculate_distance(sites[i],sites[j]);
        }
    }
    ofstream file("sites.csv");
    file << "X,Y\n";
    for (const auto& site : sites) {
        file << site.first << "," << site.second << "\n";
    }
    file.close();

    return distance;
}

int main(){
    int num_reads = 1;
    int mc_steps = 100;
    int anneal_steps = 100;  

    int n = 10; // num of sites
    vector<vector<double>>distance = generate_sites(n);
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
  
    bit_to_csv(result[best_queue].first,n,"TSP");

    cout << duration[0];
    for(int i = 1;i < num_reads; ++i) cout <<" "<< duration[i] ;
    cout << endl;

    return 0;
}