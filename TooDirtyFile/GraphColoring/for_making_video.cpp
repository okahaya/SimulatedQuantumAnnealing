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
            Q[i][i] = k*(1 - 2 * n);
            for (int j = i + 1; j < end; ++j) {
                Q[i][j] = k*2;
                Q[j][i] = k*2;
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
    filename = "all_bit";
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

void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e6) {
    int N = bits[0].size();
    int L = bits.size();
    double Bt = -1.0 / 2.0 * log(tanh(Gamma / (L * T)));
    double At = 1/ (L * T);
    // cout << At <<endl;
    // cout << Bt/At << endl;
    // #pragma omp parallel
    {
        thread_local mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist_real(0.0, 1.0);
        uniform_int_distribution<int> dist_layer(0, L - 1);
        uniform_int_distribution<int> dist_nhot(0, nhot_memo.size() - 1);

        #pragma omp for 
        for (int i = 0; i < L; ++i) {
            int layer = dist_layer(rng);

            const vector<int>& selected_nhot = nhot_memo[dist_nhot(rng)].first;
            if (selected_nhot.size() < 2) continue; 

            uniform_int_distribution<int> dist_bit(0, selected_nhot.size() - 1);
            int idx1 = dist_bit(rng);
            int idx2 = dist_bit(rng);
            while (idx1 == idx2) {
                idx2 = dist_bit(rng);
            }
            int bit1 = selected_nhot[idx1];
            int bit2 = selected_nhot[idx2];

            int before_bit1 = bits[layer][bit1];
            int before_bit2 = bits[layer][bit2];

            bits[layer][bit1] = before_bit2;
            bits[layer][bit2] = before_bit1;

            double delta_E = 0.0;
            delta_E += calculate_delta_E(bits, Q, layer, bit1, bits[layer][bit1], At, Bt);
            delta_E += calculate_delta_E(bits, Q, layer, bit2, bits[layer][bit2], At, Bt);

            delta_E = max(-max_dE, min(delta_E, max_dE));

            if (dist_real(rng) >= exp(-delta_E / T)) {
                bits[layer][bit1] = before_bit1;
                bits[layer][bit2] = before_bit2;
            }
        }
    }
}


void execute_annealing(vector<vector<int>>& bits, const vector<vector<double>>& Q, int L, int N, double T, double Gamma, int anneal_steps, int mc_steps, double& duration, const vector<pair<vector<int>, int>>& nhot_memo, bool bit_initialized) {
    if (bit_initialized == true) {}
    else {
        bits.assign(L, vector<int>(N, 0));
        #pragma omp parallel for
        for (int i = 0; i < L; ++i) {
            for (const auto& nhot_pair : nhot_memo) {
                const vector<int>& selected_bits = nhot_pair.first;
                int n = nhot_pair.second;
                vector<bool> is_selected(selected_bits.size(), false);
                for (int k = 0; k < n; ++k) {
                    int rand_index;
                    do {
                        rand_index = randint(0, selected_bits.size() - 1);
                    } while (is_selected[rand_index]);
                    bits[i][selected_bits[rand_index]] = 1;
                    is_selected[rand_index] = true;
                }
            }
        }
    }


    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(anneal_steps);


    vector<vector<double>>energies(anneal_steps,vector<double>(L,0));
    vector<vector<double>>driver_energies(anneal_steps,vector<double>(L,0));
    vector<vector<vector<int>>>keep_bit;
    showProgressBar(0, anneal_steps,"annealing step");
    omp_set_num_threads(1);
    #pragma omp parallel
    {
    for (int i = 0; i < anneal_steps; ++i) {
        for (int j = 0; j < mc_steps; ++j) {
            monte_carlo_step(bits, Q, T, Gamma, nhot_memo);
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
            keep_bit.push_back(bits);
        }
    }
    all_bit_to_csv(keep_bit,4,"all_bit");
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
    ofstream file1("energies.csv");

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
    // saq_execute_annealing(bits, Q, L, N, T, Gamma, anneal_steps, mc_steps, duration, nhot_memo);

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


void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,int start,int end, int n,std::vector<std::pair<std::vector<int>,int>>& nhot_memo, int k) {
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
    vector<pair<vector<int>,int>>nhot_memo; 
    auto preQ = SQA.init_jij();
    int h = hw.first;
    int w = hw.second;
    int size = h*w;
    for(int i=0;i<size;++i){
        generate_n_hot_qubo(preQ,i*num_colors,(1+i)*num_colors,1,nhot_memo,1);   
    }
    
    SQA.init_default_bit(SQA.create_default_bit(preQ,nhot_memo));
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
    int mc_steps = 1000;
    int anneal_steps = 1000;  


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

    // PreAnnealing(SQA,hw,colors);
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