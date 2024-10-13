#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

using namespace std;

thread_local std::mt19937 rng(std::random_device{}());

double qubo_energy(const vector<int>& bits, const vector<vector<double>>& Q) {
    int N = bits.size();
    double energy = 0.0;
    for (int j = 0; j < N; ++j) {
        if (bits[j] == 0) continue;
        for (int k = 0; k < N; ++k) {
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
    double cool = pow(1e-5, 1.0 /(double(anneal_steps)-1));
    return cool;
}

double init_gamma(int anneal_steps){
    if (anneal_steps <= 1) return 1.0;
    double gamma = pow(1e-10, 1.0 /(double(anneal_steps)-1));
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

double calculate_delta_E(const vector<vector<int>> &bits, const vector<vector<double>>& Q, int layer, int bit_index, int new_bit_value, double At, double Bt) {
    double delta_E = 0.0;
    int N = bits[0].size();
    int L = bits.size();
    const vector<int>& bits_layer = bits[layer];
    int delta_bit = 2*new_bit_value - 1;

    for (int j = 0; j < N; ++j) {
        if (Q[bit_index][j] != 0.0 && bits_layer[j] != 0) {
            delta_E += Q[bit_index][j] * bits_layer[j];
        }
    }
    delta_E *= static_cast<double>(delta_bit);
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


std::ofstream mc_log_file("montecarlo_log.csv");

void select_bits(vector<int>& bits, vector<int> idx1, vector<int> idx2, vector<int>& flip_bits) {
    flip_bits.clear();
    for (int i=0; i<idx1.size(); ++i) {
        if (bits[idx1[i]] != bits[idx2[i]]) {
            flip_bits.push_back(idx1[i]);
            flip_bits.push_back(idx2[i]);
        }
    }
}

void flip_bits(vector<int>& bits, const vector<int>& flip_bits) {
    for (int i=0; i<flip_bits.size(); ++i) {
        bits[flip_bits[i]] = 1 - bits[flip_bits[i]];
    }
}

bool Is_contains(const vector<int>& a, int N) {
    unordered_set<int> set(a.begin(), a.end());
    return set.find(N) != set.end();
}

double calculate_delta_E_rowswap(const vector<vector<int>> &bits, const vector<vector<double>>& Q, int layer, const vector<int>& fliped_bits, double At, double Bt) {
    double delta_E = 0.0;
    int N = bits[0].size();
    int L = bits.size();
    const vector<int>& bits_layer = bits[layer];
    vector<double> delta_bits;
    for (int i = 0; i < fliped_bits.size(); ++i) {
        delta_bits.push_back(static_cast<double>(-2*bits_layer[fliped_bits[i]] + 1));
    }   

    for (int i = 0; i < fliped_bits.size(); i++)
    {
        for (int j = 0; j < fliped_bits.size(); j++)
        {
            if (fliped_bits[i] <= fliped_bits[j])
            {
                delta_E += At*delta_bits[i]*Q[fliped_bits[i]][fliped_bits[j]]/2.0;
            }
            else
            {
                delta_E += At*delta_bits[i]*Q[fliped_bits[j]][fliped_bits[i]]/2.0;
            }
        }
    }
    for (int i = 0; i < fliped_bits.size(); i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (bits_layer[j] == 1 && (Q[fliped_bits[i]][j] != 0.0 || Q[j][fliped_bits[i]] != 0.0))
            {
                if (Is_contains(fliped_bits,j) == false)   
                {
                    delta_E += At*delta_bits[i]*(Q[fliped_bits[i]][j] + Q[j][fliped_bits[i]]);                  
                }
            }
            
        }
        
    }
    
    int next_layer = (layer + 1) % L;
    int prev_layer = (layer - 1 + L) % L;
    for (int i = 0; i < fliped_bits.size(); ++i) {
        delta_E += (Bt / L) * delta_bits[i] * (bits[next_layer][fliped_bits[i]] + bits[prev_layer][fliped_bits[i]]);
    }
    return delta_E;
}

void monte_carlo_step(vector<vector<int>>& bits, const vector<vector<double>>& Q, int layer, double T, double Gamma, const vector<pair<vector<int>, int>>& nhot_memo, double max_dE = 1e100) {
    int N = bits[layer].size();
    int L = bits.size();
    // double At = 1 / (L * T);
    // double Bt = (-1.0 / 2.0 * log(tanh(Gamma / (L * T))));
    double Bt = 10.0*(-1.0 / 2.0 * log(tanh(Gamma / (L * T))))*(L * T);
    if (Bt < 1e-3)Bt = 0;
    double At = 1.0;

    thread_local mt19937 rng(random_device{}());
    uniform_real_distribution<double> dist_real(0.0, 1.0);
    uniform_int_distribution<int> dist_nhot(0, nhot_memo.size() - 1);

    int idx1 = dist_nhot(rng);
    int idx2 = dist_nhot(rng);
    while (idx1 % 2 != idx2 % 2) {
        idx2 = dist_nhot(rng);
    }

    vector<int> fliped_bits(1,0);
    select_bits(bits[layer], nhot_memo[idx1].first, nhot_memo[idx2].first, fliped_bits);

    double delta_E = calculate_delta_E_rowswap(bits, Q, layer, fliped_bits, At, Bt);
    
    delta_E = max(-max_dE, min(delta_E, max_dE));

    double acceptance_probability = exp(-delta_E / T);
    
    bool accept = false;
    if (dist_real(rng) < exp(-delta_E / T) && delta_E != 0) {
        flip_bits(bits[layer],fliped_bits);
        accept = true;
    }
    if (layer == 0) {
    mc_log_file << "Layer: " << layer << ", Bt :" << Bt << ", At: " << At <<", Delta_E: " << delta_E << ", Acceptance Probability: " << acceptance_probability <<", Is Accepted:" << accept << "\n";
    }
}
// データをCSV形式で保存する関数
void saveToCSV(const std::string& filename, const std::vector<int>& data) {
    std::ofstream file;

    // 追記モードでファイルを開く
    file.open(filename, std::ios::app);
    if (file.is_open()) {
        for (size_t i = 0; i < data.size(); ++i) {
            file << data[i];
            if (i < data.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
        file.close();
    } else {
        std::cerr << "ファイルを開けませんでした。" << std::endl;
    }

}
void resetCSV(const std::string& filename) {
    std::ofstream file;

    // 上書きモードでファイルを開く（中身を空にする）
    file.open(filename, std::ios::trunc);
    if (file.is_open()) {
        file.close();
    } else {
        std::cerr << "ファイルを開けませんでした。" << std::endl;
    }
}   

void execute_annealing(vector<vector<int>>& bits, const vector<vector<double>>& Q, int L, int N, double T, double Gamma, int anneal_steps, int mc_steps, double& duration, const vector<pair<vector<int>, int>>& nhot_memo, bool bit_initialized) {
    if (!bit_initialized) {
        cout << "bits should be initialized" << endl;
        return;
    }
    std::string filename = "allroute.csv";
    const double coolingrate = init_coolingrate(anneal_steps);
    const double gamma = init_gamma(anneal_steps);

    vector<double> energies;

    
    mc_log_file << "Acceptance Probability\n";
    showProgressBar(0, anneal_steps, "annealing step");
    omp_set_num_threads(4);
    resetCSV(filename);
    #pragma omp parallel for
    for (int  layer = 0; layer < L; layer++) {
        for (int i = 0; i < anneal_steps; i++) {
            for (int j = 0; j < mc_steps; j++) {
                monte_carlo_step(bits,Q,layer,T,Gamma,nhot_memo);
                if(layer == 0){
                energies.push_back(qubo_energy(bits[layer], Q));
                vector<int>route(51,-1);
                for(int site=0;site<51;++site){
                    for(int jun = 0;jun<51;++jun){
                        if(bits[0][site*51+jun]==1)route[jun]=site;
                    }
                }
                saveToCSV(filename, route);
                }
            }
            

            if (layer == 0) {
                showProgressBar(i + 1, anneal_steps, "annealing step");
                T *= coolingrate;
                Gamma *= gamma;
            }
        }
    }

    ofstream file1("energies.csv");

    for (size_t i = 0; i < energies.size(); ++i) {
        file1 << energies[i];
        if (i < energies.size() - 1) {
            file1 << ","; 
        }
    }
file1.close();

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


void generate_n_hot_qubo(std::vector<std::vector<double>>& Q,vector<int>& bits, int n,std::vector<std::pair<std::vector<int>,int>>& nhot_memo, double k) {
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

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                for (int k = 0; k < n - 1; ++k) {
                    Q[i*n + k][j*n + k + 1] += distance[i][j];
                }
                Q[i*n + (n - 1)][j*n] += distance[i][j];
            }
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
    uniform_real_distribution<double> dist_real(0.0, 10);
    vector<vector<double>>distance(n,vector<double>(n,0.0));
    vector<pair<double,double>>sites;
    for (int i=0;i<n;++i) {
        double x = dist_real(rng);
        double y = dist_real(rng);
        sites.push_back(make_pair(x,y));
    }
    
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            if (i == j)distance[i][j] = 0.0;
            else distance[i][j] = calculate_distance(sites[i],sites[j]);
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

struct City {
    // int id;
    double x;
    double y;
};

std::vector<City> readCSV(const std::string& filename) {
    std::vector<City> cities;
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return cities;
    }

    while (std::getline(file, line)) {  
        std::istringstream ss(line);
        City city;
        int temp;
        ss >> temp >> city.x >> city.y;

        cities.push_back(city);  
    }
    
    file.close(); 
    return cities;
}

vector<vector<double>> generate_sites_from_file(string filename){
    vector<City>cities = readCSV(filename);
    int n = cities.size();
    vector<vector<double>>distance(n,vector<double>(n,0.0));
    vector<pair<double,double>>sites;
    for (int i=0;i<n;++i) {
        sites.push_back(make_pair(cities[i].x,cities[i].y));
    }
    
    for (int i=0;i<n;++i) {
        for (int j=0;j<n;++j) {
            if (i == j)distance[i][j] = 0.0;
            else distance[i][j] = calculate_distance(sites[i],sites[j]);
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

    int n = 51; // num of sites
    vector<vector<double>>distance = generate_sites_from_file("eli51tsp.csv");

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