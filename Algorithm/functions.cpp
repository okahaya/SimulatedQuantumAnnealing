#include <iostream>
#include <vector>
#include <random>

using namespace std;

random_device seed_gen;
mt19937 engine(seed_gen());

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

int randint(int low, int high)
{
    uniform_int_distribution<> dist(low, high);
    return dist(engine);
}
    
double init_coolingrate(int anneal_steps){
    double cool = pow(10,-10/(double(anneal_steps)-1));
    return cool;
}

double init_gamma(int mc_steps){
    double gamma = pow(10,10/(double(mc_steps)-1));
    return gamma;
}

void showProgressBar(int progress, int total) {
    int barWidth = 50; // 進捗バーの幅

    float progressRatio = (float)progress / total;
    int pos = barWidth * progressRatio;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progressRatio * 100.0) << " %\r";
    std::cout.flush();
}