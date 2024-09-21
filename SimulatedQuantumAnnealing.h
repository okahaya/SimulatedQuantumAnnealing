#include<vector>
using namespace std;
#ifndef _SimulatedQuantumAnnealing_H_
#define _SimulatedQuantumAnnealing_H_

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

  pair<vector<int>, double> simulated_quantum_annealing(vector<vector<double>> Q,vector<pair<vector<int>,int>>nhot_memo) ;
  
  void init_bool_bit_initialized();

  void init_default_bit(vector<int> &bit);
  
};

#endif // _SimulatedQuantumAnnealing_H_
