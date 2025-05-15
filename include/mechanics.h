
#include <vector>
#include <map>
#include <set>
#include "matrix.h"


void applyVelocityBC(int ndim, vector<int>& boundarySet, vector<long double> v0, vector<vector<long double>>& velocity, vector<vector<long double>>& acce);
void applyDispBC(int ndim, vector<int>& boundarySet, vector<long double>& dispBC, vector<Particle>& localParticles, vector<vector<long double>>& velocity, vector<vector<long double>>& acce);
void applyFixedBC(int ndim, vector<int>& boundarySet, vector<Particle>& localParticles);
void buildLocalNeighborlist(int rank, int ndim, double dx, double horizon, vector<Particle>& localParticles, vector<Particle>& ghostParticles, const unordered_map<int, int>& globalLocalIDmap, const map<int, int>& globalPartitionIDmap, 
                            const unordered_map<int, int>& globalGhostIDmap, vector<vector<Particle*>>& Neighborslist);
//vector<matrix> computeShapeTensors(int ndim, double n1, double n2, double dx, double horizon, const vector<Particle*>& piNeighbors, Particle& pi, Particle& pj);
// New:

struct ShapePair {

  double K00, K01, K11;

  double L00, L01, L11;

};



ShapePair computeShapeTensors(

    int ndim,

    double n1,

    double n2,

    double dx,

    double horizon,

    const std::vector<Particle*>& nbrs,

    const Particle& pi,

    const Particle& pj);


vector<long double> StrainVector(const matrix& strain);
matrix getStiffnessTensor(int ndim, double E, double nv);
matrix computeStressTensor(matrix& shapeRef, matrix& shapeCur, int ndim, matrix& StiffnessTensor, vector<vector<double>>& bondDamage, int piIndex, int pjIndex);
vector<long double> computeForceDensityStates(int ndim, int rank, double n1, double n2, double horizon, matrix& StiffnessTensor, double dx, vector<Particle*>& piNeighbors, Particle& pi, Particle& pj, 
                                              const unordered_map<int, int>& globalLocalIDmap, vector<vector<double>>& bondDamage);
void updateGhostParticlePositions(int ndim, int rank, vector<Particle>& ghostParticles, const vector<Particle>& localParticles, const unordered_map<int, int>& globalLocalIDmap, const unordered_map<int, int>& globalGhostIDmap, MPI_Comm comm);
void updatePositions(int ndim, vector<Particle>& localParticles, long double stepSize, long double massDensity, vector<vector<long double>>& velocity, vector<vector<long double>>& acce, vector<vector<long double>>& netF);
void computeVelocity(int rank, int ndim, double n1, double n2, double horizon, double dx, long double massDensity, matrix& StiffnessTensor, long double stepSize, vector<vector<long double>>& velocity, vector<vector<Particle*>>& Neighborslist,
                     vector<vector<long double>>& acce, vector<vector<long double>>& netF, vector<Particle>& localParticles, vector<Particle>& ghostParticles, const unordered_map<int, int>& globalLocalIDmap, 
                     const map<int, int>& globalPartitionIDmap, const unordered_map<int, int>& globalGhostIDmap, vector<vector<double>>& bondDamage);
double computeDamagevariable(int ndim, vector<long double> strainV, long double strain_eq, long double damageThreshold);
void computeParticleDamage(vector<Particle>& localParticles, vector<vector<double>>& bondDamage);
void computeDamageStatus(int ndim, double n1, double n2, double dx, double horizon, vector<vector<double>>& bondDamageThreshold, vector<Particle>& localParticles, const vector<vector<Particle*>>& Neighborslist, vector<vector<double>>& bondDamage, const unordered_map<int, int>& globalLocalIDmap);
