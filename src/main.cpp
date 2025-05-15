#include "SetupParticleSystem.h"   // <-- for Particle, readMeshFile(), findNeighbor(), etc.
#include "mechanics.h"             // <-- for computeDamageStatus(), computeVelocity(), etc.
#include "matrix.h"                // <-- for idx_t, getStiffnessTensor()
#include <mpi.h>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>       // for computeDamageStatus, computeVelocity, etc.
              // for idx_t, getStiffnessTensor



using namespace std;


int main(int argc, char *argv[]){

    MPI_Init(&argc, &argv);
  

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (rank == 0) {
        cerr << "[DEBUG] argv:";
        for (int i = 0; i < argc; ++i) cerr << " " << argv[i];
        cerr << endl;
    }

    // Parse --steps N flag to limit iterations
    long long maxSteps = -1;
    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--steps" && i + 1 < argc) {
            maxSteps = atoll(argv[i+1]);
            if (rank == 0) cerr << "[INFO] Limiting to " << maxSteps << " steps" << endl;
            ++i;
        }
    }

    vector<Particle> globalParticles;
    vector<Particle> ghostParticles;
    vector<Particle> localParticles;

    int numParticlesRow = 30, ndim = 2;
    long double dx = 1.0, horizon = 3.0;
    double n1 = 10.0, n2 = 10.0;
    vector<long double> boxlimit(2*ndim);


    vector<idx_t> vtxdist(size + 1);
    vector<idx_t> xadj;
    vector<idx_t> adjncy;

    map<int, vector<idx_t>> xadjMap;
    map<int, vector<idx_t>> adjncyMap;

    if (rank == 0){
        //globalParticles = computeInitialPositions(numParticlesRow, dx, ndim, boxlimit);
        globalParticles = readMeshFile(boxlimit, dx);
        findNeighbor(globalParticles, dx, horizon, ndim);
        cout << "findNeighbor completed." << endl;
        idx_t numVertices = globalParticles.size();
        defineVtxdist(numVertices, size, vtxdist, comm);
        buildGraph(globalParticles, size, vtxdist, xadjMap, adjncyMap);
        cout << "buildGraph completed." << endl;
        //outputParticles(rank, globalParticles);
        //for(int i = 0; i < boxlimit.size(); ++i){ cout << " boxlimit " << i << " : " <<  boxlimit[i] << endl;}
    }

    //distribute the boxlimit for determining boundary particles at each processor
    MPI_Bcast(boxlimit.data(), 2*ndim, MPI_LONG_DOUBLE, 0, comm);
    
//if(rank == size - 1) for(int i = 0; i < boxlimit.size(); ++i){ cout << " boxlimit " << i << " : " <<  boxlimit[i] << endl;}

    //distribute the vtxdist and send local xadj, adjncy
    MPI_Bcast(vtxdist.data(), size + 1, MPI_INT, 0, comm);
    distributeGraph(xadjMap, adjncyMap, size, rank, xadj, adjncy, comm);

    int localNumVertices = vtxdist[rank+1] - vtxdist[rank];
    idx_t *part = new idx_t[localNumVertices];

    map<int, int> globalPartitionIDmap;
    partitionGraph(rank, vtxdist, xadj, adjncy, size, part, comm);
    updatePartitionID(globalParticles, rank, size, part, localNumVertices, globalPartitionIDmap, comm);

  
    if(rank == 0){
        cout << "updatePartitionID completed." << endl;
    }


    distributeParticles(rank, size, globalParticles, localParticles, comm);
    
    std::cout << "Rank " << rank
          << "  localParticles=" << localParticles.size()
          << "  ghostParticles=" << ghostParticles.size() << std::endl;

    int localNumParticles = localParticles.size();
    if(rank == 0){
        cout << "distributeParticles completed." << endl;
    }

    //define the particle ID map (global->local)
    unordered_map<int, int> globalLocalIDmap;
    defineParticleIDmap(localParticles, globalLocalIDmap);
    //outputGlobalLocalIDmap(rank, localParticles, globalLocalIDmap);


    //remove the globalParticles and release memory
    if (rank == 0) {
        globalParticles.clear(); // Removes all elements
        cout << "Rank 0: globalParticles deleted." << endl;
        cout << "Rank 0: globalParticles size after deletion = " << globalParticles.size() << endl;
    }

    createGhostParticles(localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, rank, comm);
    if (rank == 0) {
        cout << "createGhostParticles completed." << endl;
    }

    unordered_map<int, int> globalGhostIDmap;
    defineParticleIDmap(ghostParticles, globalGhostIDmap);

    //outputParticles(rank, localParticles);
    //outputGhostParticles(rank, ghostParticles);

    //identify boundary particles
    vector<set<int>> boundarySet(2 * ndim);
    defineBoundarySet(ndim, dx, boxlimit, boundarySet, localParticles);


    vector<vector<long double>> netF(localNumParticles, vector<long double>(ndim, 0.0)), velocity(localNumParticles, vector<long double>(ndim, 0.0)), acce(localNumParticles, vector<long double>(ndim, 0.0));
    vector<long double> dispBC(ndim, 0.0);
    vector<int> boundaryLeft(boundarySet[0].begin(), boundarySet[0].end());
    vector<int> boundaryRight(boundarySet[1].begin(), boundarySet[1].end());
    vector<int> boundaryBottom(boundarySet[2].begin(), boundarySet[2].end());
    vector<int> boundaryTop(boundarySet[3].begin(), boundarySet[3].end());
    vector<vector<Particle*>> Neighborslist;
    vector<vector<double>> bondDamage(localNumParticles + 1);
    vector<vector<double>> bondDamageThreshold(localNumParticles);

    vector<long double> vt = {0.0, 5.0};
    vector<long double> vb = {0.0, 0.0};

    //unit system used in the current simulation
    //  Length - mm
    //  Time - s
    //  Mass - Kg
    //  Stress - MPa

 #ifdef MINI_PROFILE

  // ~500 steps total

  long double totalTime = 5e-4;   // 0.0005 s
  long double stepSize  = 1e-6;   // 0.000001 s

#else

  long double totalTime = 1e-2;
  long double stepSize  = 1e-9;

#endif

long long totalSteps  = (long long)(totalTime / stepSize);
long double massDensity  = 2.5e-6;  // restore this!
long double E = 3.5e4, nv = 0.2, tensileStrength = 3.0;
long double damageThreshold = tensileStrength / E;
      
    matrix StiffnessTensor = getStiffnessTensor(ndim, E, nv);

    //cout << " totalSteps " << totalSteps << endl;

    buildLocalNeighborlist(rank, ndim, dx, horizon, localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, Neighborslist);
    
    
    MPI_Barrier(comm);
    //outputBuildLocalNeighborList(rank, Neighborslist);

    //initialize bondDamageThreshold
    for (int piIndex= 0; piIndex < localNumParticles; ++piIndex){
        bondDamageThreshold[piIndex].resize(Neighborslist[piIndex].size(), damageThreshold);
    }

    for (int j = 0; j < totalSteps; ++j){
        if (maxSteps >= 0 && j >= maxSteps) {

            if (rank == 0) {

                std::cerr << "[INFO] Reached forced stop at step " << j << "\n";

            }

            break;
        }
        if(rank==0 && j % 100 == 0){
            cout << "--------time step---------" << j << "--------"<< endl;
        }
        if (Neighborslist.empty()) {
            std::cerr << "Rank " << rank << "  Neighborslist is empty\n";
            MPI_Abort(comm, 1); // abort the whole job             // nothing to do on this rank
        }
        
        computeDamageStatus(ndim, n1, n2, dx, horizon, bondDamageThreshold, localParticles, Neighborslist, bondDamage, globalLocalIDmap);

        if (boundaryTop.size() > 0) {applyVelocityBC(ndim, boundaryTop, vt, velocity, acce);}
        //if (boundaryBottom.size() > 0) {applyVelocityBC(ndim, boundaryBottom, vb, velocity, acce);}

        updatePositions(ndim, localParticles, stepSize, massDensity, velocity, acce, netF);
        applyFixedBC(ndim, boundaryBottom, localParticles);

        //printf("Process %d: Iteration %d before barrier\n", rank, j);
        MPI_Barrier(comm);

        updateGhostParticlePositions(ndim, rank, ghostParticles, localParticles, globalLocalIDmap, globalGhostIDmap, comm);
        
        MPI_Barrier(comm);

        //long double start_time = MPI_Wtime();
        computeVelocity(rank, ndim, n1, n2, horizon, dx, massDensity, StiffnessTensor, stepSize, velocity, Neighborslist, acce, netF, localParticles, ghostParticles, globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, bondDamage);
        //long double end_time = MPI_Wtime();
        //long double elapsed_time = end_time - start_time;
        //cout << "Process " << rank << " execution time: " << elapsed_time << " seconds" << endl;

        MPI_Barrier(comm);
        if (j % 5000 == 0){
            //outputLocalParticlesPositions(rank, localParticles, ndim, j);
            outputGatheredPositions(rank, size, ndim, j, localParticles, comm);
        }

    }

    MPI_Finalize();
    return 0;

}
