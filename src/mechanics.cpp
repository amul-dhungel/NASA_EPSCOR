#include "SetupParticleSystem.h"
#include "mechanics.h"
#include "matrix.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>


void applyDispBC(int ndim, vector<int>& boundarySet, vector<long double>& dispBC, vector<Particle>& localParticles, vector<vector<long double>>& velocity, vector<vector<long double>>& acce){
    for (int i = 0; i < boundarySet.size(); ++i){
        int pi = boundarySet[i];
        for (int j = 0; j < ndim; ++j){
            localParticles[pi].currentPositions[j] = localParticles[pi].initialPositions[j] + dispBC[j];
            // cout << " dispbC " << dispBC[j] << endl;
            // cout << " local particles pi " << pi << " currentPositions[j] " << localParticles[pi].currentPositions[j] << endl;
            velocity[pi][j] = 0.0;
            acce[pi][j] = 0.0;
        }
    }
}

void applyVelocityBC(int ndim, vector<int>& boundarySet, vector<long double> v0, vector<vector<long double>>& velocity, vector<vector<long double>>& acce){
    for (int i = 0; i < boundarySet.size(); ++i){
        int pi = boundarySet[i];
        for (int j = 0; j < ndim; ++j){
            velocity[pi][j] = v0[j];
            acce[pi][j] = 0.0;
        }
    }
}

void applyFixedBC(int ndim, vector<int>& boundarySet, vector<Particle>& localParticles){

    for (int i = 0; i < boundarySet.size(); ++i){
        int pi = boundarySet[i];
        for (int j = 0; j < ndim; j++){
            localParticles[pi].currentPositions[j] = localParticles[pi].initialPositions[j];
        }
    }
}


void buildLocalNeighborlist(int rank, int ndim, double dx, double horizon, vector<Particle>& localParticles, vector<Particle>& ghostParticles, const unordered_map<int, int>& globalLocalIDmap, const map<int, int>& globalPartitionIDmap, 
                            const unordered_map<int, int>& globalGhostIDmap, vector<vector<Particle*>>& Neighborslist){
        
    //build Neighborslist for local particles;
    for (int i = 0; i < localParticles.size(); ++i){
        vector<Particle*> piNeighbors;
        for (const int nb : localParticles[i].neighbors){

            if (globalPartitionIDmap.at(nb) == rank){
                int localNbId = globalLocalIDmap.at(nb);
                piNeighbors.push_back(&localParticles[localNbId]);
               
            }

            else if(globalGhostIDmap.find(nb) != globalGhostIDmap.end()){
                int ghostID = globalGhostIDmap.at(nb);
                piNeighbors.push_back(&ghostParticles[ghostID]);

            }
            else{
                cout << "Rank " << rank << " Error in buildLocalNeighborlist: Key " << nb << " not found in either  globalLocalIDmap or globalGhostIDmap." << endl;
                cout << " pi = " << localParticles[i].globalID << endl;
            }

        }
        
        Neighborslist.push_back(piNeighbors);    

    }

}


// Change the signature to return just the 6 numbers you need:


ShapePair computeShapeTensors(

    int ndim, double n1, double n2, double dx, double horizon,

    const vector<Particle*>& nbrs,

    const Particle& pi, const Particle& pj)

{

  // 1) build the three entries of K and Kc

  double a=0, b=0, c=0,  // K entries

         ac=0, bc=0, cc=0; // Kc entries

  double length2 = 0;

  for(int i=0; i<ndim; ++i){

    double d = pj.initialPositions[i] - pi.initialPositions[i];

    length2 += d*d;

  }

  double length = sqrt(length2);



  for (auto *nb : nbrs) {

    // compute bonds

    double dx0 = nb->initialPositions[0] - pi.initialPositions[0];

    double dy0 = nb->initialPositions[1] - pi.initialPositions[1];

    double dx1 = nb->currentPositions[0]     - pi.currentPositions[0];

    double dy1 = nb->currentPositions[1]     - pi.currentPositions[1];



    double lenN2 = dx0*dx0 + dy0*dy0;

    double lenN  = sqrt(lenN2);

    double cost  = ((pj.initialPositions[0]-pi.initialPositions[0])*dx0 + 

                    (pj.initialPositions[1]-pi.initialPositions[1])*dy0) 

                   / (length*lenN);

    cost = std::clamp(cost,-1.0,1.0);



    double w = exp(-n1*fabs(length-lenN)/(horizon*dx)) 

             * pow(0.5+0.5*cost, n2) * nb->volume;



    // accumulate K = Σ w * [dx0;dy0]*[dx0 dy0]

    a  += w*dx0*dx0; 

    b  += w*dx0*dy0; 

    c  += w*dy0*dy0;

    // accumulate Kc = Σ w * [dx1;dy1]*[dx0 dy0]

    ac += w*dx1*dx0; 

    bc += w*dx1*dy0; 

    cc += w*dy1*dy0;

  }



  // 2) 2×2 closed‐form inverse of K

  double det = a*c - b*b;

  double i00 =  c/det, i01 = -b/det;

  double i10 = -b/det, i11 =  a/det;



  // 3) build L = Kc * Kinv  (only 3 entries needed)

  double L00 = i00*ac + i01*bc;

  double L01 = i10*ac + i11*bc;  // note Kinv is symmetric

  double L11 = i10*bc + i11*cc;



  return {a,b,c, L00,L01,L11};

}


vector<long double> StrainVector(const matrix& strain){
    vector<long double> strainV;
    if(strain.ndim == 2){
        strainV.resize(3);
        strainV[0] = strain.elements[0][0];
        strainV[1] = strain.elements[1][1];
        strainV[2] = strain.elements[0][1];
    }

    if(strain.ndim == 3){
        strainV.resize(6);
        strainV[0] = strain.elements[0][0];
        strainV[1] = strain.elements[1][1];
        strainV[2] = strain.elements[2][2];
        strainV[3] = strain.elements[1][2];
        strainV[4] = strain.elements[0][2];
        strainV[5] = strain.elements[0][1];
    }

    return strainV;
}

matrix getStiffnessTensor(int ndim, double E, double nv){
    double preFactor = 0.0;
    matrix StiffnessTensor;
    if (ndim == 2){
        //asume plane stress case
        preFactor = E / (1 - pow(nv, 2));
        StiffnessTensor = matrix(3, {1.0, nv, 0.0, nv, 1.0, 0.0, 0.0, 0.0, 1 - nv});
        StiffnessTensor = StiffnessTensor.timeScalar(preFactor);
    }
    else if (ndim == 3){
        preFactor = E / (1 + nv) / (1 - 2 * nv);
        StiffnessTensor = matrix(6, {1.0 - nv, nv, nv, 0.0, 0.0, 0.0,
                                     nv, 1.0 - nv, nv, 0.0, 0.0, 0.0,
                                     nv, nv, 1.0 - nv, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 1 - 2 * nv, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 1 - 2 * nv, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 1 - 2 * nv});
        StiffnessTensor = StiffnessTensor.timeScalar(preFactor);
    }

    return StiffnessTensor;
}


matrix computeStressTensor(matrix& shapeRef, matrix& shapeCur, int ndim, matrix& StiffnessTensor, vector<vector<double>>& bondDamage, int piIndex, int pjIndex){

    matrix stress;
    matrix deformationGradient = shapeCur.timeMatrix(shapeRef.inverse2D());

    matrix Imatrix = matrix(ndim, vector<long double> (ndim*ndim, 0.0));
    for(int i = 0; i < ndim; ++i) {Imatrix.elements[i][i] = 1.0;}

    matrix strain = (deformationGradient.transpose()).matrixAdd(deformationGradient);
    strain = strain.timeScalar(0.5);
    strain = strain.matrixSub(Imatrix);

    vector<long double> strainV = StrainVector(strain);
    vector<long double> stressV = StiffnessTensor.timeVector(strainV);


    double d = bondDamage[piIndex][pjIndex];

    if(ndim == 2){
        stress = matrix(2, {stressV[0], stressV[2],
                            stressV[2], stressV[1]});
    }
    else if(ndim == 3){
        stress = matrix(3, {stressV[0], stressV[5], stressV[4],
                            stressV[5], stressV[1], stressV[3],
                            stressV[4], stressV[3], stressV[2]});
    }
    stress.timeScalar(1.0 - d);

    return stress;
}


vector<long double> computeForceDensityStates(int ndim, int rank, double n1, double n2, double horizon, matrix& StiffnessTensor, double dx, vector<Particle*>& piNeighbors, Particle& pi, Particle& pj, const unordered_map<int, int>& globalLocalIDmap, vector<vector<double>>& bondDamage){

    matrix Tmatrix = matrix(ndim, vector<long double>(ndim * ndim, 0.0));
    Particle pnb;
    int localNbId;
    int index = 0;

    vector<long double> bondIJ(ndim), bondINb(ndim);
    double length2 = 0.0, lengthNb2, numerator;

    for (int i = 0; i < ndim; ++i){
        bondIJ[i] = pj.initialPositions[i] - pi.initialPositions[i];
        length2 += pow(bondIJ[i], 2);
    }

    double length = sqrt(length2), horizonVolume = 0.0;
    
    int piIndex;
    if (pi.partitionID == rank) {
        piIndex = globalLocalIDmap.at(pi.globalID);
    }
    else{
        piIndex = bondDamage.size() - 1;
    }

    int pjIndex = 0;

    for (Particle* nb : piNeighbors){

        lengthNb2 = 0.0;
        numerator = 0.0;
        for (int i = 0; i < ndim; ++i){
            bondINb[i] =  nb->initialPositions[i] - pi.initialPositions[i];
            lengthNb2 += pow(bondINb[i], 2);
            numerator += bondIJ[i] * bondINb[i];
        }

        double lengthNb = sqrt(lengthNb2);

        double cosAngle = numerator / (length * lengthNb);
        if (cosAngle > 1.0) cosAngle = 1.0; else if (cosAngle < -1.0) cosAngle = -1.0;

        double lengthRatio = abs(length - lengthNb) / (horizon * dx);
        double weight = exp(-n1 * lengthRatio) * pow(0.5 + 0.5 * cosAngle, n2);

//        vector<matrix> shapeTensors = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);
	ShapePair sp = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);


       // matrix stress = computeStressTensor(shapeTensors[0], shapeTensors[1], ndim, StiffnessTensor, bondDamage, piIndex, pjIndex);
        // Rebuild only the tiny K and L matrices if computeStressTensor needs full matrices:
	 matrix K(2, { sp.K00, sp.K01, sp.K01, sp.K11 });
	 matrix L(2, { sp.L00, sp.L01, sp.L01, sp.L11 });
	 matrix stress = computeStressTensor( K, L, ndim, StiffnessTensor, bondDamage, piIndex, pjIndex);
	
	 if (pi.partitionID == rank) {pjIndex += 1;}

 //       Tmatrix = Tmatrix.matrixAdd((stress.timeMatrix(shapeTensors[0].inverse2D())).timeScalar(weight * nb->volume));
	// Build the tiny inverse of K from sp.K00,K01,K11:

	double det = sp.K00*sp.K11 - sp.K01*sp.K01;

	double i00 =  sp.K11/det, i01 = -sp.K01/det;

	double i10 = -sp.K01/det, i11 =  sp.K00/det;



	// Now form stress · Kinv without ever constructing a matrix object:

	// stress is 2×2: [s00 s01; s10 s11]

	// Kinv is [i00 i01; i10 i11]

	double F00 = stress.elements[0][0]*i00 + stress.elements[0][1]*i10;

	double F01 = stress.elements[0][0]*i01 + stress.elements[0][1]*i11;

	double F10 = stress.elements[1][0]*i00 + stress.elements[1][1]*i10;

	double F11 = stress.elements[1][0]*i01 + stress.elements[1][1]*i11;



	// Now scale by weight*volume and add into Tmatrix:

	double scale = weight * nb->volume;

	Tmatrix.elements[0][0] += F00 * scale;

	Tmatrix.elements[0][1] += F01 * scale;

	Tmatrix.elements[1][0] += F10 * scale;

	Tmatrix.elements[1][1] += F11 * scale;


        horizonVolume += nb->volume;
    }

    vector<long double> Tvector = (Tmatrix.timeScalar(1.0 / horizonVolume)).timeVector(bondIJ);

    return Tvector;

}


void updateGhostParticlePositions(int ndim, int rank, vector<Particle>& ghostParticles, const vector<Particle>& localParticles, const unordered_map<int, int>& globalLocalIDmap, const unordered_map<int, int>& globalGhostIDmap, MPI_Comm comm){

    set<int> neighborRanks;
    map<int, vector<int>> particlesRequestVec;

    for (const auto& p : ghostParticles){
        neighborRanks.insert(p.partitionID);
        particlesRequestVec[p.partitionID].push_back(p.globalID);

    }

    map<int, vector<int>> particlesSend;
    for (int nbrank : neighborRanks){
        vector<int> sendparticles = particlesRequestVec[nbrank];
        int sendsize  = sendparticles.size(), recvsize;
        // exchange send size
        MPI_Sendrecv(&sendsize, 1, MPI_INT, nbrank, 0,
                     &recvsize, 1, MPI_INT, nbrank, 0, comm, MPI_STATUS_IGNORE);

        //exchange particle IDs
        vector<int> recvparticles(recvsize);
        MPI_Sendrecv(sendparticles.data(), sendsize, MPI_INT, nbrank, 1,
                     recvparticles.data(), recvsize, MPI_INT, nbrank, 1, comm, MPI_STATUS_IGNORE);

        particlesSend[nbrank] = recvparticles;

    }

    map<int, vector<long double>> ghostPositions;
    for (int nbrank : neighborRanks){
        vector<long double> sendParticlesPositions;
        for (const auto& pid : particlesSend[nbrank]){
            Particle particle = localParticles[globalLocalIDmap.at(pid)];
            sendParticlesPositions.push_back(particle.globalID);
            sendParticlesPositions.insert(sendParticlesPositions.end(), particle.currentPositions.begin(), particle.currentPositions.end());
        }


        // exchange send size
        int sendPdataSize = sendParticlesPositions.size(), recvPdataSize;
        MPI_Sendrecv(&sendPdataSize, 1, MPI_INT, nbrank, 2,
                     &recvPdataSize, 1, MPI_INT, nbrank, 2, comm,  MPI_STATUS_IGNORE);

        // exchange particles data
        vector<long double> recvParticlesPositions(recvPdataSize);
        MPI_Sendrecv(sendParticlesPositions.data(), sendPdataSize, MPI_LONG_DOUBLE, nbrank, 3,
                     recvParticlesPositions.data(), recvPdataSize, MPI_LONG_DOUBLE, nbrank, 3, comm, MPI_STATUS_IGNORE);



        int idx = 0;
        while (idx < recvParticlesPositions.size()){
            int k = recvParticlesPositions[idx++];
            vector<long double> ghostPosition(recvParticlesPositions.begin() + idx, recvParticlesPositions.begin() + idx + ndim);
            ghostPositions[k] = ghostPosition;
            idx += ndim;

        }

    }

    for (const auto& [id, position] : ghostPositions){
        int localID = globalGhostIDmap.at(id);
        ghostParticles[localID].currentPositions = position;
    }

}


void updatePositions(int ndim, vector<Particle>& localParticles, long double stepSize, long double massDensity, vector<vector<long double>>& velocity, vector<vector<long double>>& acce, vector<vector<long double>>& netF){
    for (int i = 0; i < localParticles.size(); ++i){
        for(int m = 0; m < ndim; ++m){
            acce[i][m] = netF[i][m] / massDensity;
            localParticles[i].currentPositions[m] += velocity[i][m] * stepSize + 0.5 * acce[i][m] * pow(stepSize, 2);
        }
    }
}


void computeVelocity(int rank, int ndim, double n1, double n2, double horizon, double dx, long double massDensity, matrix& StiffnessTensor, long double stepSize, vector<vector<long double>>& velocity, vector<vector<Particle*>>& Neighborslist,
                     vector<vector<long double>>& acce, vector<vector<long double>>& netF, vector<Particle>& localParticles, vector<Particle>& ghostParticles, 
                     const unordered_map<int, int>& globalLocalIDmap, const map<int, int>& globalPartitionIDmap, const unordered_map<int, int>& globalGhostIDmap, vector<vector<double>>& bondDamage){
    
    vector<long double> forceIJ, forceJI;
 for (int i = 0; i < localParticles.size(); ++i) {
        Particle& pi = localParticles[i];
        // grab precomputed neighbor pointers
        auto& piNeighbors = Neighborslist[i];

        // ---------------------------------------------------------
        // 1) Precompute all shape-tensors (only once per neighbor)
        // ---------------------------------------------------------
        vector<ShapePair> bondShapes;
        bondShapes.reserve(piNeighbors.size());
        for (size_t idx = 0; idx < piNeighbors.size(); ++idx) {
            bondShapes[idx] = computeShapeTensors(
                ndim, n1, n2, dx, horizon,
                piNeighbors, pi, *piNeighbors[idx]
            );
        }
        
        vector<long double> acceNew(ndim, 0.0);
        
        //compute netForce
        vector<long double> netForce(ndim, 0.0);

        // ---------------------------------------------------------
        // 2) Single loop: compute force contributions using precomputed shapes
        // ---------------------------------------------------------
        for (size_t idx = 0; idx < piNeighbors.size(); ++idx) {
            Particle* nb = piNeighbors[idx];
            const ShapePair &sp = bondShapes[idx];
            //build pjNeighbors
            // inline computeForceDensityStates → force density vector f

            // 1) compute small inverse of K:

            double detK = sp.K00*sp.K11 - sp.K01*sp.K01;

            double i00 =  sp.K11/detK, i01 = -sp.K01/detK;

            double i10 = -sp.K01/detK, i11 =  sp.K00/detK;



            // 2) compute stress tensor once (you can still call your helper)

            matrix K(2, {sp.K00, sp.K01, sp.K01, sp.K11});

            matrix Kc(2, {sp.L00*sp.K00 + sp.L01*sp.K01,

                         sp.L00*sp.K01 + sp.L01*sp.K11,

                         sp.L01*sp.K00 + sp.L11*sp.K01,

                         sp.L01*sp.K01 + sp.L11*sp.K11});
            matrix stress = computeStressTensor(K, Kc, ndim, StiffnessTensor, bondDamage, i, idx);



            // 3) multiply stress * Kinv:

            double s00 = stress.elements[0][0], s01 = stress.elements[0][1];

            double s11 = stress.elements[1][1];

            double F00 = s00*i00 + s01*i10;

            double F01 = s00*i01 + s01*i11;

            double F10 = s01*i00 + s11*i10;

            double F11 = s01*i01 + s11*i11;



            // 4) convert to vector along bond direction:

            double bj0 = nb->initialPositions[0] - pi.initialPositions[0];

            double bj1 = nb->initialPositions[1] - pi.initialPositions[1];

            vector<long double> fIJ(2);

            fIJ[0] = (F00 * bj0 + F01 * bj1) * nb->volume;

            fIJ[1] = (F10 * bj0 + F11 * bj1) * nb->volume;



            // accumulate into netForce

           netForce[0] += fIJ[0];
             netForce[1] += fIJ[1];
        }

        netF[i] = netForce;

        // netF[i] = computeNetForce(rank, ndim, horizon, dx, localParticles, ghostParticles,
        //           globalLocalIDmap, globalPartitionIDmap, globalGhostIDmap, pi);

        for(int m = 0; m < ndim; ++m){
            acceNew[m] = netF[i][m] / massDensity;
            velocity[i][m] += 0.5 * (acce[i][m] + acceNew[m]) * stepSize;
        }
    
    }

}


void computeDamageStatus(int ndim, double n1, double n2, double dx, double horizon, vector<vector<double>>& bondDamageThreshold, vector<Particle>& localParticles, const vector<vector<Particle*>>& Neighborslist, vector<vector<double>>& bondDamage, const unordered_map<int, int>& globalLocalIDmap){

    //Mazars Damage model is implemented
    //model parameters
    double alpha_t, alpha_c, d_t = 0.0, d_c = 0.0, d = 0.0;

    //material parameters
    double a_t = 0.87, b_t = 20000, a_c = 0.65, b_c = 2150;

    int piIndex = 0;
    bool damageOccur = false;
    for (auto& pi : localParticles){
        
        int pjIndex = 0;
        
        vector<Particle*> piNeighbors = Neighborslist[piIndex];
        
        double pDamage = 0;
        vector<double> principalStrains(ndim);
        bondDamage[piIndex].resize(piNeighbors.size());

        for (Particle* nb : piNeighbors){
            alpha_t = 0.0, alpha_c = 0.0;
         //   vector<matrix> shapeTensors = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);
          //  matrix deformationGradient = shapeTensors[1].timeMatrix(shapeTensors[0].inverse2D());
       ShapePair sp = computeShapeTensors(ndim, n1, n2, dx, horizon, piNeighbors, pi, *nb);
	matrix K(2, { sp.K00, sp.K01, sp.K01, sp.K11 });
	double det = sp.K00*sp.K11 - sp.K01*sp.K01;
	matrix Kinv(2, { sp.K11/det, -sp.K01/det, -sp.K01/det, sp.K00/det });
	matrix L(2, { sp.L00, sp.L01, sp.L01, sp.L11 });
	matrix deformationGradient = L;  // because L == Kc * Kin

            matrix Imatrix = matrix(ndim, vector<long double> (ndim*ndim, 0.0));
            for(int i = 0; i < ndim; ++i) {Imatrix.elements[i][i] = 1.0;}

            matrix strain = (deformationGradient.transpose()).matrixAdd(deformationGradient);
            strain = strain.timeScalar(0.5);
            strain = strain.matrixSub(Imatrix);

            Eigen::MatrixXd strain_temp(ndim, ndim);
            Eigen::VectorXd strain_eigenvalues(ndim);
            Eigen::MatrixXd strain_eigenvectors(ndim, ndim);

            Eigen::MatrixXd strain_t(ndim, ndim);
            Eigen::VectorXd strain_t_eigenvalues(ndim);
            Eigen::MatrixXd strain_t_eigenvectors(ndim, ndim);

            Eigen::MatrixXd strain_c(ndim, ndim);
            Eigen::VectorXd strain_c_eigenvalues(ndim);
            Eigen::MatrixXd strain_c_eigenvectors(ndim, ndim);

            for(int i = 0; i < ndim; ++i){
                for(int j = 0; j < ndim; ++j){
                    strain_temp(i,j) = strain.elements[i][j]; 
                }
            }
            
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver;
            solver.compute(strain_temp);
            strain_eigenvalues = solver.eigenvalues();
            strain_eigenvectors = solver.eigenvectors();
            for (int i = 0; i < ndim; ++i) {principalStrains[i] = strain_eigenvalues(i);}

            //compute the equivalent strain
            double strain2 = 0.0;
            for (int i = 0; i < ndim; ++i){
                if (principalStrains[i] > 0.0){
                    strain2 += pow(principalStrains[i], 2);
                }
            }
            long double strain_eq = sqrt(strain2);

            //if(pi.globalID == 220) {cout << "strain_eq = " << strain_eq << endl;}
            
            if(strain_eq > bondDamageThreshold[piIndex][pjIndex]){
                damageOccur = true;
                double damageThreshold = bondDamageThreshold[piIndex][pjIndex];
                //compute the strain_t and strain_c
                for (int i = 0; i < ndim; ++i){
                    if (strain_eigenvalues(i) > 0.0){
                        for (int j = 0; j < ndim; ++j){
                            for (int k = 0; k < ndim; ++k){
                                strain_t(j,k) += strain_eigenvalues(i) * strain_eigenvectors(j,i) * strain_eigenvectors(k,i);
                            }
                        }
                    }
                    else if (strain_eigenvalues(i) < 0.0){
                        for (int j = 0; j < ndim; ++j){
                            for (int k = 0; k < ndim; ++k){
                                strain_c(j,k) += strain_eigenvalues(i) * strain_eigenvectors(j,i) * strain_eigenvectors(k,i);
                            }
                        }
                    }
                }

                //compute the eigen values of strain_t and strain_c
                solver.compute(strain_t);
                strain_t_eigenvalues = solver.eigenvalues();

                solver.compute(strain_c);
                strain_c_eigenvalues = solver.eigenvalues();

                //compute alpha_t and alpha_c
                for (int i = 0; i < ndim; ++i){
                    if(strain_t_eigenvalues(i) > 0.0){
                        alpha_t += strain_t_eigenvalues(i) * (strain_t_eigenvalues(i) + strain_c_eigenvalues(i)) / (strain_eq * strain_eq);
                    }
                    if(strain_c_eigenvalues(i) > 0.0){
                        alpha_c += strain_c_eigenvalues(i) * (strain_t_eigenvalues(i) + strain_c_eigenvalues(i)) / (strain_eq * strain_eq);
                    }
                }

                //compute d_t and d_c and d
                d_t = 1.0 - damageThreshold * (1 - a_t) / strain_eq - a_t / exp(b_t * (strain_eq - damageThreshold));
                d_c = 1.0 - damageThreshold * (1 - a_c) / strain_eq - a_c / exp(b_t * (strain_eq - damageThreshold));

                d = alpha_t * d_t + alpha_c * d_c;

                bondDamageThreshold[piIndex][pjIndex] = strain_eq;

            }

            bondDamage[piIndex][pjIndex] = d;
            pDamage += d / pi.neighbors.size();
            pjIndex += 1;
            
        }

        pi.damageStatus = pDamage;
        piIndex++;

    }
    bondDamage[piIndex].resize(30, 0.0); //to avoid the sengmentation caused by the ghost particle.
    if(damageOccur) cout << "***Bond Damage Occurs*** " << endl;
    damageOccur = false;
}
