module load cmake

module load eigen-3.3.9-gcc-9.3.0-guszy2x

module load openmpi-4.1.1-gcc-9.3.0-5tetfmb#!/bin/bash
#SBATCH --job-name=peridynamics_test
#SBATCH --account=coa_ext_rmaiti_uksr
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --time=08:00:00
#SBATCH --output=job_output.log
#SBATCH --error=job_error.log

# Load required modules
module load cmake
module load eigen-3.3.9-gcc-9.3.0-guszy2x
module load openmpi-4.1.1-gcc-9.3.0-5tetfmb

# Move to the directory where the executable is
cd /project/ext_rmaiti_uksr/NASA/NASA_EPSCOR/build

# Run your executable with MPI
mpirun -np 64 ./2D_Hassa_test

