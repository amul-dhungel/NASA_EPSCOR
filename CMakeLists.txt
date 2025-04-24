# ---------------------------------------------------------------------
#  User‑installed libraries (GKlib, METIS, ParMETIS)
# ---------------------------------------------------------------------
set(USER_PREFIX   "$ENV{HOME}/local")          # where you ran ‘make install’
set(USER_INC_DIR  "${USER_PREFIX}/include")
set(USER_LIB_DIR  "${USER_PREFIX}/lib")

# Pick the file that really exists in $HOME/local/lib
#   $ ls $HOME/local/lib | grep -i parmetis
#     → libparmetis.a
set(PARMETIS_LIB  "${USER_LIB_DIR}/libparmetis.a")   # or .so / .so.4 if present
set(METIS_LIB     "${USER_LIB_DIR}/libmetis.so")
set(GKLIB_LIB     "${USER_LIB_DIR}/libGKlib.a")      # change to .so if you built shared

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    /usr/include/eigen3
    ${USER_INC_DIR}
)

link_directories(${USER_LIB_DIR})

# ---------------------------------------------------------------------
#  Executable and sources
# ---------------------------------------------------------------------
add_executable(2D_Hassa_test
    src/main.cpp
    src/SetupParticleSystem.cpp
    src/matrix.cpp
    src/mechanics.cpp
)

# ---------------------------------------------------------------------
#  MPI and other libraries
# ---------------------------------------------------------------------
find_package(MPI REQUIRED)                      # make sure this is present
include_directories(${MPI_CXX_INCLUDE_DIRS})    # gives access to <mpi.h>

target_link_libraries(2D_Hassa_test
    ${PARMETIS_LIB}
    ${METIS_LIB}
    ${GKLIB_LIB}
    MPI::MPI_CXX
)
