#!/bin/sh

# This is a place that would generally contain ParElag and the necessary
# libraries.
WORKSPACE=~/parelag_workspace

# This is the path to ParElag; i.e. the directory that has src/, examples/,
# testsuite/, cmake/, Documentation/, etc.
PARELAG_BASE_DIR=$WORKSPACE/parelag

# THIS SHOULD *NOT* BE ${PARELAG_BASE_DIR}!
PARELAG_BUILD_DIR=$PARELAG_BASE_DIR/build

EXTRA_ARGS=$@

# This warns you if your build directory already exists; it does NOT
# stop execution of this script, which will proceed on its merry way
# overwriting any files it needs to.
if [ -e $PARELAG_BUILD_DIR ]; then 
    echo Warning: $PARELAG_BUILD_DIR already exists.
fi

mkdir -p $PARELAG_BUILD_DIR
cd $PARELAG_BUILD_DIR

# Force a complete reconfigure; if this is not the desired behavior,
# comment out these two lines.
rm -f CMakeCache.txt
rm -rf CMakeFiles

cmake \
    -DCMAKE_BUILD_TYPE=OPTIMIZED \
    -DSuiteSparse_DIR=$WORKSPACE/SuiteSparse \
    -DMETIS_DIR=$WORKSPACE/metis-install \
    -DHYPRE_DIR=$WORKSPACE/hypre-install \
    -DMFEM_DIR=$WORKSPACE/mfem \
    -DElagMeshes_DIR=$WORKSPACE/meshes \
    -DParELAG_ENABLE_TESTING=OFF \
    -DParELAG_USE_MARS=OFF \
    -DMARS_DIR=$WORKSPACE/mars/install \
    ${EXTRA_ARGS} \
    ${PARELAG_BASE_DIR}
