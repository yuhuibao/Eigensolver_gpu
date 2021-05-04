#!/bin/bash

cd lib_eigsolve
# make clean
make -f mk
cd ../test_hipified
make clean
make
./test_dsygvdx mat1_32.dat mat2_32.dat
#./test_dsygvdx mat1.dat mat2.dat
#./test_dsygvdx 32
cd ..