#!/bin/sh

cd ../lib_eigsolve
make -f mk
cd -
make clean
make
./test_dsygvdx mat1.dat mat2.dat
