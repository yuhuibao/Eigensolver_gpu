#!/bin/bash

cd lib_eigsolve
# make clean
make -f mk
cd ../test_hipified
make clean
make 
./test_dsytrd
cd ..