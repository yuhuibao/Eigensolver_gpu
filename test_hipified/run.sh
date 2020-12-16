#!/bin/sh

cd ../lib_eigsolve
make -f mk
cd -
make clean
make
./test_dsygvdx 5
