all: test_dsygvdx

include ../Makefile.hipfort
LINKOPTS += ../lib_eigsolve/lib_eigsolve.a -lhipblas -L$(ROCM)/hipblas/lib -lrocsolver -L$(ROCM)/rocsolver/lib -llapack -lblas
MODULES_DIR = ../lib_eigsolve

test_dsygvdx: test_dsygvdx.hipified.o wallclock.o
	$(FC) $^ $(LINKOPTS) -o $@

test_dsygvdx.hipified.o: test_dsygvdx.hipified.F90
	$(FC) -c test_dsygvdx.hipified.F90 -I$(MODULES_DIR) -o $@

wallclock.o: wallclock.c
	$(CXX) -c wallclock.c -o $@

clean:
	rm -f *.o *.mod