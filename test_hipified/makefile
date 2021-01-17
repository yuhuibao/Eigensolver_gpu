all: test_dsygvdx

include ../Makefile.hipfort
LINKOPTS += ../lib_eigsolve/lib_eigsolve.a -lhipblas -L$(ROCM)/hipblas/lib  -llapack -lblas -lrocblas -L/opt/rocm/rocblas/lib -lrocsolver -L$(ROCM)/rocsolver/lib -lamdhip64 -lhipfort-amdgcn
MODULES_DIR = ../lib_eigsolve

test_dsygvdx: test_dsygvdx.hipified.o wallclock.o utils.o
	$(FC) $^ $(LINKOPTS) -o $@

test_dsygvdx.hipified.o: test_dsygvdx.hipified.F90 utils.o
	$(FC) -g -c test_dsygvdx.hipified.F90 -I$(MODULES_DIR) -o $@
utils.o: utils.F90
	$(FC) -g -c utils.F90 -o $@

wallclock.o: wallclock.c
	$(CXX) -g -c wallclock.c -o $@

clean:
	rm -f *.o *.mod