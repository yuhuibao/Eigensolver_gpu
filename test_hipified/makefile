all: test_dsygvdx

include ../Makefile.hipfort
LINKOPTS += ../lib_eigsolve/lib_eigsolve.a -lhipblas -L$(ROCM)/hipblas/lib  -llapack -lblas -lrocblas -L/opt/rocm/rocblas/lib -lrocsolver -L$(ROCM)/rocsolver/lib -lamdhip64 -lhipfort-amdgcn
MODULES_DIR = ../lib_eigsolve

test_dsygvdx: test_dsygvdx.hipified.o wallclock.o toolbox.o
	$(FC) $^ $(LINKOPTS) -o $@

test_dsytrd: test_dsytrd.hipified.o
	$(FC) $^ $(LINKOPTS) -o $@	

test_dsyevd: test_dsyevd.hipified.o
	$(FC) $^ $(LINKOPTS) -o $@	

test_dsygvdx.hipified.o: test_dsygvdx.hipified.F90 toolbox.o
	$(FC) -g -c test_dsygvdx.hipified.F90 -I$(MODULES_DIR) -o $@

test_dsytrd.hipified.o: test_dsytrd.hipified.F90
	$(FC) -g -c test_dsytrd.hipified.F90 -I$(MODULES_DIR) -o $@

test_dsyevd.hipified.o: test_dsyevd.hipified.F90
	$(FC) -g -c test_dsyevd.hipified.F90 -I$(MODULES_DIR) -o $@

wallclock.o: wallclock.c
	$(CXX) -g -c wallclock.c -o $@

toolbox.o: toolbox.F90
	$(FC) -g -c toolbox.F90 -o $@

clean:
	rm -f *.o *.mod test_dsygvdx