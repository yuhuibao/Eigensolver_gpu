APP=fort_gst

MODULES_DIR=../../../gpufort/modules/rocm3.3

CXX=hipcc -g -DHIP_ENABLE_PRINTF
FC=gfortran -g
LDOPTS = -lstdc++ -L/opt/rocm/lib -lhip_hcc

FSOURCES=$(wildcard *.f08)
FOBJS  = $(patsubst %.f08,%.o,$(FSOURCES))
CPPSOURCES=$(wildcard *.cpp)
CPPOBJS = $(patsubst %.cpp,%.o,$(CPPSOURCES))

.DEFAULT_GOAL := all

%.o: %.cpp
	$(CXX) -fno-gpu-rdc -fPIC -c $<

%.o: %.f08 %.F90.hipified
	$(FC) -I$(MODULES_DIR) -c $<

$(APP): $(FOBJS) $(CPPOBJS)
	$(FC) $(FOBJS) $(CPPOBJS) $(LDOPTS) $(MODULES_DIR)/hip_check.o $(MODULES_DIR)/hip_types.o -o $@

all: $(APP)

run: $(APP)
	HIP_TRACE_API=1 ./$(APP)

clean:
	rm -f $(APP) *.o *.mod *~