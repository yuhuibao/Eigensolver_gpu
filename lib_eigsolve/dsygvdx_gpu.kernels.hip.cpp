// This file was generated by gpufort

#include "hip/hip_complex.h"
#include "hip/hip_runtime.h"
#include "hip/math_functions.h"
#include <cstdio>

namespace {
// make float
float make_float(short int a) { return static_cast<float>(a); }
float make_float(unsigned short int a) { return static_cast<float>(a); }
float make_float(unsigned int a) { return static_cast<float>(a); }
float make_float(int a) { return static_cast<float>(a); }
float make_float(long int a) { return static_cast<float>(a); }
float make_float(unsigned long int a) { return static_cast<float>(a); }
float make_float(long long int a) { return static_cast<float>(a); }
float make_float(unsigned long long int a) { return static_cast<float>(a); }
float make_float(signed char a) { return static_cast<float>(a); }
float make_float(unsigned char a) { return static_cast<float>(a); }
float make_float(float a) { return static_cast<float>(a); }
float make_float(double a) { return static_cast<float>(a); }
float make_float(long double a) { return static_cast<float>(a); }
float make_float(hipFloatComplex &a) { return static_cast<float>(a.x); }
float make_float(hipDoubleComplex &a) { return static_cast<float>(a.x); }
// make double
double make_double(short int a) { return static_cast<double>(a); }
double make_double(unsigned short int a) { return static_cast<double>(a); }
double make_double(unsigned int a) { return static_cast<double>(a); }
double make_double(int a) { return static_cast<double>(a); }
double make_double(long int a) { return static_cast<double>(a); }
double make_double(unsigned long int a) { return static_cast<double>(a); }
double make_double(long long int a) { return static_cast<double>(a); }
double make_double(unsigned long long int a) { return static_cast<double>(a); }
double make_double(signed char a) { return static_cast<double>(a); }
double make_double(unsigned char a) { return static_cast<double>(a); }
double make_double(float a) { return static_cast<double>(a); }
double make_double(double a) { return static_cast<double>(a); }
double make_double(long double a) { return static_cast<double>(a); }
double make_double(hipFloatComplex &a) { return static_cast<double>(a.x); }
double make_double(hipDoubleComplex &a) { return static_cast<double>(a.x); }
// conjugate complex type
hipFloatComplex conj(hipFloatComplex &c) { return hipConjf(c); }
hipDoubleComplex conj(hipDoubleComplex &z) { return hipConj(z); }

// TODO Add the following functions:
// - sign(x,y) = sign(y) * |x| - sign transfer function
// ...
} // namespace
#define divideAndRoundUp(x, y) ((x) / (y) + ((x) % (y) != 0))

// BEGIN krnl_959801_0
/* Fortran original:
      ! kernel do(2) <<<*,*, 0, stream1>>>
      do j = 1, N
         do i = 1, N
            if (i > j) then
               Z(i, j) = A(i, j)
            endif
         end do
      end do

*/
// NOTE: The following information was given in the orignal Cuf kernel pragma:
// - Nested outer-most do-loops that are directly mapped to threads: 2
// - Number of blocks (CUDA): [-1, -1, -1]. ('-1' means not specified)
// - Threads per block (CUDA): [-1, -1, -1]. ('-1' means not specified)
// - Shared Memory: 0
// - Stream: stream1

__global__ void krnl_959801_0(double *z, double *a, int n) {
#undef _idx_a
#undef _idx_z
#define _idx_a(a, b) ((a - 1) + n * (b - 1))
#define _idx_z(a, b) ((a - 1) + n * (b - 1))

  unsigned int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
  if ((j <= n) && (i <= n)) {
    if ((i > j)) {
      z[_idx_z(i, j)] = a[_idx_a(i, j)];
    }
  }
}

extern "C" void launch_krnl_959801_0(dim3 *grid, dim3 *block,
                                     const int sharedMem, hipStream_t stream,
                                     double *z, double *a, int n) {
  hipLaunchKernelGGL((krnl_959801_0), *grid, *block, sharedMem, stream, z, a,
                     n);
}
extern "C" void launch_krnl_959801_0_auto(const int sharedMem,
                                          hipStream_t stream, double *z,
                                          double *a, int n) {
  const unsigned int krnl_959801_0_NX = n;
  const unsigned int krnl_959801_0_NY = n;

  const unsigned int krnl_959801_0_blockX = 16;
  const unsigned int krnl_959801_0_blockY = 16;

  const unsigned int krnl_959801_0_gridX =
      divideAndRoundUp(krnl_959801_0_NX, krnl_959801_0_blockX);
  const unsigned int krnl_959801_0_gridY =
      divideAndRoundUp(krnl_959801_0_NY, krnl_959801_0_blockY);

  dim3 grid(krnl_959801_0_gridX, krnl_959801_0_gridY);
  dim3 block(krnl_959801_0_blockX, krnl_959801_0_blockY);
  hipLaunchKernelGGL((krnl_959801_0), grid, block, sharedMem, stream, z, a, n);
}
// END krnl_959801_0
