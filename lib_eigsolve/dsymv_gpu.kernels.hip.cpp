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

// BEGIN dsymv_gpu
/* Fortran original:
    use cudafor
    implicit none

    integer, value                                    :: N, lda
    real(8), dimension(lda, N), device, intent(in)    :: A
    real(8), dimension(N), device, intent(in)         :: x
    real(8), dimension(N), device                     :: y

    real(8), dimension(BX + 1, BX), shared              :: Ar_s
    real(8), dimension(BX), shared                    :: r_s

    integer                                           :: tx, ty, ii, jj, i, j, k, istat
    real(8)                                           :: rv1, rv2, mysum
    real(8)                                           :: Ar, xl

    ! ii,jj is index of top left corner of block
    ii = (blockIdx%y - 1)*blockDim%x + 1

    mysum = 0.0_8

    tx = threadIdx%x
    ty = threadIdx%y

    if (ii + (blockIdx%x - 1)*blockDim%x > N) return

    i = ii + tx - 1
    if (i <= N) then
       xl = x(i) ! read part of x for lower triangular multiply
    endif

    ! Loop over columns (skip all lower triangular blocks)
    do jj = ii + (blockIdx%x - 1)*blockDim%x, N, gridDim%x*blockDim%x
       j = jj + ty - 1

       ! Load block into shared memory
       ! CASE 1: Diagonal block
       if (ii == jj) then

          ! Load full block into shared memory
          do k = 0, NTILES - 1
             if (i <= N .and. j + k*blockDim%y <= N) then
                Ar_s(tx, ty + k*blockDim%y) = A(i, j + k*blockDim%y)
             endif
          end do

          call syncthreads()

          ! Reflect to populate lower triangular part with true values of A
          do k = 0, NTILES - 1
             if (tx > ty + k*blockDim%y) then
                Ar_s(tx, ty + k*blockDim%y) = Ar_s(ty + k*blockDim%y, tx)
             endif
          end do

          call syncthreads()

          do k = 0, NTILES - 1
             if (i <= N .and. j + k*blockDim%y <= N) then
                mysum = mysum + Ar_s(tx, ty + k*blockDim%y)*x(j + k*blockDim%y)
             endif
          end do

          !call syncthreads()

          ! CASE 2: Upper triangular block
       else if (ii < jj) then
          do k = 0, NTILES - 1
             if (j + k*blockDim%y <= N) then
                Ar = A(i, j + k*blockDim%y)
             endif

             if (i <= N .and. j + k*blockDim%y <= N) then
                mysum = mysum + Ar*x(j + k*blockDim%y)
             endif

             ! Perform product for symmetric lower block here
             if (i <= N .and. j + k*blockDim%y <= N) then
                rv1 = Ar*xl
             else
                rv1 = 0.0_8
             endif

             !Partial sum within warps using shuffle
             rv2 = __shfl_down(rv1, 1)
             rv1 = rv1 + rv2
             rv2 = __shfl_down(rv1, 2)
             rv1 = rv1 + rv2
             rv2 = __shfl_down(rv1, 4)
             rv1 = rv1 + rv2
             rv2 = __shfl_down(rv1, 8)
             rv1 = rv1 + rv2
             rv2 = __shfl_down(rv1, 16)
             rv1 = rv1 + rv2

             if (tx == 1) then
                r_s(ty + k*blockDim%y) = rv1
             endif
          enddo

          call syncthreads()

          if (ty == 1 .and. jj + tx - 1 <= N) then
             istat = atomicadd(y(jj + tx - 1), r_s(tx))
          endif
          !call syncthreads()

       endif

       call syncthreads()

    end do

    if (i <= N) then
       istat = atomicadd(y(i), mysum)
    endif


*/

__global__ void dsymv_gpu(int n,
                          double *a,
                          const int a_n1,
                          const int a_lb1,
                          const int a_lb2,
                          double *x,
                          const int x_n1,
                          const int x_lb1,
                          double *y,
                          const int y_n1,
                          const int y_lb1) {

int ar_s_n1, ar_s_lb1, ar_s_n2, ar_s_lb2, r_s_n1, r_s_lb1;
#undef _idx_a
#define _idx_a(a, b) ((a - (a_lb1)) + a_n1 * (b - (a_lb2)))
#undef _idx_x
#define _idx_x(a) ((a - (x_lb1)))
#undef _idx_y
#define _idx_y(a) ((a - (y_lb1)))
#undef _idx_ar_s
#define _idx_ar_s(a, b) ((a - (ar_s_lb1)) + ar_s_n1 * (b - (ar_s_lb2)))
#undef _idx_r_s
#define _idx_r_s(a) ((a - (r_s_lb1))) 
#define BX 32
#define BY 8
#define NTILES 4
  // ! TODO could not parse:      real(8), dimension(bx + 1, bx), shared              :: ar_s
  // ! TODO could not parse:      real(8), dimension(bx), shared                    :: r_s
  __shared__ double r_s[BX];
  __shared__ double* ar_s;
  ar_s_n1 = BX + 1;
  ar_s_n2 = BX;
  ar_s_lb1 = 1;
  ar_s_lb2 = 1;
  r_s_n1 = BX;
  r_s_lb1 = 1;
  int tx;
  int ty;
  int ii;
  int jj;
  int i;
  int j;
  int k;
  int istat;
  double rv1;
  double rv2;
  double mysum;
  double ar;
  double xl;
  // ! ii,jj is index of top left corner of block
  ii = ((blockIdx.y) * blockDim.x + 1);
  mysum = 0.0 /*_8*/;
  tx = threadIdx.x + 1;
  ty = threadIdx.y + 1;
  if (((ii + (blockIdx.x) * blockDim.x) > n)) {
    return;
  }
  i = (ii + tx - 1);
  if ((i <= n)) {
    xl = x[_idx_x(i)];
    // ! read part of x for lower triangular multiply
  }
  // ! Loop over columns (skip all lower triangular blocks)
  for (int jj = (ii + (blockIdx.x - 1) * blockDim.x); jj <= n; jj += (gridDim.x * blockDim.x)) {
    j = (jj + ty - 1);
    // ! Load block into shared memory
    // ! CASE 1: Diagonal block
    if (ii == jj) {
      // ! Load full block into shared memory
      for (int k = 0; k <= (NTILES - 1); k += 1) {
        if ((i <= n && (j + k * blockDim.y) <= n)) {
          ar_s[_idx_ar_s(tx, (ty + k * blockDim.y))] = a[_idx_a(i, (j + k * blockDim.y))];
        }
      }
      __syncthreads(); // ! Reflect to populate lower triangular part with true values of A
          for (int k = 0; k <= (NTILES - 1); k += 1) {
        if ((tx > (ty + k * blockDim.y))) {
          ar_s[_idx_ar_s(tx, (ty + k * blockDim.y))] = ar_s[_idx_ar_s((ty + k * blockDim.y), tx)];
        }
      }
      __syncthreads();
       for (int k = 0; k <= (NTILES - 1); k += 1) {
        if ((i <= n && (j + k * blockDim.y) <= n)) {
          mysum = (mysum + ar_s[_idx_ar_s(tx, (ty + k * blockDim.y))] * x[_idx_x((j + k * blockDim.y))]);
        }

      } // !call __syncthreads()
      // ! CASE 2: Upper triangular block

    } else if ((ii < jj)) {
      for (int k = 0; k <= (NTILES - 1); k += 1) {
        if (((j + k * blockDim.y) <= n)) {
          ar = a[_idx_a(i, (j + k * blockDim.y))];
        }
        if ((i <= n && (j + k * blockDim.y) <= n)) {
          mysum = (mysum + ar * x[_idx_x((j + k * blockDim.y))]);
        }
        // ! Perform product for symmetric lower block here
        if ((i <= n && (j + k * blockDim.y) <= n)) {
          rv1 = (ar * xl);

        } else {
          rv1 = 0.0 /*_8*/;
        }
        // !Partial sum within warps using shuffle
        rv2 = __shfl_down(rv1, 1);
        rv1 = (rv1 + rv2);
        rv2 = __shfl_down(rv1, 2);
        rv1 = (rv1 + rv2);
        rv2 = __shfl_down(rv1, 4);
        rv1 = (rv1 + rv2);
        rv2 = __shfl_down(rv1, 8);
        rv1 = (rv1 + rv2);
        rv2 = __shfl_down(rv1, 16);
        rv1 = (rv1 + rv2);
        if (tx == 1) {
          r_s[_idx_r_s((ty + k * blockDim.y))] = rv1;
        }
      }
      __syncthreads(); 
      if ((ty == 1 && (jj + tx - 1) <= n)) { 
          istat = atomicAdd(y + _idx_y((jj + tx - 1)*8), r_s[_idx_r_s(tx)]); 
      }
      // !call __syncthreads()
    }
    __syncthreads();
  }
  if ((i <= n)) {
    istat = atomicAdd(y + _idx_y(i)*8, mysum);
  }
}

extern "C" void launch_dsymv_gpu(dim3 *grid,
                                 dim3 *block,
                                 const int sharedMem,
                                 hipStream_t stream,
                                 int n,
                                 double *a,
                                 const int a_n1,
                                 const int a_lb1,
                                 const int a_lb2,
                                 double *x,
                                 const int x_n1,
                                 const int x_lb1,
                                 double *y,
                                 const int y_n1,
                                 const int y_lb1) {
  hipLaunchKernelGGL((dsymv_gpu), *grid, *block, sharedMem, stream, n, a, a_n1, a_lb1, a_lb2, x, x_n1, x_lb1, y, y_n1, y_lb1);
}
// END dsymv_gpu
