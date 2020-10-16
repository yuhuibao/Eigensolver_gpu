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

// BEGIN krnl_e26a05_0
/* Fortran original:
      ! kernel do(2)<<<*,*>>>
      do j = 1, N
         do i = 1, N
            if (i > j) then
               A(i, j) = Z(i, j)
            endif
         end do
      end do

*/
// NOTE: The following information was given in the orignal Cuf kernel pragma:
// - Nested outer-most do-loops that are directly mapped to threads: 2
// - Number of blocks (CUDA): -1-1-1._ ('-1' means not specified)
// - Threads per block (CUDA): -1-1-1._ ('-1' means not specified)
// - Shared Memory: 0
// - Stream: 0

__global__ void krnl_e26a05_0(hipDoubleComplex *z,
                              const int z_n1,
                              const int z_n2,
                              const int z_lb1,
                              const int z_lb2,
                              int n,
                              hipDoubleComplex *a,
                              const int a_n1,
                              const int a_n2,
                              const int a_lb1,
                              const int a_lb2) {
#undef _idx_z
#define _idx_z(a, b) ((a - (z_lb1)) + z_n1 * (b - (z_lb2)))
#undef _idx_a
#define _idx_a(a, b) ((a - (a_lb1)) + a_n1 * (b - (a_lb2)))

  unsigned int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int i = 1 + threadIdx.x + blockIdx.x * blockDim.x;
  if ((j <= n) && (i <= n)) {
    if ((i > j)) {
      a[_idx_a(i, j)] = z[_idx_z(i, j)];
    }
  }
}

extern "C" void launch_krnl_e26a05_0(dim3 *grid,
                                     dim3 *block,
                                     const int sharedMem,
                                     hipStream_t stream,
                                     hipDoubleComplex *z,
                                     const int z_n1,
                                     const int z_n2,
                                     const int z_lb1,
                                     const int z_lb2,
                                     int n,
                                     hipDoubleComplex *a,
                                     const int a_n1,
                                     const int a_n2,
                                     const int a_lb1,
                                     const int a_lb2) {
  hipLaunchKernelGGL((krnl_e26a05_0), *grid, *block, sharedMem, stream, z, z_n1, z_n2, z_lb1, z_lb2, n, a, a_n1, a_n2, a_lb1, a_lb2);
}
extern "C" void launch_krnl_e26a05_0_auto(const int sharedMem,
                                          hipStream_t stream,
                                          hipDoubleComplex *z,
                                          const int z_n1,
                                          const int z_n2,
                                          const int z_lb1,
                                          const int z_lb2,
                                          int n,
                                          hipDoubleComplex *a,
                                          const int a_n1,
                                          const int a_n2,
                                          const int a_lb1,
                                          const int a_lb2) {
  const unsigned int krnl_e26a05_0_NX = n;
  const unsigned int krnl_e26a05_0_NY = n;

  const unsigned int krnl_e26a05_0_blockX = 16;
  const unsigned int krnl_e26a05_0_blockY = 16;

  const unsigned int krnl_e26a05_0_gridX = divideAndRoundUp(krnl_e26a05_0_NX, krnl_e26a05_0_blockX);
  const unsigned int krnl_e26a05_0_gridY = divideAndRoundUp(krnl_e26a05_0_NY, krnl_e26a05_0_blockY);

  dim3 grid(krnl_e26a05_0_gridX, krnl_e26a05_0_gridY);
  dim3 block(krnl_e26a05_0_blockX, krnl_e26a05_0_blockY);
  hipLaunchKernelGGL((krnl_e26a05_0), grid, block, sharedMem, stream, z, z_n1, z_n2, z_lb1, z_lb2, n, a, a_n1, a_n2, a_lb1, a_lb2);
}
// END krnl_e26a05_0

// BEGIN krnl_b1ccc1_1
/* Fortran original:
      ! kernel do(2)<<<*, *, 0, stream1>>>
      do j = 1, K
         do i = N - K + 1, N
            if (i - N + K == j) then
               V(i, j) = dcmplx(1, 0)
            else if (i - N + k > j) then
               W(i - N + k, j) = V(i, j)
               V(i, j) = dcmplx(0, 0)
            endif
         end do
      end do

*/
// NOTE: The following information was given in the orignal Cuf kernel pragma:
// - Nested outer-most do-loops that are directly mapped to threads: 2
// - Number of blocks (CUDA): -1-1-1._ ('-1' means not specified)
// - Threads per block (CUDA): -1-1-1._ ('-1' means not specified)
// - Shared Memory: 0
// - Stream: stream1

__global__ void krnl_b1ccc1_1(int k,
                              int n,
                              hipDoubleComplex *w,
                              const int w_n1,
                              const int w_n2,
                              const int w_lb1,
                              const int w_lb2,
                              hipDoubleComplex *v,
                              const int v_n1,
                              const int v_n2,
                              const int v_lb1,
                              const int v_lb2) {
#undef _idx_w
#define _idx_w(a, b) ((a - (w_lb1)) + w_n1 * (b - (w_lb2)))
#undef _idx_v
#define _idx_v(a, b) ((a - (v_lb1)) + v_n1 * (b - (v_lb2)))

  unsigned int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int i = (n - k + 1) + threadIdx.x + blockIdx.x * blockDim.x;
  if ((j <= k) && (i <= n)) {
    if (((i - n + k) == j)) {
      v[_idx_v(i, j)] = make_doubleComplex(1, 0);

    } else if (((i - n + k) > j)) {
      w[_idx_w((i - n + k), j)] = v[_idx_v(i, j)];
      v[_idx_v(i, j)] = make_doubleComplex(0, 0);
    }
  }
}

extern "C" void launch_krnl_b1ccc1_1(dim3 *grid,
                                     dim3 *block,
                                     const int sharedMem,
                                     hipStream_t stream,
                                     int k,
                                     int n,
                                     hipDoubleComplex *w,
                                     const int w_n1,
                                     const int w_n2,
                                     const int w_lb1,
                                     const int w_lb2,
                                     hipDoubleComplex *v,
                                     const int v_n1,
                                     const int v_n2,
                                     const int v_lb1,
                                     const int v_lb2) {
  hipLaunchKernelGGL((krnl_b1ccc1_1), *grid, *block, sharedMem, stream, k, n, w, w_n1, w_n2, w_lb1, w_lb2, v, v_n1, v_n2, v_lb1, v_lb2);
}
extern "C" void launch_krnl_b1ccc1_1_auto(const int sharedMem,
                                          hipStream_t stream,
                                          int k,
                                          int n,
                                          hipDoubleComplex *w,
                                          const int w_n1,
                                          const int w_n2,
                                          const int w_lb1,
                                          const int w_lb2,
                                          hipDoubleComplex *v,
                                          const int v_n1,
                                          const int v_n2,
                                          const int v_lb1,
                                          const int v_lb2) {
  const unsigned int krnl_b1ccc1_1_NX = k;
  const unsigned int krnl_b1ccc1_1_NY = n;

  const unsigned int krnl_b1ccc1_1_blockX = 16;
  const unsigned int krnl_b1ccc1_1_blockY = 16;

  const unsigned int krnl_b1ccc1_1_gridX = divideAndRoundUp(krnl_b1ccc1_1_NX, krnl_b1ccc1_1_blockX);
  const unsigned int krnl_b1ccc1_1_gridY = divideAndRoundUp(krnl_b1ccc1_1_NY, krnl_b1ccc1_1_blockY);

  dim3 grid(krnl_b1ccc1_1_gridX, krnl_b1ccc1_1_gridY);
  dim3 block(krnl_b1ccc1_1_blockX, krnl_b1ccc1_1_blockY);
  hipLaunchKernelGGL((krnl_b1ccc1_1), grid, block, sharedMem, stream, k, n, w, w_n1, w_n2, w_lb1, w_lb2, v, v_n1, v_n2, v_lb1, v_lb2);
}
// END krnl_b1ccc1_1

// BEGIN krnl_b95769_2
/* Fortran original:
      ! kernel do(2)<<<*, *>>>
      do j = 1, K
         do i = M - K + 1, M
            if (i - M + k > j) then
               V(i, j) = W(i - M + k, j)
            endif
         end do
      end do

*/
// NOTE: The following information was given in the orignal Cuf kernel pragma:
// - Nested outer-most do-loops that are directly mapped to threads: 2
// - Number of blocks (CUDA): -1-1-1._ ('-1' means not specified)
// - Threads per block (CUDA): -1-1-1._ ('-1' means not specified)
// - Shared Memory: 0
// - Stream: 0

__global__ void krnl_b95769_2(int m,
                              int k,
                              hipDoubleComplex *w,
                              const int w_n1,
                              const int w_n2,
                              const int w_lb1,
                              const int w_lb2,
                              hipDoubleComplex *v,
                              const int v_n1,
                              const int v_n2,
                              const int v_lb1,
                              const int v_lb2) {
#undef _idx_w
#define _idx_w(a, b) ((a - (w_lb1)) + w_n1 * (b - (w_lb2)))
#undef _idx_v
#define _idx_v(a, b) ((a - (v_lb1)) + v_n1 * (b - (v_lb2)))

  unsigned int j = 1 + threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int i = (m - k + 1) + threadIdx.x + blockIdx.x * blockDim.x;
  if ((j <= k) && (i <= m)) {
    if (((i - m + k) > j)) {
      v[_idx_v(i, j)] = w[_idx_w((i - m + k), j)];
    }
  }
}

extern "C" void launch_krnl_b95769_2(dim3 *grid,
                                     dim3 *block,
                                     const int sharedMem,
                                     hipStream_t stream,
                                     int m,
                                     int k,
                                     hipDoubleComplex *w,
                                     const int w_n1,
                                     const int w_n2,
                                     const int w_lb1,
                                     const int w_lb2,
                                     hipDoubleComplex *v,
                                     const int v_n1,
                                     const int v_n2,
                                     const int v_lb1,
                                     const int v_lb2) {
  hipLaunchKernelGGL((krnl_b95769_2), *grid, *block, sharedMem, stream, m, k, w, w_n1, w_n2, w_lb1, w_lb2, v, v_n1, v_n2, v_lb1, v_lb2);
}
extern "C" void launch_krnl_b95769_2_auto(const int sharedMem,
                                          hipStream_t stream,
                                          int m,
                                          int k,
                                          hipDoubleComplex *w,
                                          const int w_n1,
                                          const int w_n2,
                                          const int w_lb1,
                                          const int w_lb2,
                                          hipDoubleComplex *v,
                                          const int v_n1,
                                          const int v_n2,
                                          const int v_lb1,
                                          const int v_lb2) {
  const unsigned int krnl_b95769_2_NX = k;
  const unsigned int krnl_b95769_2_NY = m;

  const unsigned int krnl_b95769_2_blockX = 16;
  const unsigned int krnl_b95769_2_blockY = 16;

  const unsigned int krnl_b95769_2_gridX = divideAndRoundUp(krnl_b95769_2_NX, krnl_b95769_2_blockX);
  const unsigned int krnl_b95769_2_gridY = divideAndRoundUp(krnl_b95769_2_NY, krnl_b95769_2_blockY);

  dim3 grid(krnl_b95769_2_gridX, krnl_b95769_2_gridY);
  dim3 block(krnl_b95769_2_blockX, krnl_b95769_2_blockY);
  hipLaunchKernelGGL((krnl_b95769_2), grid, block, sharedMem, stream, m, k, w, w_n1, w_n2, w_lb1, w_lb2, v, v_n1, v_n2, v_lb1, v_lb2);
}
// END krnl_b95769_2

// BEGIN finish_t_block_kernel
/* Fortran original:
      implicit none
      integer, value                        :: N, ldt
      complex(8), dimension(ldt, K), device :: T
      complex(8), dimension(K), device      :: tau
      ! T_s contains only lower triangular elements of T in linear array, by row
      complex(8), dimension(2080), shared   :: T_s
      ! (i,j) --> ((i-1)*i/2 + j)
#define IJ2TRI(i,j) (ISHFT((i-1)*i,-1) + j)

      integer     :: tid, tx, ty, i, j, k, diag
      complex(8)  :: cv

      tx = threadIdx%x
      ty = threadIdx%y
      tid = (threadIdx%y - 1)*blockDim%x + tx ! Linear thread id

      ! Load T into shared memory
      if (tx <= N) then
         do j = ty, N, blockDim%y
            cv = tau(j)
            if (tx > j) then
               T_s(IJ2TRI(tx, j)) = -cv*T(tx, j)
            else if (tx == j) then
               T_s(IJ2TRI(tx, j)) = cv
            endif
         end do
      end if

      call syncthreads()

      ! Perform column by column update by first thread column
      do i = N - 1, 1, -1
         if (ty == 1) then
            if (tx > i .and. tx <= N) then
               cv = cmplx(0, 0)
               do j = i + 1, tx
                  cv = cv + T_s(IJ2TRI(j, i))*T_s(IJ2TRI(tx, j))
               end do
            endif

         endif

         call syncthreads()
         if (ty == 1 .and. tx > i .and. tx <= N) then
            T_s(IJ2TRI(tx, i)) = cv
         endif
         call syncthreads()

      end do

      call syncthreads()

      ! Write T_s to global
      if (tx <= N) then
         do j = ty, N, blockDim%y
            if (tx >= j) then
               T(tx, j) = T_s(IJ2TRI(tx, j))
            endif
         end do
      end if


*/

__global__ void finish_t_block_kernel(int n,
                                      int ldt,
                                      hipDoubleComplex *t,
                                      const int t_n1,
                                      const int t_n2,
                                      const int t_lb1,
                                      const int t_lb2,
                                      hipDoubleComplex *tau,
                                      const int tau_n1,
                                      const int tau_lb1) {
#undef _idx_t
#define _idx_t(a, b) ((a - (t_lb1)) + t_n1 * (b - (t_lb2)))
#undef _idx_tau
#define _idx_tau(a) ((a - (tau_lb1)))

  // ! T_s contains only lower triangular elements of T in linear array, by row
  __shared__ hipDoubleComplex t_s[2080]; /* Fortran qualifiers: dimension(2080),SHARED */
  // ! (i,j) --> ((i-1)*i/2 + j)
  // ! TODO could not parse: #define ij2tri(i,j) (ishft((i-1)*i,-1) + j)
  int tid;
  int tx;
  int ty;
  int i;
  int j;
  int k;
  int diag;
  hipDoubleComplex cv;
  tx = threadIdx.x;
  ty = threadIdx.y;
  tid = ((threadIdx.y - 1) * blockDim.x + tx);
  // ! Linear thread id
  // ! Load T into shared memory
  if ((tx <= n)) {
    for (int j = ty; j <= n; j += blockDim.y) {
      cv = tau[_idx_tau(j)];
      if ((tx > j)) {
        t_s[_idx_t_s(ij2tri[_idx_ij2tri(tx, j)])] = (-cv * t[_idx_t(tx, j)]);

      } else if ((tx == j)) {
        t_s[_idx_t_s(ij2tri[_idx_ij2tri(tx, j)])] = cv;
      }
    }
  }
  __syncthreads() // ! Perform column by column update by first thread column
      for (int i = (n - 1); i <= 1; i += -1) {
    if ((ty == 1)) {
      if ((tx > i & tx <= n)) {
        cv = make_floatComplex(0, 0);
        for (int j = (i + 1); j <= tx; j += 1) {
          cv = (cv + t_s[_idx_t_s(ij2tri[_idx_ij2tri(j, i)])] * t_s[_idx_t_s(ij2tri[_idx_ij2tri(tx, j)])]);
        }
      }
    }
    __syncthreads() if ((ty == 1 & tx > i & tx <= n)) { t_s[_idx_t_s(ij2tri[_idx_ij2tri(tx, i)])] = cv; }
    __syncthreads()
  }
  __syncthreads() // ! Write T_s to global
      if ((tx <= n)) {
    for (int j = ty; j <= n; j += blockDim.y) {
      if ((tx >= j)) {
        t[_idx_t(tx, j)] = t_s[_idx_t_s(ij2tri[_idx_ij2tri(tx, j)])];
      }
    }
  }
}

extern "C" void launch_finish_t_block_kernel(dim3 *grid,
                                             dim3 *block,
                                             const int sharedMem,
                                             hipStream_t stream,
                                             int n,
                                             int ldt,
                                             hipDoubleComplex *t,
                                             const int t_n1,
                                             const int t_n2,
                                             const int t_lb1,
                                             const int t_lb2,
                                             hipDoubleComplex *tau,
                                             const int tau_n1,
                                             const int tau_lb1) {
  hipLaunchKernelGGL((finish_t_block_kernel), *grid, *block, sharedMem, stream, n, ldt, t, t_n1, t_n2, t_lb1, t_lb2, tau, tau_n1, tau_lb1);
}
// END finish_t_block_kernel
