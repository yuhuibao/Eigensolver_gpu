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

// BEGIN dsytd2_gpu
/* Fortran original:
      use cudafor

      implicit none

      integer, value    :: lda

      real(8), device    :: a(lda, 32), tau(32)

      real(8), device    :: d(32), e(32)

      real(8), shared    :: a_s(32, 32)

      real(8), shared    :: alpha

      real(8), shared    :: taui

      real(8)           :: beta

      real(8)           :: alphar

      real(8)           :: xnorm, x, y, z, w

      real(8)           :: wc

      integer, value    :: n

      integer           :: tx, ty, tl, i, j, ii

      tx = threadIdx%x

      ty = threadIdx%y

      ! Linear id of the thread (tx,ty)

      tl = tx + blockDim%x*(ty - 1)

      ! Load a_d in shared memory

      if (tx <= N .and. ty <= N) then

         a_s(tx, ty) = a(tx, ty)

      endif

      call syncthreads()

      ! Symmetric matrix from upper triangular

      if (tx > ty) then

         a_s(tx, ty) = a_s(ty, tx)

      end if

      call syncthreads()

      ! For each column working backward

      do i = n - 1, 1, -1

         ! Generate elementary reflector

         ! Sum the vectors above the diagonal, only one warp active

         ! Reduce in a warp

         if (tl <= 32) then

            if (tl < i) then

               w = a_s(tl, i + 1)*a_s(tl, i + 1)

            else

               w = 0._8

            endif

            xnorm = __shfl_down(w, 1)

            w = w + xnorm

            xnorm = __shfl_down(w, 2)

            w = w + xnorm

            xnorm = __shfl_down(w, 4)

            w = w + xnorm

            xnorm = __shfl_down(w, 8)

            w = w + xnorm

            xnorm = __shfl_down(w, 16)

            w = w + xnorm

         end if

         if (tl == 1) then

            alpha = a_s(i, i + 1)

            alphar = dble(alpha)

            xnorm = dsqrt(w)

            if (xnorm .eq. 0_8) then

               ! H=1

               taui = 0._8

               alpha = 1.d0 ! To prevent scaling by dscal in this case

            else

               !Compute sqrt(alphar^2+xnorm^2) with  dlapy2(alphar,xnorm)

               x = abs(alphar)

               y = abs(xnorm)

               w = max(x, y)

               z = min(x, y)

               if (z .eq. 0.d0) then

                  beta = -sign(w, alphar)

               else

                  beta = -sign(w*sqrt(1.d0 + (z/w)**2), alphar)

               endif

               taui = (beta - alphar)/beta

               alpha = 1.d0/(alphar - beta) ! scale factor for dscal

            end if

         end if

         call syncthreads()

         ! dscal

         if (tl < i) then

            a_s(tl, i + 1) = a_s(tl, i + 1)*alpha

         end if

         if (tl == 1) then

            if (xnorm .ne. 0_8) then

               alpha = beta

            else

               alpha = a_s(i, i + 1) ! reset alpha to original value

            endif

            e(i) = alpha

         end if

         if (taui .ne. (0.d0, 0.d0)) then

            a_s(i, i + 1) = 1.d0

            call syncthreads()

            if (tl <= i) then

               tau(tl) = 0.d0

               do j = 1, i

                  tau(tl) = tau(tl) + taui*a_s(tl, j)*a_s(j, i + 1)

               end do

            end if

            call syncthreads()

            if (tl <= 32) then

               if (tl <= i) then

                  x = -.5d0*taui*tau(tl)*a_s(tl, i + 1)

               else

                  x = 0._8

               endif

               z = __shfl_xor(x, 1)

               x = x + z

               z = __shfl_xor(x, 2)

               x = x + z

               z = __shfl_xor(x, 4)

               x = x + z

               z = __shfl_xor(x, 8)

               x = x + z

               z = __shfl_xor(x, 16)

               x = x + z

            end if

            call syncthreads()

            if (tl <= i) then

               tau(tl) = tau(tl) + x*a_s(tl, i + 1)

            end if

            if (tl == 1) alpha = x

            call syncthreads()

            if (tx <= i .and. ty <= i) then

               a_s(tx, ty) = a_s(tx, ty) - a_s(tx, i + 1)*tau(ty) - a_s(ty, i +
   1)*tau(tx)

            end if

            call syncthreads()

         endif

         if (tl == 1) then

            a_s(i, i + 1) = e(i)

            d(i + 1) = a_s(i + 1, i + 1)

            tau(i) = taui

         end if

         call syncthreads()

      end do

      if (tl == 1) then

         d(1) = a_s(1, 1)

      endif

      call syncthreads()

      ! Back to device memory

      if (tx <= N .and. ty <= N) then

         a(tx, ty) = a_s(tx, ty)

      endif


*/

__global__ void dsytd2_gpu(int lda, double *a, const int a_n1, const int a_n2,
                           double *tau, const int tau_n1, double *d,
                           const int d_n1, double *e, const int e_n1, int n) {
#undef _idx
#undef _idx_a
#undef _idx_a_s
#define _idx(a) ((a - 1))
#define _idx_a(a, b) ((a - 1) + a_n1 * (b - 1))
#define _idx_a_s(a, b) ((a - 1) + 32 * (b - 1))
  __shared__ double a_s[1024];
  __shared__ double alpha;
  __shared__ double taui;
  double beta;
  double alphar;
  double xnorm;
  double x;
  double y;
  double z;
  double w;
  double wc;
  int tx;
  int ty;
  int tl;
  int i;
  int j;
  int ii;

  tx = threadIdx.x + 1;
  ty = threadIdx.y + 1;
  // ! Linear id of the thread (tx,ty)
  tl = tx + blockDim.x * ty;
  // ! Load a_d in shared memory
  if ((tx <= n & ty <= n)) {
    a_s[_idx_a_s(tx, ty)] = a[_idx_a(tx, ty)];
  }
  __syncthreads(); // ! Symmetric matrix from upper triangular
  if ((tx > ty)) {
    a_s[_idx_a_s(tx, ty)] = a_s[_idx_a_s(ty, tx)];
  }
  __syncthreads(); // ! For each column working backward

  // ! Generate elementary reflector
  // ! Sum the vectors above the diagonal, only one warp active
  // ! Reduce in a warp
  for (i = n - 1; i >= 1; i--) {
    if ((tl <= 32)) {
      if ((tl < i)) {
        w = (a_s[_idx_a_s(tl, (i + 1))] * a_s[_idx_a_s(tl, (i + 1))]);

      } else {
        w = 0. /*_8*/;
      }
      xnorm = __shfl_down(w, 1);
      w = (w + xnorm);
      xnorm = __shfl_down(w, 2);
      w = (w + xnorm);
      xnorm = __shfl_down(w, 4);
      w = (w + xnorm);
      xnorm = __shfl_down(w, 8);
      w = (w + xnorm);
      xnorm = __shfl_down(w, 16);
      w = (w + xnorm);
    }
    if (tl == 1) {
      alpha = a_s[_idx_a_s(i, (i + 1))];
      //alphar = make_double(alpha);
      xnorm = sqrt(w);
      if (xnorm == 0 /*_8*/) {
        // ! H=1
        taui = 0. /*_8*/;
        alpha = 1.e0;
        // ! To prevent scaling by dscal in this case

      } else {
        // !Compute sqrt(alphar^2+xnorm^2) with  dlapy2(alphar,xnorm)
        x = abs(alphar);
        y = abs(xnorm);
        w = max(x, y);
        z = min(x, y);
        if (z == 0.e0) {
          beta = -sign(w, alphar);

        } else {
          beta = -sign((w * sqrt(pow((1.e0 + (z / w)),2))), alphar);
        }
        taui = ((beta - alphar) / beta);
        alpha = (1.e0 / (alphar - beta));
        // ! scale factor for dscal
      }
    }
    __syncthreads(); // ! dscal
    if ((tl < i)) {
      a_s[_idx_a_s(tl, (i + 1))] = (a_s[_idx_a_s(tl, (i + 1))] * alpha);
    }
    if (tl == 1) {
      if ((xnorm != 0 /*_8*/)) {
        alpha = beta;

      } else {
        alpha = a_s[_idx_a_s(i, (i + 1))];
        // ! reset alpha to original value
      }
      e[_idx(i)] = alpha;
    }
    // ! TODO could not parse:           if (taui .ne. (0.d0, 0.d0)) then
    if (taui != 0.e0) {
      a_s[_idx_a_s(i, (i + 1))] = 1.e0;
      __syncthreads();
      if ((tl <= i)) {
        tau[tl] = 0.e0;
        for (int j = 1; j <= i; j += 1) {
          tau[tl] = (tau[tl] +
                     taui * a_s[_idx_a_s(tl, j)] * a_s[_idx_a_s(j, (i + 1))]);
        }
      }
      __syncthreads();
      if ((tl <= 32)) {
        if ((tl <= i)) {
          x = (-.5e0 * taui * tau[tl] * a_s[_idx_a_s(tl, (i + 1))]);

        } else {
          x = 0. /*_8*/;
        }
        z = __shfl_xor(x, 1);
        x = (x + z);
        z = __shfl_xor(x, 2);
        x = (x + z);
        z = __shfl_xor(x, 4);
        x = (x + z);
        z = __shfl_xor(x, 8);
        x = (x + z);
        z = __shfl_xor(x, 16);
        x = (x + z);
      }
      __syncthreads();
      if ((tl <= i)) {
        tau[tl] = (tau[tl] + x * a_s[_idx_a_s(tl, (i + 1))]);
      }
      if (tl == 1) {
        alpha = x;
      }
      __syncthreads();
      if ((tx <= i & ty <= i)) {
        a_s[_idx_a_s(tx, ty)] = (a_s[_idx_a_s(tx, ty)] -
                                 a_s[_idx_a_s(tx, (i + 1))] * tau[_idx(ty)] -
                                 a_s[_idx_a_s(ty, (i + 1))] * tau[_idx(tx)]);
      }
      __syncthreads(); // ! TODO could not parse:           endif
    }
    if (tl == 1) {
      a_s[_idx_a_s(i, (i + 1))] = e[_idx(i)];
      d[_idx((i + 1))] = a_s[_idx_a_s((i + 1), (i + 1))];
      tau[_idx(i)] = taui;
    }
    __syncthreads(); // ! TODO could not parse:        end do
  }
  if (tl == 1) {
    d[_idx(1)] = a_s[_idx_a_s(1, 1)];
  }
  __syncthreads(); // ! Back to device memory
  if ((tx <= n & ty <= n)) {
    a[_idx_a(tx, ty)] = a_s[_idx_a_s(tx, ty)];
  }
}

extern "C" void launch_dsytd2_gpu(dim3 *grid, dim3 *block, const int sharedMem,
                                  hipStream_t stream, int lda, double *a,
                                  const int a_n1, const int a_n2, double *tau,
                                  const int tau_n1, double *d, const int d_n1,
                                  double *e, const int e_n1, int n) {
  hipLaunchKernelGGL((dsytd2_gpu), *grid, *block, sharedMem, stream, lda, a,
                     a_n1, a_n2, tau, tau_n1, d, d_n1, e, e_n1, n);
}
// END dsytd2_gpu
