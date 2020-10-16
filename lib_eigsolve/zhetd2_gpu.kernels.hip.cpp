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

// BEGIN zhetd2_gpu
/* Fortran original:
      use cudafor
      implicit none
      integer, value    :: lda
      complex(8), device :: a(lda, 32), tau(32)
      real(8), device    :: d(32), e(32)
      complex(8), shared :: a_s(32, 32)
      complex(8), shared :: alpha
      complex(8), shared :: taui
      real(8)           :: beta
      real(8)           :: alphar, alphai
      real(8)           :: xnorm, x, y, z, w
      complex(8)        :: wc
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
      ! Hermitian matrix from upper triangular
      if (tx > ty) then
         a_s(tx, ty) = conjg(a_s(ty, tx))
      end if

      ! Enforce diagonal element to be real
      if (tl == 1) a_s(n, n) = dble(a_s(n, n))

      call syncthreads()

      ! For each column working backward
      do i = n - 1, 1, -1
         ! Generate elementary reflector
         ! Sum the vectors above the diagonal, only one warp active
         ! Reduce in a warp
         if (tl <= 32) then
            if (tl < i) then
               w = a_s(tl, i + 1)*conjg(a_s(tl, i + 1))
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
            alphai = dimag(alpha)
            xnorm = dsqrt(w)

            if (xnorm .eq. 0_8 .and. alphai .eq. 0._8) then
               ! H=1
               taui = 0._8
               alpha = dcmplx(1.d0, 0.d0) ! To prevent scaling by zscal in this case
            else
               !Compute sqrt(alphar^2+alphai^2+xnorm^2) with  dlapy3(alphar,alphai,xnorm)
               x = abs(alphar)
               y = abs(alphai)
               z = abs(xnorm)
               w = max(x, y, z)
               beta = -sign(w*sqrt((x/w)**2 + (y/w)**2 + (z/w)**2), alphar)

               taui = dcmplx((beta - alphar)/beta, -alphai/beta)

               !zladiv(dcmplx(one),alpha-beta)
               x = dble(alpha - beta)
               y = dimag(alpha - beta)
               if (abs(y) .lt. abs(x)) then
                  w = y/x
                  z = x + y*w
                  alpha = dcmplx(1/z, -w/z)
               else
                  w = x/y
                  z = y + x*w
                  alpha = dcmplx(w/z, -1/z)
               end if
            end if
         end if

         call syncthreads()

         ! zscal
         if (tl < i) then
            a_s(tl, i + 1) = a_s(tl, i + 1)*alpha
         end if

         if (tl == 1) then
            if (xnorm .ne. 0_8 .or. alphai .ne. 0._8) then
               alpha = dcmplx(beta, 0._8)
            else
               alpha = a_s(i, i + 1) ! reset alpha to original value
            endif

            e(i) = alpha
         end if

         if (taui .ne. (0.d0, 0.d0)) then
            a_s(i, i + 1) = dcmplx(1.d0, 0.d0)
            call syncthreads()
            if (tl <= i) then
               tau(tl) = dcmplx(0.d0, 0.d0)
               do j = 1, i
                  tau(tl) = tau(tl) + taui*a_s(tl, j)*a_s(j, i + 1)
               end do
            end if

            call syncthreads()

            if (tl <= 32) then
               if (tl <= i) then
                  wc = taui*conjg(tau(tl))*a_s(tl, i + 1)
                  x = -.5d0*dble(wc)
                  y = -.5d0*dimag(wc)
               else
                  x = 0._8
                  y = 0._8
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

               w = __shfl_xor(y, 1)
               y = y + w
               w = __shfl_xor(y, 2)
               y = y + w
               w = __shfl_xor(y, 4)
               y = y + w
               w = __shfl_xor(y, 8)
               y = y + w
               w = __shfl_xor(y, 16)
               y = y + w
            end if

            call syncthreads()

            if (tl <= i) then
               tau(tl) = tau(tl) + dcmplx(x, y)*a_s(tl, i + 1)
            end if

            if (tl == 1) alpha = dcmplx(x, y)

            call syncthreads()

            if (tx <= i .and. ty <= i) then
               a_s(tx, ty) = a_s(tx, ty) - a_s(tx, i + 1)*dconjg(tau(ty)) - dconjg(a_s(ty, i + 1))*tau(tx)
            end if
            call syncthreads()

         else
            if (tl == 1) a_s(i, i) = dble(a_s(i, i))
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

__global__ void zhetd2_gpu(int lda,
                           hipDoubleComplex *a,
                           const int a_n1,
                           const int a_n2,
                           const int a_lb1,
                           const int a_lb2,
                           hipDoubleComplex *tau,
                           const int tau_n1,
                           const int tau_lb1,
                           double *d,
                           const int d_n1,
                           const int d_lb1,
                           double *e,
                           const int e_n1,
                           const int e_lb1,
                           int n) {
#undef _idx_a
#define _idx_a(a, b) ((a - (a_lb1)) + a_n1 * (b - (a_lb2)))
#undef _idx_tau
#define _idx_tau(a) ((a - (tau_lb1)))
#undef _idx_d
#define _idx_d(a) ((a - (d_lb1)))
#undef _idx_e
#define _idx_e(a) ((a - (e_lb1)))
#undef _idx_a_s
#define _idx_a_s(a, b) ((a - 1) + 32 * (b - 1))

  __shared__ hipDoubleComplex* a_s; /* Fortran qualifiers: SHARED */
  __shared__ hipDoubleComplex alpha;       /* Fortran qualifiers: SHARED */
  __shared__ hipDoubleComplex taui;        /* Fortran qualifiers: SHARED */
  double beta;
  double alphar;
  double alphai;
  double xnorm;
  double x;
  double y;
  double z;
  double w;
  hipDoubleComplex wc;
  int tx;
  int ty;
  int tl;
  int i;
  int j;
  int ii;
  tx = threadIdx.x + 1;
  ty = threadIdx.y + 1;
  // ! Linear id of the thread (tx,ty)
  tl = (tx + blockDim.x * (ty - 1));
  // ! Load a_d in shared memory
  if ((tx <= n & ty <= n)) {
    a_s[_idx_a_s(tx, ty)] = a[_idx_a(tx, ty)];
  }
  __syncthreads(); // ! Hermitian matrix from upper triangular
  if ((tx > ty)) {
    a_s[_idx_a_s(tx, ty)] = conj(a_s[_idx_a_s(ty, tx)]);
  }
  // ! Enforce diagonal element to be real
  if ((tl == 1)) {
    a_s[_idx_a_s(n, n)] = make_double(a_s[_idx_a_s(n, n)]);
  }
  __syncthreads(); // ! For each column working backward
      for (i = n - 1; i >= 1; i--) {
    // ! Generate elementary reflector
    // ! Sum the vectors above the diagonal, only one warp active
    // ! Reduce in a warp
    if ((tl <= 32)) {
      if ((tl < i)) {
        w = (a_s[_idx_a_s(tl, (i + 1))] * conj(a_s[_idx_a_s(tl, (i + 1))]));

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
    if ((tl == 1)) {
      alpha = a_s[_idx_a_s(i, (i + 1))];
      alphar = make_double(alpha);
      alphai = dimag[(alpha)];
      xnorm = sqrt(w);
      if ((xnorm == 0 /*_8*/ & alphai == 0. /*_8*/)) {
        // ! H=1
        taui = 0. /*_8*/;
        alpha = make_doubleComplex(1.e0, 0.e0);
        // ! To prevent scaling by zscal in this case

      } else {
        // !Compute sqrt(alphar^2+alphai^2+xnorm^2) with  dlapy3(alphar,alphai,xnorm)
        x = abs(alphar);
        y = abs(alphai);
        z = abs(xnorm);
        w = max(x, y, z);
        beta = -sign((w * sqrt(((x / w) * *((2 + (y / w)) * *((2 + (z / w)) * *2))))), alphar);
        taui = make_doubleComplex(((beta - alphar) / beta), (-alphai / beta));
        // !zladiv(dcmplx(one),alpha-beta)
        x = make_double((alpha - beta));
        y = dimag[((alpha - beta))];
        if ((abs(y) < abs(x))) {
          w = (y / x);
          z = (x + y * w);
          alpha = make_doubleComplex((1 / z), (-w / z));

        } else {
          w = (x / y);
          z = (y + x * w);
          alpha = make_doubleComplex((w / z), (-1 / z));
        }
      }
    }
    __syncthreads(); // ! zscal
    if ((tl < i)) {
      a_s[_idx_a_s(tl, (i + 1))] = (a_s[_idx_a_s(tl, (i + 1))] * alpha);
    }
    if ((tl == 1)) {
      if ((xnorm != 0 /*_8*/ | alphai != 0. /*_8*/)) {
        alpha = make_doubleComplex(beta, 0. /*_8*/);

      } else {
        alpha = a_s[_idx_a_s(i, (i + 1))];
        // ! reset alpha to original value
      }
      e[_idx_e(i)] = alpha;
    }
    if ((taui != make_hipComplex(0.e0, 0.e0))) {
      a_s[_idx_a_s(i, (i + 1))] = make_doubleComplex(1.e0, 0.e0);
      __syncthreads();
       if ((tl <= i)) {
        tau[_idx_tau(tl)] = make_doubleComplex(0.e0, 0.e0);
        for (int j = 1; j <= i; j += 1) {
          tau[_idx_tau(tl)] = (tau[_idx_tau(tl)] + taui * a_s[_idx_a_s(tl, j)] * a_s[_idx_a_s(j, (i + 1))]);
        }
       }
      __syncthreads();
      if ((tl <= 32)) {
        if ((tl <= i)) {
          wc = (taui * conj(tau[_idx_tau(tl)]) * a_s[_idx_a_s(tl, (i + 1))]);
          x = (-.5e0 * make_double(wc));
          y = (-.5e0 * dimag[(wc)]);

        } else {
          x = 0. /*_8*/;
          y = 0. /*_8*/;
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
        w = __shfl_xor(y, 1);
        y = (y + w);
        w = __shfl_xor(y, 2);
        y = (y + w);
        w = __shfl_xor(y, 4);
        y = (y + w);
        w = __shfl_xor(y, 8);
        y = (y + w);
        w = __shfl_xor(y, 16);
        y = (y + w);
      }
      __syncthreads();
      if (tl <= i) { 
          tau[_idx_tau(tl)] = (tau[_idx_tau(tl)] + make_doubleComplex(x, y) * a_s[_idx_a_s(tl, (i + 1))]);
      }
      if (tl == 1) {
        alpha = make_doubleComplex(x, y);
      }
      __syncthreads();
      if ((tx <= i & ty <= i)) {
        a_s[_idx_a_s(tx, ty)] = (a_s[_idx_a_s(tx, ty)] - a_s[_idx_a_s(tx, (i + 1))] * conj(tau[_idx_tau(ty)]) -
                                 conj(a_s[_idx_a_s(ty, (i + 1))]) * tau[_idx_tau(tx)]);
      }
      __syncthreads();
    } else {
      if (tl == 1) {
        a_s[_idx_a_s(i, i)] = make_double(a_s[_idx_a_s(i, i)]);
      }
    }
    if (tl == 1) {
      a_s[_idx_a_s(i, (i + 1))] = e[_idx_e(i)];
      d[_idx_d((i + 1))] = a_s[_idx_a_s((i + 1), (i + 1))];
      tau[_idx_tau(i)] = taui;
    }
    __syncthreads();
  }
  if (tl == 1) {
    d[_idx_d(1)] = a_s[_idx_a_s(1, 1)];
  }
  __syncthreads(); // ! Back to device memory
  if ((tx <= n & ty <= n)) {
    a[_idx_a(tx, ty)] = a_s[_idx_a_s(tx, ty)];
  }
}


// END zhetd2_gpu
