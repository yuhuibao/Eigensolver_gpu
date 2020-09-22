!
! Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
!
!
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.
!

module dsygvdx_gpu
   use hip
   use iso_c_binding
   use iso_c_binding_ext
   use hipblas
   use rocsolver
   implicit none

contains

   ! dsygvdx_gpu
   ! This solver computes eigenvalues and associated eigenvectors over a specified integer range for a
   ! symmetric-positive-definite eigenproblem in the following form:
   !     A * x = lambda * B * x
   ! where A and B are symmetric matrices and B is positive definite. The solver expects the upper-triangular parts of the
   ! input A and B arguments to be populated. This configuration corresponds to calling DSYGVX within LAPACK with the configuration
   ! arguments 'ITYPE = 1', 'JOBZ = 'V'', 'RANGE = 'I'', and 'UPLO = 'U''.
   !
   ! Input:
   ! On device:
   !   -  A(lda, N), B(ldb, N) are real(8) matrices on device with upper triangular portion populated
   !   -  il, iu are integers specifying range of eigenvalues/vectors to compute. Range is [il, iu]
   !   -  work is a real(8) array for real workspace of length lwork.
   !   -  lwork is an integer specifying length of work. lwork >=  2*64*64 + 66*N
   !
   ! On host:
   !   -  work_h is a double-complex array for complex workspace of length lwork_h.
   !   -  lwork_h is an integer specifying length of work_h. lwork_h >= 1 + 6*N + 2*N*N
   !   -  iwork_h is a integer array for integer workspace of length liwork_h.
   !   -  liwork_h is an integer specifying length of iwork_h. liwork_h >= 3 + 5*N
   !   -  (optional) _skip_host_copy is an optional logical argument. If .TRUE., memcopy of final updated eigenvectors from
   !      device to host will be skipped.
   !
   ! Output:
   ! On device:
   !   - A(lda, N), B(ldb, N) are modified on exit. The upper triangular part of A, including the diagonal is destroyed.
   !     B is overwritten by the triangular Cholesky factor U corresponding to  B = U**H * U
   !   - Z(ldz, N) is a real(8) matrix on the device. On exit, the first iu - il + 1 columns of Z
   !     contains normalized eigenvectors corresponding to eigenvalues in the range [il, iu].
   !   - w(N) is a real(8) array on the device. On exit, the first iu - il + 1 values of w contain the computed
   !     eigenvalues
   !
   ! On host:
   !   - Z_h(ldz_h, N) is a real(8) matrix on the host. On exit, the first iu - il + 1 columns of Z
   !     contains normalized eigenvectors corresponding to eigenvalues in the range [il, iu]. This is a copy of the Z
   !     matrix on the device. This is only true if optional argument _skip_host_copy is not provided or is set to .FALSE.
   !   - w(N) is a real(8) array on the host. On exit, the first iu - il + 1 values of w contain the computed
   !     eigenvalues. This is a copy of the w array on the host.
   !   - info is an integer. info will equal zero if the function completes succesfully. Otherwise, there was an error.
   !
   subroutine dsygvdx_gpu_h(N, A, lda, B, ldb, Z, ldz, il, iu, w, work, lwork, &
                          work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info, _skip_host_copy)
      use dsygvdx_gpu_kernels
      use eigsolve_vars

      use dsygst_gpu
      use dsyevd_gpu
      implicit none
      integer                                     :: N, m, lda, ldb, ldz, il, iu, ldz_h, info, nb
      integer                                     :: lwork_h, liwork_h, lwork, istat
      type(c_ptr) :: work 
      integer(c_int) :: work_n1 = lwork, work_lb1 = 1
      type(c_ptr) :: work_h 
      integer(c_int) :: work_h_n1 = lwork_h, work_h_lb1 = 1
      type(c_ptr) :: iwork_h 
      integer(c_int) :: iwork_h_n1 = liwork_h, iwork_h_lb1 = 1

      logical, optional                           :: _skip_host_copy

      type(c_ptr) :: A 
      integer(c_int) :: A_n1 = lda, A_n2 = N, A_lb1 = 1, A_lb2 = 1
      type(c_ptr) :: B 
      integer(c_int) :: B_n1 = ldb, B_n2 = N, B_lb1 = 1, B_lb2 = 1
      type(c_ptr) :: Z 
      integer(c_int) :: Z_n1 = ldz, Z_n2 = N, Z_lb1 = 1, Z_lb2 = 1
      type(c_ptr) :: Z_h 
      integer(c_int) :: Z_h_n1 = ldz_h, Z_h_n2 = N, Z_h_lb1 = 1, Z_h_lb2 = 1
      type(c_ptr) :: w 
      integer(c_int) :: w_n1 = N, w_lb1 = 1
      type(c_ptr) :: w_h 
      integer(c_int) :: w_h_n1 = N, w_h_lb1 = 1

      real(8), parameter :: one = 1.d0
      integer :: i, j
      logical :: skip_host_copy

      type(c_ptr) :: hipblasHandle = c_null_ptr

      info = 0
      skip_host_copy = .FALSE.
      if (present(_skip_host_copy)) skip_host_copy = _skip_host_copy

      ! Check workspace sizes
      if (lwork < 2*64*64 + 66*N) then
         print *, "dsygvdx_gpu error: lwork must be at least 2*64*64 + 66*N"
         info = -1
         return
      else if (lwork_h < 1 + 6*N + 2*N*N) then
         print *, "dsygvdx_gpu error: lwork_h must be at least 1 + 6*N + 2*N*N"
         info = -1
         return
      else if (liwork_h < N) then
         print *, "dsygvdx_gpu error: liwork_h must be at least 3 + 5*N"
         info = -1
         return
      endif

      m = iu - il + 1 ! Number of eigenvalues/vectors to compute

      if (initialized == 0) call init_eigsolve_gpu

      ! Compute cholesky factorization of B
      ! 'L':only lower triangular part of A is processed, and replaced by lower triangular Cholesky factor L
      ! 'U':only upper triangular part of A is processed, and replaced by upper triangular Cholesky factor U
      istat = cusolverDnDpotrf(cusolverHandle, HIPBLAS_FILL_modE_UPPER, N, B, ldb, work, lwork, devInfo_d)
      istat = devInfo_d
      if (istat .ne. 0) then
         print *, "dsygvdx_gpu error: cusolverDnDpotrf failed!"
         info = -1
         return
      endif

      ! Store lower triangular part of A in Z
      ! extracted to HIP C++ file
      ! TODO(gpufort) fix arguments
      CALL launch_krnl_959801_0_auto(0, stream1, z, z_n1, z_n2, z_lb1, z_lb2, n, a, a_n1, a_n2, a_lb1, a_lb2)

      ! Reduce to standard eigenproblem
      nb = 448
      call dsygst_gpu(1, 'U', N, A, lda, B, ldb, nb)

      ! Tridiagonalize and compute eigenvalues/vectors
      call dsyevd_gpu('V', 'U', il, iu, N, A, lda, Z, ldz, w, work, lwork, &
                      work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)

      ! Triangle solve to get eigenvectors for original general eigenproblem
      hipblasCreate(hipblasHandle)
      call hipblasDtrsm(hipblasHandle, 'L', 'U', HIPBLAS_OP_N, HIPBLAS_OP_N, N, (iu - il + 1), one, B, ldb, Z, ldz)
      hipblasDestroy(hipblasHandle)


      ! Copy final eigenvectors to host
      if (not(skip_host_copy)) then
         istat = hipMemcpy2D(inc_c_ptr(z_h, _idx_z_h(1,1)*8), 1_8*(ldz_h)*(8),inc_c_ptr(z, _idx_z(1,1)*8) , 1_8*(ldz)*(8), 1_8*(N)*(8), 1_8*(m)*(1), hipMemcpyDeviceToHost)
         if (istat .ne. 0) then
            print *, "dsygvdx_gpu error: cudaMemcpy2D failed!"
            info = -1
            return
         endif
      endif

   end subroutine dsygvdx_gpu_h

end module dsygvdx_gpu
