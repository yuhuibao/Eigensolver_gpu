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
    use hipfort
    use iso_c_binding
    use iso_c_binding_ext
    use hipfort_hipblas
    use hipfort_rocsolver
    use hipfort_check
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
                             work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info, cskip_host_copy)
        use dsygvdx_gpu_kernels
        use eigsolve_vars

        use dsygst_gpu
        use dsyevd_gpu
        implicit none
        integer                                     :: N, m, lda, ldb, ldz, il, iu, ldz_h, info, nb
        integer                                     :: lwork_h, liwork_h, lwork, istat
        real(8), target,dimension(1:lwork)         :: work
        real(8), target,dimension(1:lwork_h)       :: work_h
        integer, target,dimension(1:liwork_h)     :: iwork_h

        logical, optional                           :: cskip_host_copy

        real(8), target,dimension(1:lda, 1:N) :: A
        real(8), target,dimension(1:ldb, 1:N)      :: B
        real(8), target,dimension(1:ldz, 1:N)      :: Z
        real(8), target,dimension(1:ldz_h, 1:N)    :: Z_h
        real(8), target,dimension(1:N)             :: w
        real(8), target,dimension(1:N)             :: w_h
        real(c_double), parameter :: one = 1.d0
        integer :: i, j
        logical :: skip_host_copy
        integer, target :: v_devInfo(1)

        skip_host_copy = .FALSE.
        if (present(cskip_host_copy)) skip_host_copy = cskip_host_copy

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
        ! 'L':only lower triangular part of B is processed, and replaced by lower triangular Cholesky factor L
        ! 'U':only upper triangular part of B is processed, and replaced by upper triangular Cholesky factor U
        istat = rocsolver_dpotrf(rocsolverHandle, rocblas_fill_upper, N, c_loc(B), ldb, c_loc(devInfo_d))
        call hipCheck(hipMemcpy(v_devInfo, devInfo_d, 1, hipMemcpyDeviceToHost))
        istat = v_devInfo(1)
        if (istat .ne. 0) then
            print *, "dsygvdx_gpu error: cusolverDnDpotrf failed!"
            info = -1
            return
        endif

        ! Store lower triangular part of A in Z
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_959801_0_auto(0, stream1, c_loc(z), ldz, N, 1, 1, n, c_loc(a), lda, N, 1, 1)

        ! Reduce to standard eigenproblem
        nb = 448
        call dsygst_gpu_h(1, 'U', N, A, lda, B, ldb, nb)

        ! Tridiagonalize and compute eigenvalues/vectors
        call dsyevd_gpu_h('V', 'U', il, iu, N, A, lda, Z, ldz, w, work, lwork, &
                          work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)

        ! Triangle solve to get eigenvectors for original general eigenproblem
        istat = hipblasDtrsm(hipblasHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, N, &
                             iu - il + 1, one, c_loc(B), ldb, c_loc(Z), ldz)

        ! Copy final eigenvectors to host
        if (.not. (skip_host_copy)) then
            istat = hipMemcpy2D(z_h, ldz_h, z, ldz, &
                                N, M, hipMemcpyDeviceToHost)
            if (istat .ne. 0) then
                print *, "dsygvdx_gpu error: cudaMemcpy2D failed!"
                info = -1
                return
            endif
        endif

    end subroutine dsygvdx_gpu_h

end module dsygvdx_gpu
