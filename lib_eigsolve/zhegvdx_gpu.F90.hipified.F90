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

module zhegvdx_gpu
    use hipfort
    use iso_c_binding
    use iso_c_binding_ext
    use hipfort_hipblas
    use hipfort_rocsolver
    use hipfort_check

    implicit none

contains

    ! zhegvdx_gpu
    ! This solver computes eigenvalues and associated eigenvectors over a specified integer range for a
    ! hermetian-definite eigenproblem in the following form:
    !     A * x = lambda * B * x
    ! where A and B are hermetian-matrices and B is positive definite. The solver expects the upper-triangular parts of the
    ! input A and B arguments to be populated. This configuration corresponds to calling ZHEGVX within LAPACK with the configuration
    ! arguments 'ITYPE = 1', 'JOBZ = 'V'', 'RANGE = 'I'', and 'UPLO = 'U''.
    !
    ! Input:
    ! On device:
    !   -  A(lda, N), B(ldb, N) are double-complex matrices on device  with upper triangular portion populated
    !   -  il, iu are integers specifying range of eigenvalues/vectors to compute. Range is [il, iu]
    !   -  work is a double-complex array for complex workspace of length lwork.
    !   -  lwork is an integer specifying length of work. lwork >=  2*64*64 + 65*N
    !   -  rwork is a real(8) array for real workspace of length lrwork.
    !   -  lrwork is an integer specifying length of rwork. lrwork >= N
    !
    ! On host:
    !   -  work_h is a double-complex array for complex workspace of length lwork_h.
    !   -  lwork_h is an integer specifying length of work_h. lwork_h >= N
    !   -  rwork_h is a real(8) array for complex workspace of length lrwork_h.
    !   -  lrwork_h is an integer specifying length of rwork_h. lrwork_h >= 1 + 5*N + 2*N*N
    !   -  iwork_h is a integer array for integer workspace of length liwork_h.
    !   -  liwork_h is an integer specifying length of iwork_h. liwork_h >= 3 + 5*N
    !   -  (optional) _skip_host_copy is an optional logical argument. If .TRUE., memcopy of final updated eigenvectors from
    !      device to host will be skipped.
    !
    ! Output:
    ! On device:
    !   - A(lda, N), B(ldb, N) are modified on exit. The upper triangular part of A, including the diagonal is destroyed.
    !     B is overwritten by the triangular Cholesky factor U corresponding to  B = U**H * U
    !   - Z(ldz, N) is a double-complex matrix on the device. On exit, the first iu - il + 1 columns of Z
    !     contains normalized eigenvectors corresponding to eigenvalues in the range [il, iu].
    !   - w(N) is a real(8) array on the device. On exit, the first iu - il + 1 values of w contain the computed
    !     eigenvalues
    !
    ! On host:
    !   - Z_h(ldz_h, N) is a double-complex matrix on the host. On exit, the first iu - il + 1 columns of Z
    !     contains normalized eigenvectors corresponding to eigenvalues in the range [il, iu]. This is a copy of the Z
    !     matrix on the device. This is only true if optional argument _skip_host_copy is not provided or is set to .FALSE.
    !   - w_h(N) is a real(8) array on the host. On exit, the first iu - il + 1 values of w contain the computed
    !     eigenvalues. This is a copy of the w array on the host.
    !   - info is an integer. info will equal zero if the function completes succesfully. Otherwise, there was an error.
    !
    subroutine zhegvdx_gpu_h(N, A, lda, B, ldb, Z, ldz, il, iu, w, work, lwork, rwork, lrwork, &
                             work_h, lwork_h, rwork_h, lrwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info, cskip_host_copy)
        use zhegvdx_gpu_kernels
        use eigsolve_vars

        use zhegst_gpu
        use zheevd_gpu
        implicit none
        integer                                     :: N, m, lda, ldb, ldz, il, iu, ldz_h, info, nb
        integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

        type(c_ptr), value :: work
        integer(c_int) :: work_n1, work_lb1

        type(c_ptr), value :: work_h
        integer(c_int) :: work_h_n1, work_h_lb1

        type(c_ptr), value :: rwork

        integer(c_int) :: rwork_n1, rwork_lb1

        type(c_ptr), value :: rwork_h
        integer(c_int) :: rwork_h_n1, rwork_h_lb1

        type(c_ptr), value :: iwork_h
        integer(c_int) :: iwork_h_n1, iwork_h_lb1
        logical, optional                           :: cskip_host_copy

        type(c_ptr), value :: A
        integer(c_int) :: A_n1, A_n2, A_lb1 = 1, A_lb2 = 1
        type(c_ptr), value :: B
        integer(c_int) :: B_n1, B_n2, B_lb1 = 1, B_lb2 = 1
        type(c_ptr), value :: Z
        integer(c_int) :: Z_n1, Z_n2, Z_lb1 = 1, Z_lb2 = 1
        type(c_ptr), value :: Z_h
        integer(c_int) :: Z_h_n1, Z_h_n2, Z_h_lb1 = 1, Z_h_lb2 = 1
        type(c_ptr), value :: w
        integer(c_int) :: w_n1, w_lb1 = 1
        type(c_ptr), value :: w_h
        integer(c_int) :: w_h_n1, w_h_lb1 = 1

        complex(8), parameter :: cone = cmplx(1, 0, 8)
        integer :: i, j
        logical :: skip_host_copy
        integer, target :: v_devInfo
        work_n1 = lwork
        work_lb1 = 1
        work_h_n1 = lwork_h
        work_h_lb1 = 1
        rwork_n1 = lrwork
        rwork_lb1 = 1
        rwork_h_n1 = lrwork_h
        rwork_h_lb1 = 1
        iwork_h_n1 = liwork_h
        iwork_h_lb1 = 1
        A_n1 = lda
        A_n2 = N
        B_n1 = ldb
        B_n2 = N
        Z_n1 = ldz
        Z_n2 = N
        info = 0
        Z_h_n1 = ldz_h
        Z_h_n2 = N
        w_n1 = N
        w_h_n1 = N
        info = 0
        skip_host_copy = .FALSE.
        if (present(cskip_host_copy)) skip_host_copy = cskip_host_copy

        ! Check workspace sizes
        if (lwork < 2*64*64 + 65*N) then
            print *, "zhegvdx_gpu error: lwork must be at least 2*64*64 + 65*N"
            info = -1
            return
        else if (lrwork < N) then
            print *, "zhegvdx_gpu error: lrwork must be at least N"
            info = -1
            return
        else if (lwork_h < N) then
            print *, "zhegvdx_gpu error: lwork_h must be at least N"
            info = -1
            return
        else if (lrwork_h < 1 + 5*N + 2*N*N) then
            print *, "zhegvdx_gpu error: lrwork_h must be at least 1 + 5*N + 2*N*N"
            info = -1
            return
        else if (liwork_h < N) then
            print *, "zhegvdx_gpu error: liwork_h must be at least 3 + 5*N"
            info = -1
            return
        endif

        m = iu - il + 1 ! Number of eigenvalues/vectors to compute

        if (initialized == 0) call init_eigsolve_gpu

        ! Compute cholesky factorization of B
        istat = rocsolver_zpotrf(rocsolverHandle, rocblas_fill_upper, N, B, ldb, devInfo_d)
        call hipCheck(hipMemcpy(c_loc(v_devInfo), devInfo_d, 1_8*(4)*(1), hipMemcpyDeviceToHost))
        istat = v_devInfo
        if (istat .ne. 0) then
            print *, "zhegvdx_gpu error: cusolverDnZpotrf failed!"
            info = -1
            return
        endif

        ! Store lower triangular part of A in Z
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_959801_0_auto(0, stream1, z, z_n1, z_n2, z_lb1, z_lb2, n, a, a_n1, a_n2, a_lb1, a_lb2)

        ! Reduce to standard eigenproblem
        nb = 448
        call zhegst_gpu_h(1, 'U', N, A, lda, B, ldb, nb)

        ! Tridiagonalize and compute eigenvalues/vectors
        call zheevd_gpu_h('V', 'U', il, iu, N, A, lda, Z, ldz, w, work, lwork, rwork, lrwork, &
                          work_h, lwork_h, rwork_h, lrwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)

        ! Triangle solve to get eigenvectors for original general eigenproblem
        istat = hipblasZtrsm(hipblasHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_LOWER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, N, &
                             (iu - il + 1), cone, B, ldb, Z, ldz)
#undef _idx_Z
#define _idx_Z(a,b) ((a-(Z_lb1))+Z_n1*(b-(Z_lb2)))
#undef _idx_Z_h
#define _idx_Z_h(a,b) ((a-(Z_h_lb1))+Z_h_n1*(b-(Z_h_lb2)))

        ! Copy final eigenvectors to host
        if (.not. (skip_host_copy)) then
            istat = hipMemcpy2D(inc_c_ptr(z_h, _idx_Z_h(1, 1)*16*1_8), 1_8*(ldz_h)*(16), &
                                inc_c_ptr(z, _idx_Z(1, 1)*16*1_8), 1_8*(ldz)*(16), &
                                1_8*(N)*(16), 1_8*(m)*(1), hipMemcpyDeviceToHost)
            if (istat .ne. 0) then
                print *, "zhegvdx_gpu error: cudaMemcpy2D failed!"
                info = -1
                return
            endif
        endif

    end subroutine zhegvdx_gpu_h

end module zhegvdx_gpu
