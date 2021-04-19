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

module dsyevd_gpu
    use dsyevd_gpu_kernels
    use hipfort
    use iso_c_binding
    use hipfort_hipblas
    use utils
    implicit none

contains

    ! Custom dsyevd routine
    subroutine dsyevd_gpu_h(jobz, uplo, il, iu, N, A, A_h, lda, Z, ldz, w, work, lwork, &
                            work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)
        use dsyevd_gpu_kernels
        use dsytrd_gpu
        use eigsolve_vars
        implicit none
        character                                   :: uplo, jobz
        integer                                     :: N, NZ, lda, lwork, istat, info
        integer                                     :: lwork_h, liwork_h, ldz_h
        integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu
        real(8), target, dimension(1:lwork)         :: work
        real(8), target, dimension(1:lwork_h)               :: work_h
        integer, target, dimension(1:liwork_h)              :: iwork_h

        real(8), target, dimension(1:lda, 1:N)      :: A
        real(8), target, dimension(1:lda, 1:N)      :: Z
        real(8), target, dimension(1:ldz_h, 1:N)    :: Z_h
        real(8), target, dimension(1:N)           :: w
        real(8), target, dimension(1:N)           :: w_h

        integer                                     :: inde, indtau, indwrk, llwork, llwork_h, indwk2, indwk3, llwrk2
        real(8), target, dimension(1:lda, 1:N) :: A_h
        real(8), parameter                          :: one = 1.0_8

        type(dim3) :: blocks, threads

        if (uplo .ne. 'U' .or. jobz .ne. 'V') then
            print *, "Provided itype/uplo not supported!"
            return
        endif

        nb1 = 64 ! Blocksize for tridiagonalization
        nb2 = min(64, N) ! Blocksize for rotation procedure, fixed at 64
        ldt = nb2
        NZ = iu - il + 1

        inde = 1
        indtau = inde + n
        indwrk = indtau + n
        llwork = lwork - indwrk + 1
        llwork_h = lwork_h - indwrk + 1
        indwk2 = indwrk + (nb2)*(nb2)
        indwk3 = indwk2 + (nb2)*(nb2)

        !JR Note: ADD SCALING HERE IF DESIRED. Not scaling for now.
        ! Call DSYTRD to reduce A to tridiagonal form
        call dsytrd_gpu_h('U', N, A, lda, w, work(inde), work(indtau), work(indwrk), llwork, nb1)
        call hipCheck(hipMemcpy(A_h, A, lda*N, hipMemcpyDeviceToHost))
        call print_matrix(A_h)
        call hipCheck(hipMemcpy(work_h(indtau),work(indtau),N-1, hipMemcpyDeviceToHost))
        call print_vector(work_h(indtau:indtau+N-2))

        ! Copy diagonal and superdiagonal to CPU
        !w_h(1:N) = w(1:N)
        call hipCheck(hipMemcpy(w_h, w, N, hipMemcpyDeviceToHost))

        !work_h(inde:inde+N-1-1) = work(inde:inde+N-1-1)
        call hipCheck(hipMemcpy(work_h(inde:inde + N - 1 -1), work(inde:inde + N - 1-1), N, hipMemcpyDeviceToHost))
        ! call print_vector(w_h)
        ! call print_vector(work_h(inde:inde + N - 1 - 1))

        ! Restore lower triangular of A (works if called from zhegvd only!)
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_e26a05_0_auto(0, c_null_ptr, c_loc(z), lda, N, 1, 1, n, c_loc(a), lda, N, 1, 1)

        ! Call DSTEDC to get eigenvalues/vectors of tridiagonal A on CPU

        call dstedc('I', N, w_h, work_h(inde), Z_h, ldz_h, work_h(indwrk), llwork_h, iwork_h, liwork_h, istat)
        if (istat /= 0) then
            write (*, *) "dsyevd_gpu error: dstedc failed!"
            info = -1
            return
        endif
        print*,"after dstedc:"
        call print_matrix(Z_h)
        !call print_vector(w_h)
        ! Copy eigenvectors and eigenvalues to GPU
        call hipCheck(hipMemcpy2D(Z, ldz, Z_h, ldz_h, N, NZ, hipMemcpyHostToDevice))
        !w(1:N) = w_h(1:N)
        call hipCheck(hipMemcpy(w, w_h, N, hipMemcpyHostToDevice))
      !! Call DORMTR to rotate eigenvectors to obtain result for original A matrix
      !! JR Note: Eventual function calls from DORMTR called directly here with associated indexing changes

        call hipCheck(hipEventRecord(event2, stream2))

        k = N - 1

        do i = 1, k, nb2
            ib = min(nb2, k - i + 1)

            ! Form block reflector T in stream 1
            call dlarft_gpu(i + ib - 1, ib, A(1, 2 + i - 1), lda, work(indtau + i - 1), work(indwrk), ldt, work(indwk2), ldt)

            mi = i + ib - 1
            ! Apply reflector to eigenvectors in stream 2
            call dlarfb_gpu(mi, NZ, ib, A(1, 2 + i - 1), lda, work(indwrk), ldt, Z, ldz, work(indwk3), N, work(indwk2), ldt)
        end do
        print*,"end iteration"
        call hipCheck(hipMemcpy(Z_h, Z, ldz*N, hipMemcpyDeviceToHost))
        call print_matrix(Z_h)
    end subroutine dsyevd_gpu_h

    subroutine dlarft_gpu(N, K, V, ldv, tau, T, ldt, W, ldw)
        use dsyevd_gpu_kernels
        use hipfort_hipblas

        use eigsolve_vars
        implicit none
        integer                               :: N, K, ldv, ldt, ldw

        real(8), target, dimension(ldv, K)    :: V
        real(8), target, dimension(K)         :: tau
        real(8), target, dimension(ldt, K)    :: T
        real(8), dimension(ldt, K)            :: T_h
        real(8), target, dimension(ldw, K)    :: W

        integer                               :: i, j, istat1
        type(dim3)                            :: threads

        call hipblasCheck(hipblasSetStream(hipblasHandle, stream1))
        ! Prepare lower triangular part of block column for dsyrk call.
        ! Requires zeros in lower triangular portion and ones on diagonal.
        ! Store existing entries (excluding diagonal) in W
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_b1f342_1_auto(0, stream1, c_loc(w), ldw, K, 1, 1, n, c_loc(v), ldt, K, 1, 1, K)

        call hipCheck(hipEventRecord(event1, stream1))
        call hipCheck(hipStreamWaitEvent(stream1, event2, 0))

        ! Form preliminary T matrix
        call hipblasCheck(hipblasdsyrk(hipblasHandle, HIPBLAS_FILL_modE_LOWER, HIPBLAS_OP_T, K, N, 1.0_8, V, ldv, 0.0_8, T, ldt))

        call hipCheck(hipMemcpy(T_h,T,ldt*K,hipMemcpyDeviceToHost))
        call print_matrix(T_h)

        ! Finish forming T
        threads = dim3(64, 4, 1)
        CALL launch_finish_t_block_kernel(dim3(1, 1, 1), threads, 1040*8, stream1, K, ldt, c_loc(T), ldt, K, 1, 1, c_loc(tau), K, 1)
        call hipCheck(hipMemcpy(T_h,T,ldt*K,hipMemcpyDeviceToHost))
        call print_matrix(T_h)
    end subroutine dlarft_gpu

    subroutine dlarfb_gpu(M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, W, ldw)
        use dsyevd_gpu_kernels
        use hipfort_hipblas

        use eigsolve_vars
        implicit none
        integer                               :: M, N, K, ldv, ldt, ldc, ldw, ldwork, istat
        integer                               :: i, j

        real(8), target, dimension(ldv, K)    :: V
        real(8), target, dimension(ldt, K)    :: T
        real(8), target, dimension(ldw, K)    :: W
        real(8), target, dimension(ldc, N)    :: C
        real(8), target, dimension(ldwork, K) :: work
        call hipCheck(hipblasSetStream(hipblasHandle, stream2))
        call hipCheck(hipStreamWaitEvent(stream2, event1, 0))
     call hipblasCheck(hipblasdgemm(hipblasHandle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, K, M, 1.0d0, C, ldc, v, ldv, 0.0d0, work, ldwork))
        call hipCheck(hipStreamSynchronize(stream1))

call hipblasCheck(hipblasdtrmm(hipblasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_modE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, N, &
                                       K, 1.0d0, T, ldt, work, ldwork))

        call hipCheck(hipEventRecord(event2, stream2))
    call hipblasCheck(hipblasdgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, -1.0d0, V, ldv, work, ldwork, 1.0d0, c, ldc))

        ! Restore clobbered section of block column (except diagonal)
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_b95769_2_auto(0, c_null_ptr, c_loc(w), ldw, K, 1, 1, M, c_loc(v), ldv, K, 1, 1, K)

    end subroutine dlarfb_gpu

    ! extracted to HIP C++ file

end module dsyevd_gpu
