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
    use hip
    use iso_c_binding
    use iso_c_binding_ext
    use hipblas

    implicit none

contains

    ! Custom dsyevd routine
    subroutine dsyevd_gpu_h(jobz, uplo, il, iu, N, A, lda, Z, ldz, w, work, lwork, &
                            work_h, lwork_h, iwork_h, liwork_h, Z_h, ldz_h, w_h, info)
        use dsyevd_gpu_kernels
        use dsytrd_gpu

        use eigsolve_vars
        implicit none
        character                                   :: uplo, jobz
        integer                                     :: N, NZ, lda, lwork, istat, info
        integer                                     :: lwork_h, liwork_h, ldz_h
        integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu
        type(c_ptr), value :: work
        integer(c_int) :: work_n1, work_lb1

        type(c_ptr), value :: work_h
        integer(c_int) :: work_h_n1, work_h_lb1

        type(c_ptr), value :: iwork_h
        integer(c_int) :: iwork_h_n1, iwork_h_lb1

        type(c_ptr), value :: A
        integer(c_int) :: A_n1, A_n2, A_lb1, A_lb2
        type(c_ptr), value :: Z
        integer(c_int) :: Z_n1, Z_n2, Z_lb1, Z_lb2
        type(c_ptr), value :: Z_h
        integer(c_int) :: Z_h_n1, Z_h_n2, Z_h_lb1, Z_h_lb2
        type(c_ptr), value :: w
        integer(c_int) :: w_n1, w_lb1
        type(c_ptr), value :: w_h
        integer(c_int) :: w_h_n1, w_h_lb1

        integer                                     :: inde, indtau, indwrk, llwork, llwork_h, indwk2, indwk3, llwrk2
        real(8), parameter                          :: one = 1.0_8

        type(dim3) :: blocks, threads
        work_n1 = lwork
        work_lb1 = 1
        work_h_n1 = lwork_h
        work_h_lb1 = 1
        iwork_h_n1 = liwork_h
        iwork_h_lb1 = 1
        A_n1 = lda
        A_n2 = N
        A_lb1 = 1
        A_lb2 = 1
        Z_n1 = lda
        Z_n2 = N
        Z_lb1 = 1
        Z_lb2 = 1
        Z_h_n1 = ldz_h
        Z_h_n2 = N
        Z_h_lb1 = 1
        Z_h_lb2 = 1
        w_n1 = N
        w_lb1 = 1
        w_h_n1 = N
        w_h_lb1 = 1
        if (uplo .ne. 'U' .or. jobz .ne. 'V') then
            print *, "Provided itype/uplo not supported!"
            return
        endif

        nb1 = 32 ! Blocksize for tridiagonalization
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
        call dsytrd_gpu_h('U', N, A, lda, w, inc_c_ptr(work, 1_8*8*inde), inc_c_ptr(work, 1_8*8*indtau), &
                          inc_c_ptr(work, 1_8*8*indwrk), llwork, nb1)

        ! Copy diagonal and superdiagonal to CPU
#undef _idx_w
#define _idx_w(a) ((a-(w_lb1)))
#undef _idx_w_h
#define _idx_w_h(a) ((a-(w_h_lb1)))
#undef _idx_work
#define _idx_work(a) ((a-(work_lb1)))
#undef _idx_work_h
#define _idx_work_h(a) ((a-(work_h_lb1)))
        !w_h(1:N) = w(1:N)
        istat = hipMemcpy(w_h, w, 1_8*(8)*(N), hipMemcpyHostToDevice)

        !work_h(inde:inde+N-1) = work(inde:inde+N-1)
        istat = hipMemcpy(inc_c_ptr(work_h, 1_8*8*_idx_work_h(inde)), inc_c_ptr(work, 1_8*_idx_work(inde)*8), &
                          1_8*(8)*((inde + N - 1) - (inde) + 1), hipMemcpyHostToDevice)

        ! Restore lower triangular of A (works if called from zhegvd only!)
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_e26a05_0_auto(0, c_null_ptr, z, z_n1, z_n2, z_lb1, z_lb2, n, a, a_n1, a_n2, a_lb1, a_lb2)

        ! Call DSTEDC to get eigenvalues/vectors of tridiagonal A on CPU
        call dstedc('I', N, w_h, inc_c_ptr(work_h, 1_8*8*_idx_work_h(inde)), Z_h, ldz_h, &
                    inc_c_ptr(work_h, 1_8*8*_idx_work_h(indwrk)), llwork_h, iwork_h, liwork_h, istat)
        if (istat /= 0) then
            write (*, *) "dsyevd_gpu error: dstedc failed!"
            info = -1
            return
        endif

        ! Copy eigenvectors and eigenvalues to GPU
#undef _idx_Z
#define _idx_Z(a,b) ((a-(Z_lb1))+Z_n1*(b-(Z_lb2)))
#undef _idx_Z_h
#define _idx_Z_h(a,b) ((a-(Z_h_lb1))+Z_h_n1*(b-(Z_h_lb2)))
#undef _idx_w
#define _idx_w(a) ((a-(w_lb1)))
        !istat = cudaMemcpy2D(Z(1, 1), ldz, Z_h, ldz_h, N, NZ)
        istat = hipMemcpy2D(inc_c_ptr(Z, _idx_Z(1,1)*8*1_8), ldz*8*1_8,&
        inc_c_ptr(Z_h, _idx_Z_h(1,1)*8*1_8), ldz_h*8*1_8,&
        N*8*1_8, NZ*1_8, hipMemcpyHostToDevice)
        !w(1:N) = w_h(1:N)
        istat = hipMemcpy(inc_c_ptr(w, _idx_w(1)*(8)*1_8),&
         inc_c_ptr(w_h, _idx_w_h(1)*8*1_8),&
          1_8*(8)*(N), hipMemcpyHostToDevice)

      !! Call DORMTR to rotate eigenvectors to obtain result for original A matrix
      !! JR Note: Eventual function calls from DORMTR called directly here with associated indexing changes

        istat = hipEventRecord(event2, stream2)

        k = N - 1

        do i = 1, k, nb2
            ib = min(nb2, k - i + 1)

            ! Form block reflector T in stream 1
            call dlarft_gpu(i + ib - 1, ib, inc_c_ptr(A, lda*(2 + i - 1 - 1)*8*1_8), lda, &
            inc_c_ptr(work, _idx_w(indtau + i - 1)*8*1_8), &
            inc_c_ptr(work, _idx_w(indwrk)*8*1_8), ldt,&
            inc_c_ptr(work, _idx_w(indwk2)*8*1_8), ldt)

            mi = i + ib - 1
            ! Apply reflector to eigenvectors in stream 2
            call dlarfb_gpu(mi, NZ, ib, inc_c_ptr(A, lda*(2 + i - 1 - 1)*8*1_8), lda,&
            inc_c_ptr(work, _idx_w(indwrk)*8*1_8), ldt, Z, ldz, &
            inc_c_ptr(work, _idx_w(indwk3)*8*1_8), N,&
            inc_c_ptr(work, _idx_w(indwk2)*8*1_8), ldt)
        end do


    end subroutine dsyevd_gpu_h

    subroutine dlarft_gpu(N, K, V, ldv, tau, T, ldt, W, ldw)
        use dsyevd_gpu_kernels
        use hipblas

        use eigsolve_vars
        implicit none
        integer                               :: N, K, ldv, ldt, ldw
        type(c_ptr), value :: V
        integer(c_int) :: V_n1, V_n2, V_lb1, V_lb2
        type(c_ptr), value :: tau
        integer(c_int) :: tau_n1, tau_lb1
        type(c_ptr), value :: T
        integer(c_int) :: T_n1, T_n2, T_lb1, T_lb2
        type(c_ptr), value :: W
        integer(c_int) :: W_n1, W_n2, W_lb1, W_lb2

        integer                               :: i, j,istat1
        type(dim3)                            :: threads
        V_n1 = ldv
        V_n2 = K
        V_lb1 = 1
        V_lb2 = 1
        tau_n1 = K
        tau_lb1 = 1
        T_n1 = ldt
        T_n2 = K
        T_lb1 = 1
        T_lb2 = 1
        W_n1 = ldw
        W_n2 = K
        W_lb1 = 1
        W_lb2 = 1
        ! Prepare lower triangular part of block column for dsyrk call.
        ! Requires zeros in lower triangular portion and ones on diagonal.
        ! Store existing entries (excluding diagonal) in W
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_b1f342_1_auto(0, stream1, w, w_n1, w_n2, w_lb1, w_lb2, n, v, v_n1, v_n2, v_lb1, v_lb2, k)

        istat1 = hipEventRecord(event1, stream1)
        istat1 = hipStreamWaitEvent(stream1, event2, 0)

        ! Form preliminary T matrix
        istat1 = hipblasdsyrk(hipblasHandle, HIPBLAS_FILL_modE_LOWER, HIPBLAS_OP_T, K, N, 1.0_8, V, ldv, 0.0_8, T, ldt)

        ! Finish forming T
        threads = dim3(64, 16, 1)
     CALL launch_finish_t_block_kernel(dim3(1, 1, 1),threads, 0, stream1,n, ldt, T, t_n1, t_n2, t_lb1, t_lb2, tau,tau_n1,tau_lb1)
    end subroutine dlarft_gpu

    subroutine dlarfb_gpu(M, N, K, V, ldv, T, ldt, C, ldc, work, ldwork, W, ldw)
        use dsyevd_gpu_kernels
        use hipblas

        use eigsolve_vars
        implicit none
        integer                               :: M, N, K, ldv, ldt, ldc, ldw, ldwork, istat
        integer                               :: i, j
        type(c_ptr), value :: V
        integer(c_int) :: V_n1, V_n2, V_lb1 = 1, V_lb2 = 1
        type(c_ptr), value :: T
        integer(c_int) :: T_n1, T_n2, T_lb1 = 1, T_lb2 = 1
        type(c_ptr), value :: W
        integer(c_int) :: W_n1, W_n2, W_lb1 = 1, W_lb2 = 1
        type(c_ptr), value :: C
        integer(c_int) :: C_n1, C_n2, C_lb1 = 1, C_lb2 = 1
        type(c_ptr), value :: work
        integer(c_int) :: work_n1, work_n2, work_lb1 = 1, work_lb2 = 1
        V_n1 = ldv
        V_n2 = K
        T_n1 = ldt
        T_n2 = K
        W_n1 = ldw
        W_n2 = K
        C_n1 = ldc
        C_n2 = N
        work_n1 = ldwork
        work_n2 = K
        istat = hipStreamWaitEvent(stream2, event1, 0)
        istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, K, M, 1.0d0, C, ldc, v, ldv, 0.0d0, work, ldwork)
        istat = hipStreamSynchronize(stream1)

      istat = hipblasdtrmm(hipblasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_modE_LOWER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, N,&
       K, 1.0d0, T, ldt, work, ldwork)

        istat = hipEventRecord(event2, stream2)
        istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_T, M, N, K, -1.0d0, V, ldv, work, ldwork, 1.0d0, c, ldc)

        ! Restore clobbered section of block column (except diagonal)
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        CALL launch_krnl_b95769_2_auto(0, c_null_ptr, w, w_n1, w_n2, w_lb1, w_lb2, m, v, v_n1, v_n2, v_lb1, v_lb2, k)

    end subroutine dlarfb_gpu

    ! extracted to HIP C++ file

end module dsyevd_gpu
