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
module dsytd2_gpu_kernels

    interface

        subroutine launch_dsytd2_gpu(grid, &
                                     block, &
                                     sharedMem, &
                                     stream, &
                                     lda, &
                                     a, &
                                     a_n1, &
                                     a_n2, &
                                     a_lb1, &
                                     a_lb2, &
                                     tau, &
                                     tau_n1, &
                                     tau_lb1, &
                                     d, &
                                     d_n1, &
                                     d_lb1, &
                                     e, &
                                     e_n1, &
                                     e_lb1, &
                                     n) bind(c, name="launch_dsytd2_gpu")
            use iso_c_binding
            use hipfort
            implicit none
            type(dim3),value, intent(IN) :: grid
            type(dim3), value, intent(IN) :: block
            integer(c_int), value, intent(IN) :: sharedMem
            type(c_ptr), value, intent(IN) :: stream
            INTEGER, value :: lda
            type(c_ptr), value :: a
            integer(c_int), value, intent(IN) :: a_n1
            integer(c_int), value, intent(IN) :: a_n2
            integer(c_int), value, intent(IN) :: a_lb1
            integer(c_int), value, intent(IN) :: a_lb2
            type(c_ptr), value :: tau
            integer(c_int), value, intent(IN) :: tau_n1
            integer(c_int), value, intent(IN) :: tau_lb1
            type(c_ptr), value :: d
            integer(c_int), value, intent(IN) :: d_n1
            integer(c_int), value, intent(IN) :: d_lb1
            type(c_ptr), value :: e
            integer(c_int), value, intent(IN) :: e_n1
            integer(c_int), value, intent(IN) :: e_lb1
            INTEGER, value :: n
        end subroutine

    end interface

end module dsytd2_gpu_kernels
module dsymv_gpu_kernels
    interface

        subroutine launch_dsymv_gpu(grid, &
                                    block, &
                                    sharedMem, &
                                    stream, &
                                    n, &
                                    a, &
                                    a_n1, &
                                    a_lb1, &
                                    a_lb2, &
                                    x, &
                                    x_n1, &
                                    x_lb1, &
                                    y, &
                                    y_n1, &
                                    y_lb1) bind(c, name="launch_dsymv_gpu")
            use iso_c_binding
            use hipfort
            implicit none
            type(dim3), intent(IN) :: grid
            type(dim3), intent(IN) :: block
            integer(c_int), value, intent(IN) :: sharedMem
            type(c_ptr), value, intent(IN) :: stream
            INTEGER, value :: n
            type(c_ptr), value :: a
            integer(c_int), value, intent(IN) :: a_n1
            integer(c_int), value, intent(IN) :: a_lb1
            integer(c_int), value, intent(IN) :: a_lb2
            type(c_ptr), value :: x
            integer(c_int), value, intent(IN) :: x_n1
            integer(c_int), value, intent(IN) :: x_lb1
            type(c_ptr), value :: y
            integer(c_int), value, intent(IN) :: y_n1
            integer(c_int), value, intent(IN) :: y_lb1
        end subroutine

    end interface

end module dsymv_gpu_kernels
module dsytrd_gpu
    use dsytrd_gpu_kernels
    use hipfort
    use iso_c_binding
    use hipfort_hipblas
    use utils

contains

    subroutine dsytrd_gpu_h(uplo, N, A, lda, d, e, tau, work, lwork, nb)
        use dsytrd_gpu_kernels
        use eigsolve_vars
        use utils
        use dsytd2_gpu_kernels
        implicit none
        character                                 :: uplo
        integer                                   :: N, lda, lwork, nb, nx, ldwork, istat,nb1
        integer                                   :: i, j, k, kk
        real(8), target, dimension(1:N)           :: d
        real(8), dimension(1:N)                   :: d_h
        real(8), target, dimension(1:N - 1)       :: e
        real(8), dimension(1:N - 1)               :: e_h
        real(8), target, dimension(1:lwork)       :: work
        real(8), target, dimension(1:lda, 1:N)    :: A
        real(8), dimension(1:lda, 1:N)            :: A_h
        real(8), target, dimension(1:N - 1)       :: tau
        real(8), dimension(1:N - 1)               :: tau_h

        real(8), parameter                        :: one = 1.0_8
        type(dim3)                                :: threads, grid

        if (uplo .ne. 'U') then
            print *, "Provided uplo type not supported!"
            return
        endif

        if (lwork < (nb + 2)*N .and. N > nb) then
            write (*, *) "Provided work array must be sized (nb+2)*N or greater!"
            return
        endif

        ldwork = N

        call hipblasCheck(hipblasSetStream(hipblasHandle, stream1))

        kk = N - ((N - 16)/nb)*nb
        print*,"kk = ",kk, "N = ",N
        k = N + 1
        do i = N - nb + 1, kk + 1, -nb
            ! Reduce columns i:i+nb-1 to tridiagonal form
            call dlatrd_gpu(uplo, i + nb - 1, nb, A, lda, e, tau, work, ldwork)

            ! Update trailing submatrix
          call hipblasCheck(hipblasdsyr2k_m(hipblasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, (i - 1), nb, -one, A(1, i), lda, &
                                              work, ldwork, one, A, lda))

            k = k - nb

        end do

        ! ! Finish any remaining columns to get final 16x16 block
        ! nb1 = k - 16 - 1
        ! i = k - nb1

        ! if (nb1 > 0) then
        !     ! Reduce columns i:i+nb-1 to tridiagonal form
        !     call dlatrd_gpu(uplo, i + nb1 - 1, nb1, A, lda, e, tau, work, ldwork)

        !     ! Update trailing submatrix
        !     call hipblasCheck(hipblasdsyr2k_m(hipblasHandle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, (i - 1), nb1, -one, &
        !                                       A(1, i), lda, work, ldwork, one, A, lda))

        ! endif

        ! Final block
        ! call hipCheck(hipMemcpy(A_h,A,N*N,hipMemcpyDeviceToHost))
        ! call print_matrix(A_h)

        threads = dim3(16, 16, 1)
        
        grid = dim3(1, 1, 1)
        CALL launch_dsytd2_gpu(grid, threads, 16*16*8*2 + 8*3, c_null_ptr, lda, c_loc(A), lda, N, 1, 1, c_loc(tau), N - 1, 1,&
            c_loc(d), N, 1, c_loc(e), N - 1, 1, min(16, N))

        ! call hipCheck(hipMemcpy(A_h,A,N*N,hipMemcpyDeviceToHost))
        ! call print_matrix(A_h)

        ! call hipCheck(hipMemcpy(d_h,d,N,hipMemcpyDeviceToHost))
        ! call print_vector(d_h)

        ! call hipCheck(hipMemcpy(e_h,e,N-1,hipMemcpyDeviceToHost))
        ! call print_vector(e_h)

        ! call hipCheck(hipMemcpy(tau_h,tau,N-1,hipMemcpyDeviceToHost))
        ! call print_vector(tau_h)
        ! Copy superdiagonal back into A, store diagonal in d
        ! extracted to HIP C++ file
        ! TODO(gpufort) fix arguments
        !CALL launch_krnl_2b8e8f_0_auto(0, c_null_ptr, n, c_loc(d), N, 1, c_loc(A), lda, N, 1, 1)

    end subroutine dsytrd_gpu_h

    subroutine dlatrd_gpu(uplo, N, nb, A, lda, e, tau, W, ldw)
        use dsytrd_gpu_kernels
        use eigsolve_vars

        use dsymv_gpu_kernels
        implicit none
        character                                  :: uplo
        integer                                    :: N, nb, lda, ldw, istat
        integer                                    :: i, j, k, iw
        type(dim3)                                    :: threads
        real(8), target, dimension(1:lda, 1:N)     :: A
        real(8), dimension(1:lda, 1:N)             :: A_h
        real(8), target, dimension(1:ldw, 1:nb)    :: W
        real(8), dimension(1:ldw, 1:nb)            :: W_h
        real(8), target, dimension(N - 1)            :: tau
        real(8), target, dimension(N - 1)            :: e
        real(8), dimension(N - 1)                  :: tau_h
        real(8),  dimension(N - 1)                 :: e_h


        real(8), parameter                         :: one = 1.0d0, zero = 0.0d0, half = 0.5d0

        type(dim3)                                 :: threads2D, blocks2D

        if (uplo .ne. 'U') then
            print *, "Provided uplo type not supported!"
            return
        endif

        !threads2D = dim3(32, 8, 1)
        threads2D = dim3(16,16,1)
        threads = dim3(256, 1, 1)

        if (N <= 0) return

        ! Complete first iteration outside loop
        if (N > 1) then
            print*,"first iter"
            iw = nb
            ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
            call hipCheck(hipMemcpy(A_h, A, N*N,hipMemcpyDeviceToHost))
            call print_vector(A_h(:,N))
            CALL launch_dlarfg_kernel_m(dim3(1, 1, 1), threads, 16, c_null_ptr, N - 1, tau(N - 1), e(N - 1), A(1, N))
            call hipCheck(hipMemcpy(A_h, A, N*N,hipMemcpyDeviceToHost))
            call print_vector(A_h(:,N))
            call hipCheck(hipMemcpy(e_h, e, N-1, hipMemcpyDeviceToHost))
            call hipCheck(hipMemcpy(tau_h, tau, N-1, hipMemcpyDeviceToHost))
            print*,e_h(N-1),tau_h(N-1)
            
            ! extracted to HIP C++ file
            ! TODO(gpufort) fix arguments
            CALL launch_krnl_37a79c_1_auto(0, c_null_ptr, c_loc(w), ldw, n, iw)

            !blocks2D = dim3(10, ceiling(real(N - 1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
            !CALL launch_dsymv_gpu_m(blocks2D, threads2D, (32+1)*32*8 + 32*8, c_null_ptr, N - 1, lda, A, A(1, N), W(1, iw))
            blocks2D = dim3(10, ceiling(real(N - 1)/16), 1) 
            CALL launch_dsymv_gpu_m(blocks2D, threads2D, (16+1)*16*8 + 16*8 + 16*16*8, c_null_ptr, N - 1, lda, A, A(1, N), W(1, iw)) 
            call hipCheck(hipMemcpy(W_h, W, ldw*nb, hipMemcpyDeviceToHost))
            call print_vector(W_h(:,iw))
            stop

            CALL launch_finish_W_col_kernel_m(dim3(1, 1, 1), threads, 8, c_null_ptr, N - 1, tau(N - 1), A(1, N), W(1, iw))
        endif

        do i = N - 1, N - nb + 1, -1
            iw = i - N + nb

            blocks2D = dim3(ceiling(real(max(i, N - i))/32), ceiling(real(N - i)/8), 1)
            CALL launch_dsyr2_mv_dlarfg_kernel_m(blocks2D, threads2D, 20, c_null_ptr, i, N - i, lda, ldw, ldw, A(1, i + 1), &
                                                 W(1, iw + 1), A(1, i), W(1, iw), e(i - 1), tau(i - 1), finished)

            if (i > 1) then
                ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)

                blocks2D = dim3(10, ceiling(real(i - 1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
                CALL launch_dsymv_gpu_m(blocks2D, threads2D, (32+1)*32*8 + 32*8, c_null_ptr, i - 1, lda, A, A(1, i), W(1, iw))

                blocks2D = dim3(ceiling(real(i - 1)/32), ceiling(real(2*(n - i))/8), 1)
              CALL launch_stacked_dgemv_T_m(blocks2D, threads2D, 0, c_null_ptr, n - i, i - 1, lda, ldw, A(1, i + 1), W(1, iw + 1), &
                                              A(1, i), W(i + 1, iw), W(i + 1, iw + 1))
                CALL launch_stacked_dgemv_N_finish_W_m(blocks2D, threads2D, 12, c_null_ptr, i - 1, n - i, lda, ldw, A(1, i + 1), &
                                              W(1, iw + 1), W(i + 1, iw), W(i + 1, iw + 1), W(1, iw), tau(i - 1), A(1, i), finished)

            end if
        end do
    end subroutine dlatrd_gpu

    subroutine launch_dlarfg_kernel_m(grid, block, sharedMem, stream, n, tau, e, x)
        use iso_c_binding
        use hipfort
        implicit none
        type(dim3), intent(IN) :: grid
        type(dim3), intent(IN) :: block
        integer(c_int), intent(IN) :: sharedMem
        type(c_ptr), value, intent(IN) :: stream
        INTEGER, value :: n
        real(8), target, dimension(1) :: tau
        real(8), target, dimension(1) :: e
        real(8), target, dimension(n) :: x

        call launch_dlarfg_kernel(grid, block, sharedMem, stream, n, c_loc(tau), c_loc(e), c_loc(x), n, 1)
    end subroutine launch_dlarfg_kernel_m

    subroutine launch_dsymv_gpu_m(grid, block, sharedMem, stream, n, lda, a, x, y)
        use iso_c_binding
        use hipfort
        use dsymv_gpu_kernels
        implicit none
        type(dim3), intent(IN) :: grid
        type(dim3), intent(IN) :: block
        integer(c_int), intent(IN) :: sharedMem
        type(c_ptr), value, intent(IN) :: stream
        INTEGER, value :: n, lda
        real(8), target, dimension(lda, N), intent(in)    :: a
        real(8), target, dimension(N), intent(in)         :: x
        real(8), target, dimension(N)                     :: y

        call launch_dsymv_gpu(grid, block, sharedMem, stream, n, c_loc(a), lda, 1, 1, c_loc(x), n, 1, c_loc(y), n, 1)
    end subroutine launch_dsymv_gpu_m

    subroutine launch_finish_W_col_kernel_m(grid, block, sharedMem, stream, n, tau, x, y)
        use iso_c_binding
        use hipfort
        implicit none
        type(dim3), intent(IN) :: grid
        type(dim3), intent(IN) :: block
        integer(c_int), intent(IN) :: sharedMem
        type(c_ptr), value, intent(IN) :: stream
        INTEGER, value :: n
        real(8), target, dimension(1) :: tau
        real(8), target, dimension(N), intent(in)    :: x
        real(8), target, dimension(N)               :: y

        call launch_finish_W_col_kernel(grid, block, sharedMem, stream, n, c_loc(tau), c_loc(x), n, 1, c_loc(y), n, 1)
    end subroutine launch_finish_W_col_kernel_m
    subroutine launch_dsyr2_mv_dlarfg_kernel_m(grid, block, sharedMem, stream, n, m, ldv, ldw, ldw2, v, w, w2, x, e, tau, finished)
        use iso_c_binding
        use hipfort
        use dsytrd_gpu_kernels
        implicit none
        type(dim3), intent(IN) :: grid
        type(dim3), intent(IN) :: block
        integer(c_int), intent(IN) :: sharedMem
        type(c_ptr), value, intent(IN) :: stream
        INTEGER, value :: n, m, ldv, ldw, ldw2
        real(8), target, dimension(1:ldv, 1:M), intent(in)  :: V
        real(8), target, dimension(1:ldw, 1:M), intent(in)  :: W
        real(8), target, dimension(1:ldw2, 2)              :: W2
        real(8), target, dimension(1:N)                   :: x
        real(8), target, dimension(1)               :: tau
        real(8), target, dimension(1)               :: e
        integer, target, dimension(1)               :: finished

        call launch_dsyr2_mv_dlarfg_kernel(grid,block,sharedMem,stream,n,m,c_loc(v),ldv,1,1,c_loc(w),ldw,1,1,c_loc(w2),ldw2,1,1,&
                                           c_loc(x), n, 1, c_loc(e), c_loc(tau), c_loc(finished))
    end subroutine launch_dsyr2_mv_dlarfg_kernel_m

    subroutine launch_stacked_dgemv_T_m(grid, block, sharedMem, stream, m, n, ldv, ldw, v, w, x, z1, z2)
        use iso_c_binding
        use hipfort
        implicit none
        type(dim3), intent(IN) :: grid
        type(dim3), intent(IN) :: block
        integer(c_int), intent(IN) :: sharedMem
        type(c_ptr), value, intent(IN) :: stream
        INTEGER, value :: n, m, ldv, ldw
        real(8), target, dimension(ldv, M), intent(in)  :: V
        real(8), target, dimension(ldw, M), intent(in)  :: W
        real(8), target, dimension(N), intent(in)       :: x
        real(8), target, dimension(M)                   :: z1, z2

        call launch_stacked_dgemv_T(grid,block,sharedMem,stream,m,n,c_loc(v),ldv,1,1,c_loc(w),ldw,1,1,c_loc(x),n,1,c_loc(z1),m,&
                                    1, c_loc(z2), m, 1)
    end subroutine launch_stacked_dgemv_T_m

    subroutine launch_stacked_dgemv_N_finish_W_m(grid, block, sharedMem, stream, m, n, ldv, ldw, v, w, z1, z2, y, tau, x, finished)
        use iso_c_binding
        use hipfort
        implicit none
        type(dim3), intent(IN) :: grid
        type(dim3), intent(IN) :: block
        integer(c_int), intent(IN) :: sharedMem
        type(c_ptr), value, intent(IN) :: stream
        INTEGER, value :: n, m, ldv, ldw
        real(8), target, dimension(ldv, M), intent(in)  :: V
        real(8), target, dimension(ldw, M), intent(in)  :: W
        real(8), target, dimension(N), intent(in)        :: z1, z2
        real(8), target, dimension(M)                   :: y
        real(8), target, dimension(M), intent(in)       :: x
        real(8), target, dimension(1) :: tau
        integer, target, dimension(1)               :: finished

        call launch_stacked_dgemv_N_finish_W(grid,block,sharedMem,stream,m,n,c_loc(v),ldv,1,1,c_loc(w),ldw,1,1,c_loc(z1),n,1,&
                                             c_loc(z2), n, 1, c_loc(y), m, 1, c_loc(tau), c_loc(x), m, 1, c_loc(finished))
    end subroutine launch_stacked_dgemv_N_finish_W_m

    function hipblasdsyr2k_m(handle, uplo, transA, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        use iso_c_binding
        use hipfort_hipblas_enums
        use hipfort_hipblas
        implicit none
        integer(kind(HIPBLAS_STATUS_SUCCESS)) :: hipblasDsyr2k_m
        type(c_ptr), value :: handle
        integer(kind(HIPBLAS_FILL_MODE_UPPER)), value :: uplo
        integer(kind(HIPBLAS_OP_N)), value :: transA
        integer(c_int), value :: n
        integer(c_int), value :: k
        real(c_double) :: alpha
        real(8), target, dimension(n, k) :: A
        integer(c_int), value :: lda
        real(8), target, dimension(n, k) :: B
        integer(c_int), value :: ldb
        real(c_double) :: beta
        real(8), target, dimension(n, n) :: C
        integer(c_int), value :: ldc

        hipblasdsyr2k_m = hipblasdsyr2k(handle, uplo, transA, n, k, alpha, c_loc(A), lda, c_loc(B), ldb, beta, c_loc(C), ldc)
    end function hipblasdsyr2k_m
end module dsytrd_gpu
