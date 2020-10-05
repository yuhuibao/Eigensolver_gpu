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

module dsygst_gpu
    use hip
    use iso_c_binding
    use iso_c_binding_ext
    use hipblas

contains

    ! dsygst completed in blocks, using 2 ztrsms to solve subblock problem on GPU
    subroutine dsygst_gpu_h(itype, uplo, N, A, lda, B, ldb, nb)
        use dsygst_gpu_kernels
        use eigsolve_vars

        implicit none
        integer, intent(in)                                   :: itype, N, lda, ldb, nb
        character, intent(in)                                 :: uplo
        type(c_ptr), value :: B
        integer(c_int) :: B_n1, B_n2, B_lb1, B_lb2
        type(c_ptr), value :: A
        integer(c_int) :: A_n1, A_n2, A_lb1, A_lb2

        real(8), parameter                                    :: one = 1.d0, half = 0.5d0

        integer                                               :: i, j
        integer                                               :: k, kb, istat

        B_n1 = ldb
        B_n2 = n
        B_lb1 = 1
        B_lb2 = 1
        a_n2 = n
        a_lb1 = 1
        a_lb2 = 1
        A_n1 = lda
        if (itype .ne. 1 .or. uplo .ne. 'U') then
            print *, "Provided itype/uplo not supported!"
            return
        endif

        istat = hipEventRecord(event2, stream2)

        do k = 1, N, nb
            kb = min(N - k + 1, nb)

            istat = hipblasSetStream(hipblasHandle, stream1)

            istat = hipStreamWaitEvent(stream1, event2, 0)
            ! Populate subblock with complete symmetric entries (needed for DTRSM calls)
            ! extracted to HIP C++ file
            CALL launch_krnl_afb01f_0_auto(0, stream1, kb, a, a_n1, a_n2, a_lb1, a_lb2, k)

            ! Solve subblock problem (this version results in fully populated A subblock)
            istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, HIPBLAS_OP_N, kb, kb, &
            one, inc_c_ptr(B, (ldb*(k - 1) + k - 1)*8*1_8), ldb - (k - 1), inc_c_ptr(A, (lda*(k - 1) + k - 1)*8*1_8), lda - (k - 1))
            istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, kb, &
            one, inc_c_ptr(B, (ldb*(k - 1) + k - 1)*8*1_8), ldb - (k - 1), inc_c_ptr(A, (lda*(k - 1) + k - 1)*8*1_8), lda - (k - 1))

            istat = hipEventRecord(event1, stream1)

            if (k + kb .le. N) then
                istat = hipblasSetStream(hipblasHandle, stream2)
                istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, HIPBLAS_OP_N, &
             kb, (N - k - kb + 1), one, inc_c_ptr(B,(ldb*(k-1)+k-1)*8*1_8), ldb-(k-1) ,inc_c_ptr(A,(lda*(k+kb-1)+k-1)*8*1_8),&
                                     lda - (k - 1))

                istat = hipStreamWaitEvent(stream2, event1, 0)

                ! Since the A subblock is fully populated, use gemm instead of hemm here
                istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, (N - k - kb + 1), kb, &
                        -half, inc_c_ptr(A, (lda*(k - 1) + k - 1)*8*1_8), lda - (k - 1), &
                        inc_c_ptr(B, (ldb*(k + kb - 1) + k - 1)*8*1_8), &
                        ldb - (k - 1),one, inc_c_ptr(A, (lda*(k + kb - 1) + k - 1)*8*1_8), lda - (k - 1))
                istat = hipblasdsyr2k(hipblasHandle, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, (N - k - kb + 1), kb, -one, &
                        inc_c_ptr(A, (lda*(k + kb - 1) + k - 1)*8*1_8), lda - (k - 1), &
                        inc_c_ptr(B, (ldb*(k + kb - 1) + k - 1)*8*1_8), ldb - (k - 1), &
                        one, inc_c_ptr(A, (lda*(k + kb - 1) + k + kb - 1)*8*1_8), lda - (k + kb - 1))

                istat = hipEventRecord(event2, stream2)

                istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, (N - k - kb + 1), kb, -half, &
                        inc_c_ptr(A, (lda*(k - 1) + k - 1)*8*1_8), lda - (k - 1), &
                        inc_c_ptr(B, (ldb*(k + kb - 1) + k - 1)*8*1_8),ldb - (k - 1), &
                        one,inc_c_ptr(A, (lda*(k + kb - 1) + k - 1)*8*1_8), lda - (k - 1))

                istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, &
                                     N - k - kb + 1, one,inc_c_ptr(B, (ldb*(k + kb - 1) + k + kb - 1)*8*1_8), ldb-(k + kb - 1) ,&
                                     inc_c_ptr(A, (lda*(k + kb - 1) + k - 1)*8*1_8), lda - (k - 1))

            end if

        end do

    end subroutine dsygst_gpu_h

end module dsygst_gpu
