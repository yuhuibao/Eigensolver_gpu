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
    use hipfort
    use iso_c_binding
    use hipfort_hipblas
    use hipfort_check

contains

    ! dsygst completed in blocks, using 2 ztrsms to solve subblock problem on GPU
    subroutine dsygst_gpu_h(itype, uplo, N, A, lda, B, ldb, nb)
        use dsygst_gpu_kernels
        use eigsolve_vars
        use hipfort_check

        implicit none
        integer, intent(in)                                   :: itype, N, lda, ldb, nb
        character, intent(in)                                 :: uplo
        real(8), target, dimension(1:ldb, 1:N), intent(in)    :: B
        real(8), target, dimension(1:lda, 1:N)                :: A
        real(8), parameter                                    :: one = 1.d0, half = 0.5d0

        integer                                               :: i, j
        integer                                               :: k, kb, istat

        if (itype .ne. 1 .or. uplo .ne. 'U') then
            print *, "Provided itype/uplo not supported!"
            return
        endif
        call hipCheck(hipEventRecord(event2, stream2))

        do k = 1, N, nb
            kb = min(N - k + 1, nb)

            istat = hipblasSetStream(hipblasHandle, stream1)

            call hipCheck(hipStreamWaitEvent(stream1, event2, 0))
            ! Populate subblock with complete symmetric entries (needed for DTRSM calls)
            ! extracted to HIP C++ file
            CALL launch_krnl_afb01f_0_auto(0, stream1, kb, c_loc(a), lda, N, 1, 1, k)

            ! Solve subblock problem (this version results in fully populated A subblock)
      istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, kb, kb, &
                                 one, B(k, k), ldb, A(k, k), lda)
     istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, kb, kb, &
                                 one, B(k, k), ldb, A(k, k), lda)

            istat = hipEventRecord(event1, stream1)

            !if (k + kb .le. N) then
            istat = hipblasSetStream(hipblasHandle, stream2)
            istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, HIPBLAS_DIAG_NON_UNIT, &
                                 kb, (N - k - kb + 1), one, B(k, k), ldb, A(k, k + kb), lda)

            istat = hipStreamWaitEvent(stream2, event1, 0)

            ! Since the A subblock is fully populated, use gemm instead of hemm here
            istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, (N - k - kb + 1), kb, -half, A(k, k), &
                                 lda, B(k, k + kb), ldb, one, A(k, k + kb), lda)
        istat = hipblasdsyr2k(hipblasHandle, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, (N - k - kb + 1), kb, -one, A(k, k + kb), lda, &
                                  B(k, k + kb), ldb, one, A(k + kb, k + kb), lda)

            istat = hipEventRecord(event2, stream2)
            istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, (N - k - kb + 1), kb, -half, A(k, k), &
                                 lda, B(k, k + kb), ldb, one, A(k, k + kb), lda)

         istat = hipblasdtrsm(hipblasHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, HIPBLAS_DIAG_NON_UNIT, kb, &
                                 N - k - kb + 1, one, B(k + kb, k + kb), ldb, A(k, k + kb), lda)

            !end if

        end do

    end subroutine dsygst_gpu_h

end module dsygst_gpu
