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
#ifdef USE_HIP
  use hip
#else
  use cudafor
#endif
#ifdef USE_HIP
  use hipblas
#else
  use cublas
#endif

  contains

    ! dsygst completed in blocks, using 2 ztrsms to solve subblock problem on GPU
    subroutine dsygst_gpu(itype, uplo, N, A, lda, B, ldb, nb)
      use eigsolve_vars
      implicit none
      integer, intent(in)                                   :: itype, N, lda, ldb, nb
      character, intent(in)                                 :: uplo
      real(8), device, dimension(1:ldb, 1:N), intent(in)    :: B
      real(8), device, dimension(1:lda, 1:N)                :: A
      real(8), parameter                                    :: one = 1.d0, half = 0.5d0

      integer                                               :: i, j
      integer                                               :: k, kb, istat

      if (itype .ne. 1 .or. uplo .ne. 'U') then
        print*, "Provided itype/uplo not supported!"
        return
      endif

#ifdef USE_HIP
      istat = hipEventRecord(event2, stream2)
#else
      istat = cudaEventRecord(event2, stream2)
#endif

      do k = 1, N, nb
        kb = min(N-k+1, nb)

#ifdef USE_HIP
        istat = hipblasSetStream(cuHandle, stream1)
#else
        istat = cublasSetStream(cuHandle, stream1)
#endif
        
#ifdef USE_HIP
        istat = hipStreamWaitEvent(stream1, event2, 0)
#else
        istat = cudaStreamWaitEvent(stream1, event2, 0)
#endif
        ! Populate subblock with complete symmetric entries (needed for DTRSM calls)
#ifdef USE_HIP
! extracted to HIP C++ file
! TODO fix arguments
CALL launch_krnl_afb01f_0(0,stream1,kb,A,k)
#else
        !$cuf kernel do(2) <<<*,*, 0, stream1>>>
        do j = k,k+kb-1
          do i = k,k+kb-1
            if (j < i) then
              A(i,j) = A(j,i)
            endif
          end do
        end do
#endif

        ! Solve subblock problem (this version results in fully populated A subblock)
#ifdef USE_HIP
        istat =  hipblasDtrsm_v2(cuHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, HIPBLAS_OP_N, kb, kb, &
                                one, B(k,k), ldb, A(k,k), lda)  
#else
        istat =  cublasdtrsm_v2(cuHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_OP_N, kb, kb, &
                                one, B(k,k), ldb, A(k,k), lda)  
#endif
#ifdef USE_HIP
        istat =  hipblasDtrsm_v2(cuHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, kb, &
                                one, B(k,k), ldb, A(k,k), lda)  
#else
        istat =  cublasdtrsm_v2(cuHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_OP_N, kb, kb, &
                                one, B(k,k), ldb, A(k,k), lda)  
#endif

#ifdef USE_HIP
        istat = hipEventRecord(event1, stream1)
#else
        istat = cudaEventRecord(event1, stream1)
#endif

        if (k + kb .le. N) then
#ifdef USE_HIP
          istat = hipblasSetStream(cuHandle, stream2)
#else
          istat = cublasSetStream(cuHandle, stream2)
#endif
#ifdef USE_HIP
          istat =  hipblasDtrsm_v2(cuHandle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, HIPBLAS_OP_N, kb, N-k-kb+1, one, &
                     B(k, k), ldb, A(k, k+kb), lda) 
#else
          istat =  cublasdtrsm_v2(cuHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_OP_N, kb, N-k-kb+1, one, &
                     B(k, k), ldb, A(k, k+kb), lda) 
#endif

#ifdef USE_HIP
          istat = hipStreamWaitEvent(stream2, event1, 0)
#else
          istat = cudaStreamWaitEvent(stream2, event1, 0)
#endif

          ! Since the A subblock is fully populated, use gemm instead of hemm here
#ifdef USE_HIP
          istat =  hipblasDgemm_v2(cuHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, N-k-kb+1, kb, -half, A(k,k), &
                     lda, B(k, k+kb), ldb, one, A(k, k+kb), lda)
#else
          istat =  cublasdgemm_v2(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, kb, N-k-kb+1, kb, -half, A(k,k), &
                     lda, B(k, k+kb), ldb, one, A(k, k+kb), lda)
#endif
#ifdef USE_HIP
          istat = hipblasDsyr2k_v2(cuHandle, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_T, N-k-kb+1, kb, -one, A(k, k+kb), lda, &
                      B(k, k+kb), ldb, one, A(k+kb, k+kb), lda)
#else
          istat = cublasdsyr2k_v2(cuHandle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, N-k-kb+1, kb, -one, A(k, k+kb), lda, &
                      B(k, k+kb), ldb, one, A(k+kb, k+kb), lda)
#endif

#ifdef USE_HIP
          istat = hipEventRecord(event2, stream2)
#else
          istat = cudaEventRecord(event2, stream2)
#endif

#ifdef USE_HIP
          istat = hipblasDgemm_v2(cuHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, N-k-kb+1, kb, -half, A(k,k), &
                     lda, B(k, k+kb), ldb, one, A(k, k+kb), lda)
#else
          istat = cublasdgemm_v2(cuHandle, CUBLAS_OP_N, CUBLAS_OP_N, kb, N-k-kb+1, kb, -half, A(k,k), &
                     lda, B(k, k+kb), ldb, one, A(k, k+kb), lda)
#endif

#ifdef USE_HIP
          istat = hipblasDtrsm_v2(cuHandle, HIPBLAS_SIDE_RIGHT, HIPBLAS_FILL_modE_UPPER, HIPBLAS_OP_N, HIPBLAS_OP_N, kb, N-k-kb+1, one, &
                     B(k+kb, k+kb), ldb, A(k, k+kb), lda) 
#else
          istat = cublasdtrsm_v2(cuHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_OP_N, kb, N-k-kb+1, one, &
                     B(k+kb, k+kb), ldb, A(k, k+kb), lda) 
#endif
        end if

      end do

    end subroutine dsygst_gpu

end module dsygst_gpu
