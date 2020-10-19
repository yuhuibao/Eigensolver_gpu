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
module zhetd2_gpu_kernels
    use hip
    implicit none
  
   
    interface
  
      subroutine launch_zhetd2_gpu(grid,&
          block,&
          sharedMem,&
          stream,&
          lda,&
          a,&
          a_n1,&
          a_n2,&
          a_lb1,&
          a_lb2,&
          tau,&
          tau_n1,&
          tau_lb1,&
          d,&
          d_n1,&
          d_lb1,&
          e,&
          e_n1,&
          e_lb1,&
          n) bind(c, name="launch_zhetd2_gpu")
        use iso_c_binding
        use hip
        implicit none
        type(dim3),intent(IN) :: grid
        type(dim3),intent(IN) :: block
        integer(c_int),intent(IN) :: sharedMem
        type(c_ptr),value,intent(IN) :: stream
        INTEGER,value :: lda
        type(c_ptr),value :: a
        integer(c_int),value,intent(IN) :: a_n1
        integer(c_int),value,intent(IN) :: a_n2
        integer(c_int),value,intent(IN) :: a_lb1
        integer(c_int),value,intent(IN) :: a_lb2
        type(c_ptr),value :: tau
        integer(c_int),value,intent(IN) :: tau_n1
        integer(c_int),value,intent(IN) :: tau_lb1
        type(c_ptr),value :: d
        integer(c_int),value,intent(IN) :: d_n1
        integer(c_int),value,intent(IN) :: d_lb1
        type(c_ptr),value :: e
        integer(c_int),value,intent(IN) :: e_n1
        integer(c_int),value,intent(IN) :: e_lb1
        INTEGER,value :: n
      end subroutine
  
    end interface
  
   
  
  
  end module zhetd2_gpu_kernels
module zhetrd_gpu
   use zhetrd_gpu_kernels
   use hip
   use iso_c_binding
   use iso_c_binding_ext
   use hipblas

contains

   subroutine zhetrd_gpu_h(uplo, N, A, lda, d, e, tau, work, lwork, nb)
      use zhetrd_gpu_kernels
      use eigsolve_vars

      use zhetd2_gpu
      implicit none
      character                                 :: uplo
      integer                                   :: N, lda, lwork, nb, nx, ldwork, istat
      integer                                   :: i, j, k, kk
      type(c_ptr),value :: d 
      integer(c_int) :: d_n1, d_lb1
      type(c_ptr),value :: e 
      integer(c_int) :: e_n1, e_lb1
      type(c_ptr),value :: work 
      integer(c_int) :: work_n1, work_lb1
      type(c_ptr),value :: A 
      integer(c_int) :: A_n1, A_n2, A_lb1, A_lb2
      type(c_ptr),value :: tau 
      integer(c_int) :: tau_n1, tau_lb1

      complex(8), parameter                     :: cone = cmplx(1, 0, 8)
      real(8), parameter                        :: one = 1.0_8
      type(dim3)                                :: threads

      d_n1 = n
      d_lb1 = 1
      e_n1 = (N - 1)
      e_lb1 = 1 
      work_n1 = lwork
      work_lb1 = 1
      A_n1 = lda
      A_n2 = N
      A_lb1 = 1
      A_lb2 = 1
      tau_n1 = (N - 1)
      tau_lb1 = 1
      if (uplo .ne. 'U') then
         print *, "Provided uplo type not supported!"
         return
      endif

      if (lwork < (nb + 2)*N .and. N > nb) then
         write (*, *), "Provided work array must be sized (nb+2)*N or greater!"
         return
      endif

      ldwork = N

      istat = hipblasSetStream(hipblasHandle, stream1)

      kk = N - ((N - 32)/nb)*nb
      k = N + 1
      do i = N - nb + 1, kk + 1, -nb
         ! Reduce columns i:i+nb-1 to tridiagonal form
         call zlatrd_gpu(uplo, i + nb - 1, nb, A, lda, e, tau, work, ldwork)

         ! Update trailing submatrix
         call hipblaszher2k(hipblasHandle,HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, (i - 1), nb, -cone,&
         inc_c_ptr(A,1_8*16*lda*(i-1)), lda, work, ldwork, one, a, lda)

         k = k - nb

      end do

      ! Finish any remaining columns to get final 32x32 block
      nb = k - 32 - 1
      i = k - nb

      if (nb > 0) then
         ! Reduce columns i:i+nb-1 to tridiagonal form
         call zlatrd_gpu(uplo, i + nb - 1, nb, A, lda, e, tau, work, ldwork)

         ! Update trailing submatrix
         call hipblaszher2k(hipblasHandle, HIPBLAS_FILL_MODE_UPPER,, HIPBLAS_OP_N, (i - 1), nb, -cone,&
         inc_c_ptr(A,1_8*16*lda*(i-1)), lda, work, ldwork, one, a, lda)

      endif

      ! Final block
      threads = dim3(32, 32, 1)
      grid = dim3(1,1,1)
      CALL launch_zhetd2_gpu(grid, threads,0,c_null_ptr, lda, A, a_n1, a_n2, a_lb1, a_lb2, tau,tau_n1, tau_lb1,d,d_n1,d_lb1,e,&
      e_n1,e_lb1,min(32, N))

      ! Copy superdiagonal back into A, store diagonal in d
      ! extracted to HIP C++ file
      ! TODO(gpufort) fix arguments
      CALL launch_krnl_2b8e8f_0_auto(0, c_null_ptr, a, a_n1, a_n2, a_lb1, a_lb2, d, d_n1, d_lb1, n)

   end subroutine zhetrd_gpu_h

   subroutine zlatrd_gpu(uplo, N, nb, A, lda, e, tau, W, ldw)
      use zhetrd_gpu_kernels
      use eigsolve_vars

      use zhemv_gpu
      implicit none
      character                                  :: uplo
      integer                                    :: N, nb, lda, ldw, istat
      integer                                    :: i, j, k, iw
      integer                                    :: blocks, threads
      type(c_ptr),value :: A 
      integer(c_int) :: A_n1, A_n2, A_lb1, A_lb2
      type(c_ptr),value :: W 
      integer(c_int) :: W_n1, W_n2, W_lb1 = 1, W_lb2 = 1
      type(c_ptr),value :: tau 
      integer(c_int) :: tau_n1, tau_lb1 = 1
      type(c_ptr),value :: e 
      integer(c_int) :: e_n1, e_lb1 = 1

      complex(8), parameter                      :: cone = cmplx(1, 0, 8), czero = cmplx(0, 0, 8), chalf = cmplx(0.5, 0, 8)

      type(dim3)                                 :: threads2D, blocks2D
      A_n1 = lda
      A_n2 = N
      A_lb1 = 1
      A_lb2 = 1
      W_n1 = ldw
      W_n2 = nb
      
      tau_n1 = (N - 1)
      
      e_n1 = (N - 1)
      
      if (uplo .ne. 'U') then
         print *, "Provided uplo type not supported!"
         return
      endif

      threads2D = dim3(32, 8, 1)
      threads = dim3(256,1,1)

      if (N <= 0) return

      ! Complete first iteration outside loop
      if (N > 1) then
         iw = nb
         ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
         CALL launch_zlarfg_kernel(dim3(1,1,1), threads, 0, c_null_ptr,N - 1, inc_c_ptr(tau,1_8*16*(N-1-1)),&
         inc_c_ptr(e,1_8*8*(N-1-1)),A,lda,-lda*(N-1))

         ! extracted to HIP C++ file
         ! TODO(gpufort) fix arguments
         CALL launch_krnl_9c27cb_1_auto(0, c_null_ptr, iw, w, w_n1, w_n2, w_lb1, w_lb2, n)

         blocks2D = dim3(10, ceiling(real(N - 1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
         CALL launch_zhemv_gpu(blocks2D, threads2D, 0, hipDefaultStream, N-1, lda,A, a_n1, a_n2, a_lb1, a_lb2,&
          A,lda ,-lda*(N-1) , W,ldw,-ldw*(iw-1))

         CALL launch_finish_W_col_kernel(dim3(1,1,1), threads,0, c_null_ptr, N - 1, inc_c_ptr(tau,1_8*16*(N-1-1)),  &
          A,lda,-lda*(N-1) ,W,ldw ,-ldw*(iw-1))
      endif

      do i = N - 1, N - nb + 1, -1
         iw = i - N + nb

         blocks2D = dim3(ceiling(real(max(i, N - i))/32), ceiling(real(N - i)/8), 1)
         !call zher2_mv_kernel<<<blocks2D, threads2D>>>(i, N-i, A(1, i+1), lda, W(1, iw+1), ldw, A(1, i), W(1, iw), ldw)
         CALL launch_zher2_mv_zlarfg_kernel(blocks2D, threads2D, 0, hipDefaultStream, i, N - i, lda, ldw, ldw, A, a_n1,1,i, W,w_n1,&
         1,iw, W, w_n1, 1,iw-1, A,lda, -lda*(i-1),A,lda, -lda*(i-1),inc_c_ptr(e,1_8*8*(i-1-1)) , inc_c_ptr(tau,1_8*16*(i-1-1)), finished)

         if (i > 1) then
            ! Generate elementary reflector H(i) to annihilate A(1:i-2, i)
            !call zlarfg_kernel<<<1, threads>>>(i-1, e(i-1), A(1, i), tau(i-1))

            blocks2D = dim3(min(10, ceiling(real(i - 1)/32)), ceiling(real(i - 1)/32), 1) !JR TODO: What is optimal number of columns for our problem size?
            CALL launch_zhemv_gpu(blocks2D, threads2D, 0, hipDefaultStream, i - 1, lda, A,a_n1, a_n2, a_lb1, a_lb2, &
            A, lda,-lda*(N-1) , W,ldw,-ldw*(iw-1))

            blocks2D = dim3(ceiling(real(i - 1)/32), ceiling(real(2*(n - i))/8), 1)
            CALL launch_stacked_zgemv_C(blocks2D, threads2D, 0, hipDefaultStream, n - i, i - 1, lda, ldw, A,a_n1,1,i,&
            W,w_n1,i,iw,A,lda,-lda*i,W, ldw-i,-ldw*(iw-1)-i,W,ldw-i,-ldw*iw-i)
            CALL launch_stacked_zgemv_N_finish_W(blocks2D, threads2D, 0, hipDefaultStream, i-1, n-i, lda, ldw,A,a_n1,1,&
            i,W,w_n1,1,iw,W,ldw-i,-ldw*(iw-1)-i, W,ldw-i,-ldw*iw-i,W,ldw,-ldw*(iw-1), inc_c_ptr(tau,1_8*16*(i-1-1)), &
            A,lda,-lda*(i-1), W, ldw, -ldw*(iw-1),finished)

            !call finish_W_col_kernel<<<1, threads>>>(i-1, tau(i-1), A(1, i), W(1, iw))

         end if
      end do
   end subroutine zlatrd_gpu

   ! extracted to HIP C++ file

   ! extracted to HIP C++ file

   ! extracted to HIP C++ file

   ! extracted to HIP C++ file

   ! extracted to HIP C++ file

   ! extracted to HIP C++ file

   ! extracted to HIP C++ file

end module zhetrd_gpu

