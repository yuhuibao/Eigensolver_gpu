! This file was generated by gpufort
          
           
module dsytd2_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_dsytd2_gpu(grid,&
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
        n) bind(c, name="launch_dsytd2_gpu")
      use iso_c_binding
      use hip
      implicit none
      type(dim3,,intent(IN, :: grid
      type(dim3,,intent(IN, :: block
      integer(c_int,,intent(IN, :: sharedMem
      type(c_ptr,,value,intent(IN, :: stream
      INTEGER(kind=),value :: lda
      type(c_ptr),value :: _a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
      type(c_ptr),value :: _tau
      integer(c_int),value,intent(IN) :: tau_n1
      integer(c_int),value,intent(IN) :: tau_lb1
      type(c_ptr),value :: _d
      integer(c_int),value,intent(IN) :: d_n1
      integer(c_int),value,intent(IN) :: d_lb1
      type(c_ptr),value :: _e
      integer(c_int),value,intent(IN) :: e_n1
      integer(c_int),value,intent(IN) :: e_lb1
      INTEGER(kind=),value :: n
    end subroutine

  end interface

  contains

    subroutine launch_dsytd2_gpu_cpu(lda,&
        _a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2,&
        _tau,&
        tau_n1,&
        tau_lb1,&
        _d,&
        d_n1,&
        d_lb1,&
        _e,&
        e_n1,&
        e_lb1,&
        n)
      use iso_c_binding
      use hip
      implicit none
      INTEGER(kind=),value :: lda
      type(c_ptr),value :: _a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
      type(c_ptr),value :: _tau
      integer(c_int),value,intent(IN) :: tau_n1
      integer(c_int),value,intent(IN) :: tau_lb1
      type(c_ptr),value :: _d
      integer(c_int),value,intent(IN) :: d_n1
      integer(c_int),value,intent(IN) :: d_lb1
      type(c_ptr),value :: _e
      integer(c_int),value,intent(IN) :: e_n1
      integer(c_int),value,intent(IN) :: e_lb1
      INTEGER(kind=),value :: n
            real(8),target :: a(a_n1,a_n2)
            real(8),target :: tau(tau_n1)
            real(8),target :: d(d_n1)
            real(8),target :: e(e_n1)
      CALL hipCheck(hipMemcpy(c_loc(a),_a,C_SIZEOF(a),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(tau),_tau,C_SIZEOF(tau),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(d),_d,C_SIZEOF(d),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(e),_e,C_SIZEOF(e),hipMemcpyDeviceToHost))
            use cudafor
            implicit none
            integer, value    :: lda
            real(8), device    :: a(lda, 32), tau(32)
            real(8), device    :: d(32), e(32)
            real(8), shared    :: a_s(32, 32)
            real(8), shared    :: alpha
            real(8), shared    :: taui
            real(8)           :: beta
            real(8)           :: alphar
            real(8)           :: xnorm, x, y, z, w
            real(8)           :: wc
            integer, value    :: n
            integer           :: tx, ty, tl, i, j, ii

            tx = threadIdx%x
            ty = threadIdx%y
            ! Linear id of the thread (tx,ty)
            tl = tx + blockDim%x*(ty - 1)

            ! Load a_d in shared memory
            if (tx <= N .and. ty <= N) then
               a_s(tx, ty) = a(tx, ty)
            endif

            call syncthreads()
            ! Symmetric matrix from upper triangular
            if (tx > ty) then
               a_s(tx, ty) = a_s(ty, tx)
            end if

            call syncthreads()

            ! For each column working backward
            do i = n - 1, 1, -1
               ! Generate elementary reflector
               ! Sum the vectors above the diagonal, only one warp active
               ! Reduce in a warp
               if (tl <= 32) then
                  if (tl < i) then
                     w = a_s(tl, i + 1)*a_s(tl, i + 1)
                  else
                     w = 0._8
                  endif

                  xnorm = __shfl_down(w, 1)
                  w = w + xnorm
                  xnorm = __shfl_down(w, 2)
                  w = w + xnorm
                  xnorm = __shfl_down(w, 4)
                  w = w + xnorm
                  xnorm = __shfl_down(w, 8)
                  w = w + xnorm
                  xnorm = __shfl_down(w, 16)
                  w = w + xnorm
               end if

               if (tl == 1) then
                  alpha = a_s(i, i + 1)
                  alphar = dble(alpha)
                  xnorm = dsqrt(w)

                  if (xnorm .eq. 0_8) then
                     ! H=1
                     taui = 0._8
                     alpha = 1.d0 ! To prevent scaling by dscal in this case
                  else
                     !Compute sqrt(alphar^2+xnorm^2) with  dlapy2(alphar,xnorm)
                     x = abs(alphar)
                     y = abs(xnorm)
                     w = max(x, y)
                     z = min(x, y)

                     if (z .eq. 0.d0) then
                        beta = -sign(w, alphar)
                     else
                        beta = -sign(w*sqrt(1.d0 + (z/w)**2), alphar)
                     endif

                     taui = (beta - alphar)/beta
                     alpha = 1.d0/(alphar - beta) ! scale factor for dscal

                  end if
               end if

               call syncthreads()

               ! dscal
               if (tl < i) then
                  a_s(tl, i + 1) = a_s(tl, i + 1)*alpha
               end if

               if (tl == 1) then
                  if (xnorm .ne. 0_8) then
                     alpha = beta
                  else
                     alpha = a_s(i, i + 1) ! reset alpha to original value
                  endif

                  e(i) = alpha
               end if

               if (taui .ne. (0.d0, 0.d0)) then
                  a_s(i, i + 1) = 1.d0
                  call syncthreads()
                  if (tl <= i) then
                     tau(tl) = 0.d0
                     do j = 1, i
                        tau(tl) = tau(tl) + taui*a_s(tl, j)*a_s(j, i + 1)
                     end do
                  end if

                  call syncthreads()

                  if (tl <= 32) then
                     if (tl <= i) then
                        x = -.5d0*taui*tau(tl)*a_s(tl, i + 1)
                     else
                        x = 0._8
                     endif

                     z = __shfl_xor(x, 1)
                     x = x + z
                     z = __shfl_xor(x, 2)
                     x = x + z
                     z = __shfl_xor(x, 4)
                     x = x + z
                     z = __shfl_xor(x, 8)
                     x = x + z
                     z = __shfl_xor(x, 16)
                     x = x + z

                  end if

                  call syncthreads()

                  if (tl <= i) then
                     tau(tl) = tau(tl) + x*a_s(tl, i + 1)
                  end if

                  if (tl == 1) alpha = x

                  call syncthreads()

                  if (tx <= i .and. ty <= i) then
                     a_s(tx, ty) = a_s(tx, ty) - a_s(tx, i + 1)*tau(ty) - a_s(ty, i + 1)*tau(tx)
                  end if
                  call syncthreads()

               endif

               if (tl == 1) then
                  a_s(i, i + 1) = e(i)
                  d(i + 1) = a_s(i + 1, i + 1)
                  tau(i) = taui
               end if

               call syncthreads()

            end do

            if (tl == 1) then
               d(1) = a_s(1, 1)
            endif

            call syncthreads()

            ! Back to device memory
            if (tx <= N .and. ty <= N) then
               a(tx, ty) = a_s(tx, ty)
            endif

      CALL hipCheck(hipMemcpy(_a,c_loc(a),C_SIZEOF(a),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_tau,c_loc(tau),C_SIZEOF(tau),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_d,c_loc(d),C_SIZEOF(d),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_e,c_loc(e),C_SIZEOF(e),hipMemcpyHostToDevice))

    end subroutine


end module dsytd2_gpu_kernels