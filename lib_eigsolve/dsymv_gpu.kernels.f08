! This file was generated by gpufort
          
           
module dsymv_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_dsymv_gpu(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        lda,&
        a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2,&
        x,&
        x_n1,&
        x_lb1,&
        y,&
        y_n1,&
        y_lb1) bind(c, name="launch_dsymv_gpu")
      use iso_c_binding
      use hip
      implicit none
      type(dim3,,intent(IN, :: grid
      type(dim3,,intent(IN, :: block
      integer(c_int,,intent(IN, :: sharedMem
      type(c_ptr,,value,intent(IN, :: stream
      INTEGER(kind=),value :: n
      INTEGER(kind=),value :: lda
      type(c_ptr),value :: _a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
      type(c_ptr),value :: _x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
      type(c_ptr),value :: _y
      integer(c_int),value,intent(IN) :: y_n1
      integer(c_int),value,intent(IN) :: y_lb1
    end subroutine

  end interface

  contains

    subroutine launch_dsymv_gpu_cpu(n,&
        lda,&
        _a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2,&
        _x,&
        x_n1,&
        x_lb1,&
        _y,&
        y_n1,&
        y_lb1)
      use iso_c_binding
      use hip
      implicit none
      INTEGER(kind=),value :: n
      INTEGER(kind=),value :: lda
      type(c_ptr),value :: _a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
      type(c_ptr),value :: _x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
      type(c_ptr),value :: _y
      integer(c_int),value,intent(IN) :: y_n1
      integer(c_int),value,intent(IN) :: y_lb1
          real(8), N), intent(in)    ,target :: a(a_n1,a_n2)
          real(8), intent(in)         ,target :: x(x_n1)
          real(8),target :: y(y_n1)
      CALL hipCheck(hipMemcpy(c_loc(a),_a,C_SIZEOF(a),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(x),_x,C_SIZEOF(x),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(y),_y,C_SIZEOF(y),hipMemcpyDeviceToHost))
          use cudafor
          implicit none

          integer, value                                    :: N, lda
          real(8), dimension(lda, N), device, intent(in)    :: A
          real(8), dimension(N), device, intent(in)         :: x
          real(8), dimension(N), device                     :: y

          real(8), dimension(BX + 1, BX), shared              :: Ar_s
          real(8), dimension(BX), shared                    :: r_s

          integer                                           :: tx, ty, ii, jj, i, j, k, istat
          real(8)                                           :: rv1, rv2, mysum
          real(8)                                           :: Ar, xl

          ! ii,jj is index of top left corner of block
          ii = (blockIdx%y - 1)*blockDim%x + 1

          mysum = 0.0_8

          tx = threadIdx%x
          ty = threadIdx%y

          if (ii + (blockIdx%x - 1)*blockDim%x > N) return

          i = ii + tx - 1
          if (i <= N) then
             xl = x(i) ! read part of x for lower triangular multiply
          endif

          ! Loop over columns (skip all lower triangular blocks)
          do jj = ii + (blockIdx%x - 1)*blockDim%x, N, gridDim%x*blockDim%x
             j = jj + ty - 1

             ! Load block into shared memory
             ! CASE 1: Diagonal block
             if (ii == jj) then

                ! Load full block into shared memory
                do k = 0, NTILES - 1
                   if (i <= N .and. j + k*blockDim%y <= N) then
                      Ar_s(tx, ty + k*blockDim%y) = A(i, j + k*blockDim%y)
                   endif
                end do

                call syncthreads()

                ! Reflect to populate lower triangular part with true values of A
                do k = 0, NTILES - 1
                   if (tx > ty + k*blockDim%y) then
                      Ar_s(tx, ty + k*blockDim%y) = Ar_s(ty + k*blockDim%y, tx)
                   endif
                end do

                call syncthreads()

                do k = 0, NTILES - 1
                   if (i <= N .and. j + k*blockDim%y <= N) then
                      mysum = mysum + Ar_s(tx, ty + k*blockDim%y)*x(j + k*blockDim%y)
                   endif
                end do

                !call syncthreads()

                ! CASE 2: Upper triangular block
             else if (ii < jj) then
                do k = 0, NTILES - 1
                   if (j + k*blockDim%y <= N) then
                      Ar = A(i, j + k*blockDim%y)
                   endif

                   if (i <= N .and. j + k*blockDim%y <= N) then
                      mysum = mysum + Ar*x(j + k*blockDim%y)
                   endif

                   ! Perform product for symmetric lower block here
                   if (i <= N .and. j + k*blockDim%y <= N) then
                      rv1 = Ar*xl
                   else
                      rv1 = 0.0_8
                   endif

                   !Partial sum within warps using shuffle
                   rv2 = __shfl_down(rv1, 1)
                   rv1 = rv1 + rv2
                   rv2 = __shfl_down(rv1, 2)
                   rv1 = rv1 + rv2
                   rv2 = __shfl_down(rv1, 4)
                   rv1 = rv1 + rv2
                   rv2 = __shfl_down(rv1, 8)
                   rv1 = rv1 + rv2
                   rv2 = __shfl_down(rv1, 16)
                   rv1 = rv1 + rv2

                   if (tx == 1) then
                      r_s(ty + k*blockDim%y) = rv1
                   endif
                enddo

                call syncthreads()

                if (ty == 1 .and. jj + tx - 1 <= N) then
                   istat = atomicadd(y(jj + tx - 1), r_s(tx))
                endif
                !call syncthreads()

             endif

             call syncthreads()

          end do

          if (i <= N) then
             istat = atomicadd(y(i), mysum)
          endif

      CALL hipCheck(hipMemcpy(_a,c_loc(a),C_SIZEOF(a),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_x,c_loc(x),C_SIZEOF(x),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_y,c_loc(y),C_SIZEOF(y),hipMemcpyHostToDevice))

    end subroutine


end module dsymv_gpu_kernels