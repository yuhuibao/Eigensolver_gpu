! This file was generated by gpufort
          
           
module dsyevd_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_krnl_e26a05_0(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        z,&
        a) bind(c, name="launch_krnl_e26a05_0")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      integer,value :: n
      TODO declaration not found :: z
      TODO declaration not found :: a
    end subroutine

    subroutine launch_krnl_e26a05_0_auto(sharedMem,&
        stream,&
        n,&
        z,&
        a) bind(c, name="launch_krnl_e26a05_0_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      integer,value :: n
      TODO declaration not found :: z
      TODO declaration not found :: a
    end subroutine

    subroutine launch_krnl_b1f342_1(grid,&
        block,&
        sharedMem,&
        stream,&
        w,&
        n,&
        k,&
        v) bind(c, name="launch_krnl_b1f342_1")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: w
      integer,value :: n
      integer,value :: k
      type(c_ptr),value :: v
    end subroutine

    subroutine launch_krnl_b1f342_1_auto(sharedMem,&
        stream,&
        w,&
        n,&
        k,&
        v) bind(c, name="launch_krnl_b1f342_1_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: w
      integer,value :: n
      integer,value :: k
      type(c_ptr),value :: v
    end subroutine

    subroutine launch_krnl_b95769_2(grid,&
        block,&
        sharedMem,&
        stream,&
        m,&
        w,&
        k,&
        v) bind(c, name="launch_krnl_b95769_2")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      integer,value :: m
      type(c_ptr),value :: w
      integer,value :: k
      type(c_ptr),value :: v
    end subroutine

    subroutine launch_krnl_b95769_2_auto(sharedMem,&
        stream,&
        m,&
        w,&
        k,&
        v) bind(c, name="launch_krnl_b95769_2_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      integer,value :: m
      type(c_ptr),value :: w
      integer,value :: k
      type(c_ptr),value :: v
    end subroutine

    subroutine launch_finish_t_block_kernel(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        ldt,&
        t,&
        tau) bind(c, name="launch_finish_t_block_kernel")
      use iso_c_binding
      use hip
      implicit none
      type(dim3,,intent(IN, :: grid
      type(dim3,,intent(IN, :: block
      integer(c_int,,intent(IN, :: sharedMem
      type(c_ptr,,value,intent(IN, :: stream
      integer,value :: n
      integer,value :: ldt
      type(c_ptr),value :: _t
      type(c_ptr),value :: _tau
    end subroutine

  end interface

  contains

    subroutine launch_krnl_e26a05_0_cpu(sharedMem,&
        stream,&
        n,&
        z,&
        a)
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      integer,value :: n
      TODO declaration not found :: z
      TODO declaration not found :: a
      integer :: i
      integer :: j
      do j = 1, N
      do i = 1, N
      if (i > j) then
         A(i, j) = Z(i, j)
      endif
      end do
      end do

    end subroutine

    subroutine launch_krnl_b1f342_1_cpu(sharedMem,&
        stream,&
        _w,&
        n,&
        k,&
        _v)
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: _w
      integer,value :: n
      integer,value :: k
      type(c_ptr),value :: _v
      integer :: i
            real(8), K),target :: w()
            real(8), K),target :: v()
      integer :: j
      CALL hipCheck(hipMemcpy(c_loc(w),_w,C_SIZEOF(w),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(v),_v,C_SIZEOF(v),hipMemcpyDeviceToHost))
      do j = 1, K
      do i = N - K + 1, N
      if (i - N + K == j) then
         V(i, j) = 1.0d0
      else if (i - N + k > j) then
         W(i - N + k, j) = V(i, j)
         V(i, j) = 0.0d0
      endif
      end do
      end do
      CALL hipCheck(hipMemcpy(_w,c_loc(w),C_SIZEOF(w),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_v,c_loc(v),C_SIZEOF(v),hipMemcpyHostToDevice))

    end subroutine

    subroutine launch_krnl_b95769_2_cpu(sharedMem,&
        stream,&
        m,&
        _w,&
        k,&
        _v)
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      integer,value :: m
      type(c_ptr),value :: _w
      integer,value :: k
      type(c_ptr),value :: _v
      integer :: i
            real(8), K),target :: w()
            real(8), K),target :: v()
      integer :: j
      CALL hipCheck(hipMemcpy(c_loc(w),_w,C_SIZEOF(w),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(v),_v,C_SIZEOF(v),hipMemcpyDeviceToHost))
      do j = 1, K
      do i = M - K + 1, M
      if (i - M + k > j) then
         V(i, j) = W(i - M + k, j)
      endif
      end do
      end do
      CALL hipCheck(hipMemcpy(_w,c_loc(w),C_SIZEOF(w),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_v,c_loc(v),C_SIZEOF(v),hipMemcpyHostToDevice))

    end subroutine

    subroutine launch_finish_t_block_kernel_cpu(n,&
        ldt,&
        _t,&
        _tau)
      use iso_c_binding
      use hip
      implicit none
      integer,value :: n
      integer,value :: ldt
      type(c_ptr),value :: _t
      type(c_ptr),value :: _tau
            real(8), K),target :: t()
            real(8),target :: tau()
      type(c_ptr) :: t_s
      integer :: tid
      integer :: tx
      integer :: ty
      integer :: i
      integer :: j
      integer :: k
      integer :: diag
      complex(kind=8) :: cv
      CALL hipCheck(hipMemcpy(c_loc(t),_t,C_SIZEOF(t),hipMemcpyDeviceToHost))
      CALL hipCheck(hipMemcpy(c_loc(tau),_tau,C_SIZEOF(tau),hipMemcpyDeviceToHost))
            implicit none

            integer, value                     :: N, ldt

            real(8), dimension(ldt, K), device :: T

            real(8), dimension(K), device      :: tau

            ! T_s contains only lower triangular elements of T in linear array, by row

            real(8), dimension(2080), shared   :: T_s

            ! (i,j) --> ((i-1)*i/2 + j)

      #define IJ2TRI(i,j) (ISHFT((i-1)*i,-1) + j)

            integer     :: tid, tx, ty, i, j, k, diag

            complex(8)  :: cv

            tx = threadIdx%x

            ty = threadIdx%y

            tid = (threadIdx%y - 1)*blockDim%x + tx ! Linear thread id

            ! Load T into shared memory

            if (tx <= N) then

               do j = ty, N, blockDim%y

                  cv = tau(j)

                  if (tx > j) then

                     T_s(IJ2TRI(tx, j)) = -cv*T(tx, j)

                  else if (tx == j) then

                     T_s(IJ2TRI(tx, j)) = cv

                  endif

               end do

            end if

            call syncthreads()

            ! Perform column by column update by first thread column

            do i = N - 1, 1, -1

               if (ty == 1) then

                  if (tx > i .and. tx <= N) then

                     cv = 0.0d0

                     do j = i + 1, tx

                        cv = cv + T_s(IJ2TRI(j, i))*T_s(IJ2TRI(tx, j))

                     end do

                  endif

               endif

               call syncthreads()

               if (ty == 1 .and. tx > i .and. tx <= N) then

                  T_s(IJ2TRI(tx, i)) = cv

               endif

               call syncthreads()

            end do

            call syncthreads()

            ! Write T_s to global

            if (tx <= N) then

               do j = ty, N, blockDim%y

                  if (tx >= j) then

                     T(tx, j) = T_s(IJ2TRI(tx, j))

                  endif

               end do

            end if

      CALL hipCheck(hipMemcpy(_t,c_loc(t),C_SIZEOF(t),hipMemcpyHostToDevice))
      CALL hipCheck(hipMemcpy(_tau,c_loc(tau),C_SIZEOF(tau),hipMemcpyHostToDevice))

    end subroutine


end module dsyevd_gpu_kernels