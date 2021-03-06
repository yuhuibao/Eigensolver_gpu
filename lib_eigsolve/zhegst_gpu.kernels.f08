! This file was generated by gpufort
          
           
module zhegst_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_krnl_185097_0(grid,&
        block,&
        sharedMem,&
        stream,&
        a,&
        k,&
        kb) bind(c, name="launch_krnl_185097_0")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      TODO declaration not found :: a
      integer,value :: k
      integer,value :: kb
    end subroutine

    subroutine launch_krnl_185097_0_auto(sharedMem,&
        stream,&
        a,&
        k,&
        kb) bind(c, name="launch_krnl_185097_0_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      TODO declaration not found :: a
      integer,value :: k
      integer,value :: kb
    end subroutine

    subroutine launch_krnl_9c3ecb_1(grid,&
        block,&
        sharedMem,&
        stream,&
        a,&
        k,&
        kb) bind(c, name="launch_krnl_9c3ecb_1")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      TODO declaration not found :: a
      integer,value :: k
      integer,value :: kb
    end subroutine

    subroutine launch_krnl_9c3ecb_1_auto(sharedMem,&
        stream,&
        a,&
        k,&
        kb) bind(c, name="launch_krnl_9c3ecb_1_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      TODO declaration not found :: a
      integer,value :: k
      integer,value :: kb
    end subroutine

  end interface

  contains

    subroutine launch_krnl_185097_0_cpu(sharedMem,&
        stream,&
        a,&
        k,&
        kb)
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      TODO declaration not found :: a
      integer,value :: k
      integer,value :: kb
      integer :: j
      integer :: i
      do j = k, k + kb - 1
      do i = k, k + kb - 1
      if (j < i) then
         A(i, j) = conjg(A(j, i))
      endif
      end do
      end do

    end subroutine

    subroutine launch_krnl_9c3ecb_1_cpu(sharedMem,&
        stream,&
        a,&
        k,&
        kb)
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      TODO declaration not found :: a
      integer,value :: k
      integer,value :: kb
      integer :: j
      integer :: i
      do j = k, k + kb - 1
      do i = k, k + kb - 1
      if (i == j) then
         A(i, j) = dble(A(i, j))
      endif
      end do
      end do

    end subroutine


end module zhegst_gpu_kernels