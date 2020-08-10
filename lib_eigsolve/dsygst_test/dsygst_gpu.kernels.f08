! This file was generated by gpufort
          
           
module dsygst_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_krnl_afb01f_0(grid,&
        block,&
        sharedMem,&
        stream,&
        kb,&
        a,&
        k,&
        N) bind(c, name="launch_krnl_afb01f_0")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int):: sharedMem
      type(c_ptr) :: stream
      integer :: kb, k, N
      type(c_ptr) :: a
      
    end subroutine

    subroutine launch_krnl_afb01f_0_auto(sharedMem,&
        stream,&
        kb,&
        a,&
        k,&
        N) bind(c, name="launch_krnl_afb01f_0_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),value :: sharedMem
      integer(c_int),value :: k ,kb, N
      type(c_ptr),value :: stream
      type(c_ptr) :: a

    end subroutine

  end interface

!   contains

!     subroutine launch_krnl_afb01f_0_cpu(sharedMem,&
!         stream,&
!         kb,&
!         a,&
!         k)
!       use iso_c_binding
!       use hip
!       implicit none
!       integer(c_int),intent(IN) :: sharedMem
!       type(c_ptr),value,intent(IN) :: stream
!       integer,value :: kb
!       TODO declaration not found :: a
!       integer,value :: k
!       integer :: i
!       integer :: j
!       do j = k, k + kb - 1
!       do i = k, k + kb - 1
!       if (j < i) then
!          A(i, j) = A(j, i)
!       endif
!       end do
!       end do

!     end subroutine


end module dsygst_gpu_kernels