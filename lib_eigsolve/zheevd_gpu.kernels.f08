! This file was generated by gpufort
          
           
module zheevd_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_krnl_e26a05_0(grid,&
        block,&
        sharedMem,&
        stream,&
        z,&
        z_n1,&
        z_n2,&
        z_lb1,&
        z_lb2,&
        n,&
        a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2) bind(c, name="launch_krnl_e26a05_0")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: z
      integer(c_int),value,intent(IN) :: z_n1
      integer(c_int),value,intent(IN) :: z_n2
      integer(c_int),value,intent(IN) :: z_lb1
      integer(c_int),value,intent(IN) :: z_lb2
      INTEGER,value :: n
      type(c_ptr),value :: a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
    end subroutine

    subroutine launch_krnl_e26a05_0_auto(sharedMem,&
        stream,&
        z,&
        z_n1,&
        z_n2,&
        z_lb1,&
        z_lb2,&
        n,&
        a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2) bind(c, name="launch_krnl_e26a05_0_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: z
      integer(c_int),value,intent(IN) :: z_n1
      integer(c_int),value,intent(IN) :: z_n2
      integer(c_int),value,intent(IN) :: z_lb1
      integer(c_int),value,intent(IN) :: z_lb2
      INTEGER,value :: n
      type(c_ptr),value :: a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
    end subroutine

    subroutine launch_krnl_b1ccc1_1(grid,&
        block,&
        sharedMem,&
        stream,&
        k,&
        n,&
        w,&
        w_n1,&
        w_n2,&
        w_lb1,&
        w_lb2,&
        v,&
        v_n1,&
        v_n2,&
        v_lb1,&
        v_lb2) bind(c, name="launch_krnl_b1ccc1_1")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: k
      INTEGER,value :: n
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_n2
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_n2
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
    end subroutine

    subroutine launch_krnl_b1ccc1_1_auto(sharedMem,&
        stream,&
        k,&
        n,&
        w,&
        w_n1,&
        w_n2,&
        w_lb1,&
        w_lb2,&
        v,&
        v_n1,&
        v_n2,&
        v_lb1,&
        v_lb2) bind(c, name="launch_krnl_b1ccc1_1_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: k
      INTEGER,value :: n
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_n2
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_n2
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
    end subroutine

    subroutine launch_krnl_b95769_2(grid,&
        block,&
        sharedMem,&
        stream,&
        m,&
        k,&
        w,&
        w_n1,&
        w_n2,&
        w_lb1,&
        w_lb2,&
        v,&
        v_n1,&
        v_n2,&
        v_lb1,&
        v_lb2) bind(c, name="launch_krnl_b95769_2")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: m
      INTEGER,value :: k
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_n2
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_n2
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
    end subroutine

    subroutine launch_krnl_b95769_2_auto(sharedMem,&
        stream,&
        m,&
        k,&
        w,&
        w_n1,&
        w_n2,&
        w_lb1,&
        w_lb2,&
        v,&
        v_n1,&
        v_n2,&
        v_lb1,&
        v_lb2) bind(c, name="launch_krnl_b95769_2_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: m
      INTEGER,value :: k
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_n2
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_n2
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
    end subroutine

    subroutine launch_finish_t_block_kernel(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        ldt,&
        t,&
        t_n1,&
        t_n2,&
        t_lb1,&
        t_lb2,&
        tau,&
        tau_n1,&
        tau_lb1) bind(c, name="launch_finish_t_block_kernel")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: n
      INTEGER,value :: ldt
      type(c_ptr),value :: t
      integer(c_int),value,intent(IN) :: t_n1
      integer(c_int),value,intent(IN) :: t_n2
      integer(c_int),value,intent(IN) :: t_lb1
      integer(c_int),value,intent(IN) :: t_lb2
      type(c_ptr),value :: tau
      integer(c_int),value,intent(IN) :: tau_n1
      integer(c_int),value,intent(IN) :: tau_lb1 
    end subroutine

  end interface

  


end module zheevd_gpu_kernels