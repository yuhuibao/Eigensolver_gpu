! This file was generated by gpufort
          
           
module zhetrd_gpu_kernels
  use hip
  implicit none

 
  interface

    subroutine launch_krnl_2b8e8f_0(grid,&
        block,&
        sharedMem,&
        stream,&
        a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2,&
        d,&
        d_n1,&
        d_lb1,&
        n) bind(c, name="launch_krnl_2b8e8f_0")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
      type(c_ptr),value :: d
      integer(c_int),value,intent(IN) :: d_n1
      integer(c_int),value,intent(IN) :: d_lb1
      INTEGER(kind=),value :: n
    end subroutine

    subroutine launch_krnl_2b8e8f_0_auto(sharedMem,&
        stream,&
        a,&
        a_n1,&
        a_n2,&
        a_lb1,&
        a_lb2,&
        d,&
        d_n1,&
        d_lb1,&
        n) bind(c, name="launch_krnl_2b8e8f_0_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      type(c_ptr),value :: a
      integer(c_int),value,intent(IN) :: a_n1
      integer(c_int),value,intent(IN) :: a_n2
      integer(c_int),value,intent(IN) :: a_lb1
      integer(c_int),value,intent(IN) :: a_lb2
      type(c_ptr),value :: d
      integer(c_int),value,intent(IN) :: d_n1
      integer(c_int),value,intent(IN) :: d_lb1
      INTEGER,value :: n
    end subroutine

    subroutine launch_krnl_9c27cb_1(grid,&
        block,&
        sharedMem,&
        stream,&
        iw,&
        w,&
        w_n1,&
        w_n2,&
        w_lb1,&
        w_lb2,&
        n) bind(c, name="launch_krnl_9c27cb_1")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: iw
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_n2
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      INTEGER,value :: n
    end subroutine

    subroutine launch_krnl_9c27cb_1_auto(sharedMem,&
        stream,&
        iw,&
        w,&
        w_n1,&
        w_n2,&
        w_lb1,&
        w_lb2,&
        n) bind(c, name="launch_krnl_9c27cb_1_auto")
      use iso_c_binding
      use hip
      implicit none
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: iw
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_n2
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      INTEGER,value :: n
    end subroutine

   

    subroutine launch_zlarfg_kernel(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        tau,&
        e,&
        x,&
        x_n1,&
        x_lb1) bind(c, name="launch_zlarfg_kernel")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: n
      type(c_ptr),value :: tau
      type(c_ptr),value :: e
      type(c_ptr),value :: x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
    end subroutine

    subroutine launch_zher2_mv_zlarfg_kernel(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        m,&
        ldv,&
        ldw,&
        ldw2,&
        v,&
        v_n1,&
        v_lb1,&
        v_lb2,&
        w,&
        w_n1,&
        w_lb1,&
        w_lb2,&
        w2,&
        w2_n1,&
        w2_lb1,&
        w2_lb2,&
        x,&
        x_n1,&
        x_lb1,&
        x2,&
        x2_n1,&
        x2_lb1,&
        tau,&
        e,&
        finished) bind(c, name="launch_zher2_mv_zlarfg_kernel")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: n
      INTEGER,value :: m
      INTEGER,value :: ldv
      INTEGER,value :: ldw
      INTEGER,value :: ldw2
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: w2
      integer(c_int),value,intent(IN) :: w2_n1
      integer(c_int),value,intent(IN) :: w2_lb1
      integer(c_int),value,intent(IN) :: w2_lb2
      type(c_ptr),value :: x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
      type(c_ptr),value :: tau
      type(c_ptr),value :: e
      type(c_ptr),value :: finished
    end subroutine

    subroutine launch_stacked_zgemv_c(grid,&
        block,&
        sharedMem,&
        stream,&
        m,&
        n,&
        ldv,&
        ldw,&
        v,&
        v_n1,&
        v_lb1,&
        v_lb2,&
        w,&
        w_n1,&
        w_lb1,&
        w_lb2,&
        x,&
        x_n1,&
        x_lb1,&
        z1,&
        z1_n1,&
        z1_lb1,&
        z2,&
        z2_n1,&
        z2_lb1) bind(c, name="launch_stacked_zgemv_c")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: m
      INTEGER,value :: n
      INTEGER,value :: ldv
      INTEGER,value :: ldw
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
      type(c_ptr),value :: z1
      integer(c_int),value,intent(IN) :: z1_n1
      integer(c_int),value,intent(IN) :: z1_lb1
      type(c_ptr),value :: z2
      integer(c_int),value,intent(IN) :: z2_n1
      integer(c_int),value,intent(IN) :: z2_lb1
    end subroutine

    

    subroutine launch_finish_w_col_kernel(grid,&
        block,&
        sharedMem,&
        stream,&
        n,&
        tau,&
        x,&
        x_n1,&
        x_lb1,&
        y,&
        y_n1,&
        y_lb1) bind(c, name="launch_finish_w_col_kernel")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: n
      type(c_ptr),value :: tau
      type(c_ptr),value :: x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
      type(c_ptr),value :: y
      integer(c_int),value,intent(IN) :: y_n1
      integer(c_int),value,intent(IN) :: y_lb1
    end subroutine

    subroutine launch_stacked_zgemv_n_finish_w(grid,&
        block,&
        sharedMem,&
        stream,&
        m,&
        n,&
        ldv,&
        ldw,&
        v,&
        v_n1,&
        v_lb1,&
        v_lb2,&
        w,&
        w_n1,&
        w_lb1,&
        w_lb2,&
        z1,&
        z1_n1,&
        z1_lb1,&
        z2,&
        z2_n1,&
        z2_lb1,&
        y,&
        y_n1,&
        y_lb1,&
        tau,&
        x,&
        x_n1,&
        x_lb1,&
        y2,&
        y2_n1,&
        y2_lb1,&
        finished) bind(c, name="launch_stacked_zgemv_n_finish_w")
      use iso_c_binding
      use hip
      implicit none
      type(dim3),intent(IN) :: grid
      type(dim3),intent(IN) :: block
      integer(c_int),intent(IN) :: sharedMem
      type(c_ptr),value,intent(IN) :: stream
      INTEGER,value :: m
      INTEGER,value :: n
      INTEGER,value :: ldv
      INTEGER,value :: ldw
      type(c_ptr),value :: v
      integer(c_int),value,intent(IN) :: v_n1
      integer(c_int),value,intent(IN) :: v_lb1
      integer(c_int),value,intent(IN) :: v_lb2
      type(c_ptr),value :: w
      integer(c_int),value,intent(IN) :: w_n1
      integer(c_int),value,intent(IN) :: w_lb1
      integer(c_int),value,intent(IN) :: w_lb2
      type(c_ptr),value :: z1
      integer(c_int),value,intent(IN) :: z1_n1
      integer(c_int),value,intent(IN) :: z1_lb1
      type(c_ptr),value :: z2
      integer(c_int),value,intent(IN) :: z2_n1
      integer(c_int),value,intent(IN) :: z2_lb1
      type(c_ptr),value :: y
      integer(c_int),value,intent(IN) :: y_n1
      integer(c_int),value,intent(IN) :: y_lb1
      type(c_ptr),value :: tau
      type(c_ptr),value :: x
      integer(c_int),value,intent(IN) :: x_n1
      integer(c_int),value,intent(IN) :: x_lb1
      type(c_ptr),value :: y2
      integer(c_int),value,intent(IN) :: y2_n1
      integer(c_int),value,intent(IN) :: y2_lb1
      type(c_ptr),value :: finished
    end subroutine

  end interface

  


end module zhetrd_gpu_kernels