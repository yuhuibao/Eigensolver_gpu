! This file was generated by gpufort

! Fortran implementation:
!   attributes(global) subroutine dsymv_gpu(N, A, lda, x, y)
!      use cudafor
!      implicit none
!
!      integer, value                                    :: N, lda
!      real(8), dimension(lda, N), device, intent(in)    :: A
!      real(8), dimension(N), device, intent(in)         :: x
!      real(8), dimension(N), device                     :: y
!
!      real(8), dimension(BX + 1, BX), shared              :: Ar_s
!      real(8), dimension(BX), shared                    :: r_s
!
!      integer                                           :: tx, ty, ii, jj, i, j, k, istat
!      real(8)                                           :: rv1, rv2, mysum
!      real(8)                                           :: Ar, xl
!
!      ! ii,jj is index of top left corner of block
!      ii = (blockIdx%y - 1)*blockDim%x + 1
!
!      mysum = 0.0_8
!
!      tx = threadIdx%x
!      ty = threadIdx%y
!
!      if (ii + (blockIdx%x - 1)*blockDim%x > N) return
!
!      i = ii + tx - 1
!      if (i <= N) then
!         xl = x(i) ! read part of x for lower triangular multiply
!      endif
!
!      ! Loop over columns (skip all lower triangular blocks)
!      do jj = ii + (blockIdx%x - 1)*blockDim%x, N, gridDim%x*blockDim%x
!         j = jj + ty - 1
!
!         ! Load block into shared memory
!         ! CASE 1: Diagonal block
!         if (ii == jj) then
!
!            ! Load full block into shared memory
!            do k = 0, NTILES - 1
!               if (i <= N .and. j + k*blockDim%y <= N) then
!                  Ar_s(tx, ty + k*blockDim%y) = A(i, j + k*blockDim%y)
!               endif
!            end do
!
!            call syncthreads()
!
!            ! Reflect to populate lower triangular part with true values of A
!            do k = 0, NTILES - 1
!               if (tx > ty + k*blockDim%y) then
!                  Ar_s(tx, ty + k*blockDim%y) = Ar_s(ty + k*blockDim%y, tx)
!               endif
!            end do
!
!            call syncthreads()
!
!            do k = 0, NTILES - 1
!               if (i <= N .and. j + k*blockDim%y <= N) then
!                  mysum = mysum + Ar_s(tx, ty + k*blockDim%y)*x(j + k*blockDim%y)
!               endif
!            end do
!
!            !call syncthreads()
!
!            ! CASE 2: Upper triangular block
!         else if (ii < jj) then
!            do k = 0, NTILES - 1
!               if (j + k*blockDim%y <= N) then
!                  Ar = A(i, j + k*blockDim%y)
!               endif
!
!               if (i <= N .and. j + k*blockDim%y <= N) then
!                  mysum = mysum + Ar*x(j + k*blockDim%y)
!               endif
!
!               ! Perform product for symmetric lower block here
!               if (i <= N .and. j + k*blockDim%y <= N) then
!                  rv1 = Ar*xl
!               else
!                  rv1 = 0.0_8
!               endif
!
!               !Partial sum within warps using shuffle
!               rv2 = __shfl_down(rv1, 1)
!               rv1 = rv1 + rv2
!               rv2 = __shfl_down(rv1, 2)
!               rv1 = rv1 + rv2
!               rv2 = __shfl_down(rv1, 4)
!               rv1 = rv1 + rv2
!               rv2 = __shfl_down(rv1, 8)
!               rv1 = rv1 + rv2
!               rv2 = __shfl_down(rv1, 16)
!               rv1 = rv1 + rv2
!
!               if (tx == 1) then
!                  r_s(ty + k*blockDim%y) = rv1
!               endif
!            enddo
!
!            call syncthreads()
!
!            if (ty == 1 .and. jj + tx - 1 <= N) then
!               istat = atomicadd(y(jj + tx - 1), r_s(tx))
!            endif
!            !call syncthreads()
!
!         endif
!
!         call syncthreads()
!
!      end do
!
!      if (i <= N) then
!         istat = atomicadd(y(i), mysum)
!      endif
!
!   end subroutine dsymv_gpu
!
function test_launch_dsymv_gpu()
    ! errorCode > 0 implies that the test has failed
    use iso_c_binding
    use hipfort
    use dsymv_gpu_kernels
    implicit none
    integer :: errorCode = 1
    ! TODO fix parameters
    ! - Add missing arguments
    ! - Determine size of arrays (typically indicated by 'type(c_ptr)' type)
    ! - Add target where we need a pointer
    type(dim3,, intent(IN, :: grid
    type(dim3,, intent(IN, :: block
    integer(c_int,, intent(IN, :: sharedMem
    type(c_ptr,, value, intent(IN, :: stream
    INTEGER(kind=), value :: n
    INTEGER(kind=), value :: lda
    type(c_ptr), value :: _a
    integer(c_int), value, intent(IN) :: a_n1
    integer(c_int), value, intent(IN) :: a_n2
    integer(c_int), value, intent(IN) :: a_lb1
    integer(c_int), value, intent(IN) :: a_lb2
    type(c_ptr), value :: _x
    integer(c_int), value, intent(IN) :: x_n1
    integer(c_int), value, intent(IN) :: x_lb1
    type(c_ptr), value :: _y
    integer(c_int), value, intent(IN) :: y_n1
    integer(c_int), value, intent(IN) :: y_lb1
    ! TODO Create initial data on host
    ! TODO Copy data to device ! (dest,src,size,direction)
    CALL hipCheck(hipMemcpy(???, c_loc(???), C_SIZEOF(???), hipMemcpyHostToDevice)) !
    CALL hipCheck(hipMemcpy(???, c_loc(???), C_SIZEOF(???), hipMemcpyHostToDevice)) !
    ! ... might be more (or less) than two memcopies
    ! TODO run the test
  CALL launch_dsymv_gpu(0,c_null_ptr,grid,block,sharedMem,stream,n,lda,a,a_n1,a_n2,a_lb1,a_lb2,x,x_n1,x_lb1,y,y_n1,y_lb1) ! Modify sharedMem if other than default 0
    CALL hipCheck(hipDeviceSynchronize())
  CALL launch_dsymv_gpu_cpu(0,c_null_ptr,grid,block,sharedMem,stream,n,lda,a,a_n1,a_n2,a_lb1,a_lb2,x,x_n1,x_lb1,y,y_n1,y_lb1)

    ! TODO Copy results back to host
    CALL hipCheck(hipMemcpy(c_loc(???), ???, C_SIZEOF(???), hipMemcpyDeviceToHost)
    CALL hipCheck(hipMemcpy(c_loc(???), ???, C_SIZEOF(???), hipMemcpyDeviceToHost)
    ! ... might be more (or less) than two memcopies
    ! TODO Compare results
    ! TODO Update error code if the results do not match
    return errorCode
end function

program test_dsymv_gpu_kernels
    implicit none
    integer :: globalErrorCode = 0, errorCode, fails = 0, tests = 0
    ! declare test functions and return type
    integer :: test_launch_dsymv_gpu
    write (*, *) "SUITE test_dsymv_gpu_kernels run ..."
    errorCode = test_launch_dsymv_gpu()
    IF (errorCode > 0) THEN
        fails = fails + 1
        write (*, *) "TEST test_launch_dsymv_gpu ... FAILURE"
    ELSE
        write (*, *) "TEST test_launch_dsymv_gpu ... SUCCESS"
    END IF
    tests = tests + 1
    globalErrorCode = globalErrorCode + errorCode

    IF (globalErrorCode > 0) THEN
        write (*, *) "SUITE test_dsymv_gpu_kernels ... FAILURE passed:", (tests - fails), " failed:", fails, " total:", tests
    ELSE
        write (*, *) "SUITE test_dsymv_gpu_kernels ... SUCCESS passed:", (tests - fails), " failed:", fails, " total:", tests
    END IF
end program test_dsymv_gpu_kernels
