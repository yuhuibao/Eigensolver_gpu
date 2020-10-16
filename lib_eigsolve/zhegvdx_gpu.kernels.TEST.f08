! This file was generated by gpufort

 
! 
! Hints:
! Device variables in scope:
!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: n, m, lda, ldb, ldz, il, iu, ldz_h, info, nb

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       integer                                     :: lwork_h, lrwork_h, liwork_h, lwork, lrwork, liwork, istat

!       real(8), dimension(1:lrwork), device        :: rwork

!       complex(8), dimension(1:lwork), device      :: work

!       logical, optional                           :: _skip_host_copy

!       complex(8), dimension(1:lda, 1:n), device   :: a

!       complex(8), dimension(1:ldb, 1:n), device   :: b

!       complex(8), dimension(1:ldz, 1:n), device   :: z

!       real(8), dimension(1:n), device             :: w

!       complex(8), parameter :: cone = cmplx(1,0,8)

!       integer :: i, j

!       integer :: i, j

!       logical :: skip_host_copy

function test_launch_krnl_959801_0_auto()
  ! errorCode > 0 implies that the test has failed
  use iso_c_binding
  use hip
  use zhegvdx_gpu_kernels
  implicit none
  integer :: errorCode = 1
  ! TODO fix parameters
  ! - Add missing arguments
  ! - Determine size of arrays (typically indicated by 'type(c_ptr)' type)
  ! - Add target where we need a pointer
  integer(c_int),intent(IN) :: sharedMem
  type(c_ptr),value,intent(IN) :: stream
  type(c_ptr),value :: z
  integer(c_int),value,intent(IN) :: z_n1
  integer(c_int),value,intent(IN) :: z_n2
  integer(c_int),value,intent(IN) :: z_lb1
  integer(c_int),value,intent(IN) :: z_lb2
  INTEGER(kind=),value :: n
  type(c_ptr),value :: a
  integer(c_int),value,intent(IN) :: a_n1
  integer(c_int),value,intent(IN) :: a_n2
  integer(c_int),value,intent(IN) :: a_lb1
  integer(c_int),value,intent(IN) :: a_lb2
  ! TODO Create initial data on host
  ! TODO Copy data to device ! (dest,src,size,direction)
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  ! ... might be more (or less) than two memcopies 
  ! TODO run the test
  CALL launch_krnl_959801_0_auto(0,c_null_ptr,sharedMem,stream,z,z_n1,z_n2,z_lb1,z_lb2,n,a,a_n1,a_n2,a_lb1,a_lb2) ! Modify sharedMem if other than default 0
  CALL hipCheck(hipDeviceSynchronize())
  CALL launch_krnl_959801_0_cpu(0,c_null_ptr,sharedMem,stream,z,z_n1,z_n2,z_lb1,z_lb2,n,a,a_n1,a_n2,a_lb1,a_lb2)

  ! TODO Copy results back to host
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  ! ... might be more (or less) than two memcopies 
  ! TODO Compare results
  ! TODO Update error code if the results do not match
  return errorCode
end function


program test_zhegvdx_gpu_kernels
  implicit none
  integer :: globalErrorCode = 0, errorCode, fails = 0, tests = 0
  ! declare test functions and return type
  integer :: test_launch_krnl_959801_0_auto
  write(*,*) "SUITE test_zhegvdx_gpu_kernels run ..."
  errorCode = test_launch_krnl_959801_0_auto()
  IF (errorCode > 0) THEN
    fails = fails + 1
    write(*,*) "TEST test_launch_krnl_959801_0_auto ... FAILURE"
  ELSE 
    write(*,*) "TEST test_launch_krnl_959801_0_auto ... SUCCESS"
  END IF
  tests = tests + 1
  globalErrorCode = globalErrorCode + errorCode

  IF (globalErrorCode > 0) THEN
    write(*,*) "SUITE test_zhegvdx_gpu_kernels ... FAILURE passed:",(tests-fails)," failed:",fails," total:",tests
  ELSE 
    write(*,*) "SUITE test_zhegvdx_gpu_kernels ... SUCCESS passed:",(tests-fails)," failed:",fails," total:",tests
  END IF
end program test_zhegvdx_gpu_kernels