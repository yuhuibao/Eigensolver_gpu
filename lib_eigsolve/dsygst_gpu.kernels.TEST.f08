! This file was generated by gpufort

 
! 
! Hints:
! Device variables in scope:
!       integer, intent(in)                                   :: itype, n, lda, ldb, nb

!       integer, intent(in)                                   :: itype, n, lda, ldb, nb

!       integer, intent(in)                                   :: itype, n, lda, ldb, nb

!       integer, intent(in)                                   :: itype, n, lda, ldb, nb

!       integer, intent(in)                                   :: itype, n, lda, ldb, nb

!       character, intent(in)                                 :: uplo

!       real(8), parameter                                    :: one = 1.d0, half = 0.5d0

!       real(8), parameter                                    :: one = 1.d0, half = 0.5d0

!       integer                                               :: i, j

!       integer                                               :: i, j

!       integer                                               :: k, kb, istat

!       integer                                               :: k, kb, istat

!       integer                                               :: k, kb, istat

function test_launch_krnl_afb01f_0_auto()
  ! errorCode > 0 implies that the test has failed
  use iso_c_binding
  use hip
  use dsygst_gpu_kernels
  implicit none
  integer :: errorCode = 1
  ! TODO fix parameters
  ! - Add missing arguments
  ! - Determine size of arrays (typically indicated by 'type(c_ptr)' type)
  ! - Add target where we need a pointer
  integer(c_int),intent(IN) :: sharedMem
  type(c_ptr),value,intent(IN) :: stream
  integer,value :: kb
  TODO declaration not found :: a
  integer,value :: k
  ! TODO Create initial data on host
  ! TODO Copy data to device ! (dest,src,size,direction)
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  ! ... might be more (or less) than two memcopies 
  ! TODO run the test
  CALL launch_krnl_afb01f_0_auto(0,c_null_ptr,sharedMem,stream,kb,a,k) ! Modify sharedMem if other than default 0
  CALL hipCheck(hipDeviceSynchronize())
  CALL launch_krnl_afb01f_0_cpu(0,c_null_ptr,sharedMem,stream,kb,a,k)

  ! TODO Copy results back to host
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  ! ... might be more (or less) than two memcopies 
  ! TODO Compare results
  ! TODO Update error code if the results do not match
  return errorCode
end function


program test_dsygst_gpu_kernels
  implicit none
  integer :: globalErrorCode = 0, errorCode, fails = 0, tests = 0
  ! declare test functions and return type
  integer :: test_launch_krnl_afb01f_0_auto
  write(*,*) "SUITE test_dsygst_gpu_kernels run ..."
  errorCode = test_launch_krnl_afb01f_0_auto()
  IF (errorCode > 0) THEN
    fails = fails + 1
    write(*,*) "TEST test_launch_krnl_afb01f_0_auto ... FAILURE"
  ELSE 
    write(*,*) "TEST test_launch_krnl_afb01f_0_auto ... SUCCESS"
  END IF
  tests = tests + 1
  globalErrorCode = globalErrorCode + errorCode

  IF (globalErrorCode > 0) THEN
    write(*,*) "SUITE test_dsygst_gpu_kernels ... FAILURE passed:",(tests-fails)," failed:",fails," total:",tests
  ELSE 
    write(*,*) "SUITE test_dsygst_gpu_kernels ... SUCCESS passed:",(tests-fails)," failed:",fails," total:",tests
  END IF
end program test_dsygst_gpu_kernels