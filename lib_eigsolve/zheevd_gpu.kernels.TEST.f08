! This file was generated by gpufort

 
! 
! Hints:
! Device variables in scope:
!       character                                   :: uplo, jobz

!       character                                   :: uplo, jobz

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: n, nz, lda, lwork, lrwork, liwork, istat, info

!       integer                                     :: lwork_h, lrwork_h, liwork_h, ldz_h

!       integer                                     :: lwork_h, lrwork_h, liwork_h, ldz_h

!       integer                                     :: lwork_h, lrwork_h, liwork_h, ldz_h

!       integer                                     :: lwork_h, lrwork_h, liwork_h, ldz_h

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       integer                                     :: i, j, k, nb1, nb2, ib, mi, ldt, ldz, il, iu

!       real(8), dimension(1:lrwork), device        :: rwork

!       complex(8), dimension(1:lwork), device      :: work

!       complex(8), dimension(1:lda, 1:n), device   :: a

!       complex(8), dimension(1:ldz, 1:n), device   :: z

!       real(8), dimension(1:n), device             :: w

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       integer                                     :: inde, indtau, indwrk, indrwk, indwk2, indwk3, llwork, llrwk

!       complex(8), parameter                       :: cone = cmplx(1,0,8)

!       real(8), parameter                          :: one = 1.0_8

function test_launch_krnl_e26a05_0_auto()
  ! errorCode > 0 implies that the test has failed
  use iso_c_binding
  use hip
  use zheevd_gpu_kernels
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
  CALL launch_krnl_e26a05_0_auto(0,c_null_ptr,sharedMem,stream,z,z_n1,z_n2,z_lb1,z_lb2,n,a,a_n1,a_n2,a_lb1,a_lb2) ! Modify sharedMem if other than default 0
  CALL hipCheck(hipDeviceSynchronize())
  CALL launch_krnl_e26a05_0_cpu(0,c_null_ptr,sharedMem,stream,z,z_n1,z_n2,z_lb1,z_lb2,n,a,a_n1,a_n2,a_lb1,a_lb2)

  ! TODO Copy results back to host
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  ! ... might be more (or less) than two memcopies 
  ! TODO Compare results
  ! TODO Update error code if the results do not match
  return errorCode
end function

 
! 
! Hints:
! Device variables in scope:
!       integer                               :: n, k, ldv, ldt, ldw

!       integer                               :: n, k, ldv, ldt, ldw

!       integer                               :: n, k, ldv, ldt, ldw

!       integer                               :: n, k, ldv, ldt, ldw

!       integer                               :: n, k, ldv, ldt, ldw

!       complex(8), dimension(ldv, k), device :: v

!       complex(8), dimension(k), device      :: tau

!       complex(8), dimension(ldt, k), device :: t

!       complex(8), dimension(ldw, k), device :: w

!       integer                               :: i, j, istat

!       integer                               :: i, j, istat

!       integer                               :: i, j, istat

!       type(dim3)                            :: threads

function test_launch_krnl_b1ccc1_1_auto()
  ! errorCode > 0 implies that the test has failed
  use iso_c_binding
  use hip
  use zheevd_gpu_kernels
  implicit none
  integer :: errorCode = 1
  ! TODO fix parameters
  ! - Add missing arguments
  ! - Determine size of arrays (typically indicated by 'type(c_ptr)' type)
  ! - Add target where we need a pointer
  integer(c_int),intent(IN) :: sharedMem
  type(c_ptr),value,intent(IN) :: stream
  INTEGER(kind=),value :: k
  INTEGER(kind=),value :: n
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
  ! TODO Create initial data on host
  ! TODO Copy data to device ! (dest,src,size,direction)
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  ! ... might be more (or less) than two memcopies 
  ! TODO run the test
  CALL launch_krnl_b1ccc1_1_auto(0,c_null_ptr,sharedMem,stream,k,n,w,w_n1,w_n2,w_lb1,w_lb2,v,v_n1,v_n2,v_lb1,v_lb2) ! Modify sharedMem if other than default 0
  CALL hipCheck(hipDeviceSynchronize())
  CALL launch_krnl_b1ccc1_1_cpu(0,c_null_ptr,sharedMem,stream,k,n,w,w_n1,w_n2,w_lb1,w_lb2,v,v_n1,v_n2,v_lb1,v_lb2)

  ! TODO Copy results back to host
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  ! ... might be more (or less) than two memcopies 
  ! TODO Compare results
  ! TODO Update error code if the results do not match
  return errorCode
end function

 
! 
! Hints:
! Device variables in scope:
!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: m, n, k, ldv, ldt, ldc, ldw, ldwork, istat

!       integer                                  :: i, j

!       integer                                  :: i, j

!       complex(8), dimension(ldv, k), device    :: v

!       complex(8), dimension(ldt, k), device    :: t

!       complex(8), dimension(ldw, k), device    :: w

!       complex(8), dimension(ldc, n), device    :: c

!       complex(8), dimension(ldwork, k), device :: work

function test_launch_krnl_b95769_2_auto()
  ! errorCode > 0 implies that the test has failed
  use iso_c_binding
  use hip
  use zheevd_gpu_kernels
  implicit none
  integer :: errorCode = 1
  ! TODO fix parameters
  ! - Add missing arguments
  ! - Determine size of arrays (typically indicated by 'type(c_ptr)' type)
  ! - Add target where we need a pointer
  integer(c_int),intent(IN) :: sharedMem
  type(c_ptr),value,intent(IN) :: stream
  INTEGER(kind=),value :: m
  INTEGER(kind=),value :: k
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
  ! TODO Create initial data on host
  ! TODO Copy data to device ! (dest,src,size,direction)
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  ! ... might be more (or less) than two memcopies 
  ! TODO run the test
  CALL launch_krnl_b95769_2_auto(0,c_null_ptr,sharedMem,stream,m,k,w,w_n1,w_n2,w_lb1,w_lb2,v,v_n1,v_n2,v_lb1,v_lb2) ! Modify sharedMem if other than default 0
  CALL hipCheck(hipDeviceSynchronize())
  CALL launch_krnl_b95769_2_cpu(0,c_null_ptr,sharedMem,stream,m,k,w,w_n1,w_n2,w_lb1,w_lb2,v,v_n1,v_n2,v_lb1,v_lb2)

  ! TODO Copy results back to host
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  ! ... might be more (or less) than two memcopies 
  ! TODO Compare results
  ! TODO Update error code if the results do not match
  return errorCode
end function

 
! Fortran implementation:
!     attributes(global) subroutine finish_T_block_kernel(N, T, ldt, tau)
!        implicit none
!        integer, value                        :: N, ldt
!        complex(8), dimension(ldt, K), device :: T
!        complex(8), dimension(K), device      :: tau
!        ! T_s contains only lower triangular elements of T in linear array, by row
!        complex(8), dimension(2080), shared   :: T_s
!        ! (i,j) --> ((i-1)*i/2 + j)
! #define IJ2TRI(i,j) (ISHFT((i-1)*i,-1) + j)
! 
!        integer     :: tid, tx, ty, i, j, k, diag
!        complex(8)  :: cv
! 
!        tx = threadIdx%x
!        ty = threadIdx%y
!        tid = (threadIdx%y - 1)*blockDim%x + tx ! Linear thread id
! 
!        ! Load T into shared memory
!        if (tx <= N) then
!           do j = ty, N, blockDim%y
!              cv = tau(j)
!              if (tx > j) then
!                 T_s(IJ2TRI(tx, j)) = -cv*T(tx, j)
!              else if (tx == j) then
!                 T_s(IJ2TRI(tx, j)) = cv
!              endif
!           end do
!        end if
! 
!        call syncthreads()
! 
!        ! Perform column by column update by first thread column
!        do i = N - 1, 1, -1
!           if (ty == 1) then
!              if (tx > i .and. tx <= N) then
!                 cv = cmplx(0, 0)
!                 do j = i + 1, tx
!                    cv = cv + T_s(IJ2TRI(j, i))*T_s(IJ2TRI(tx, j))
!                 end do
!              endif
! 
!           endif
! 
!           call syncthreads()
!           if (ty == 1 .and. tx > i .and. tx <= N) then
!              T_s(IJ2TRI(tx, i)) = cv
!           endif
!           call syncthreads()
! 
!        end do
! 
!        call syncthreads()
! 
!        ! Write T_s to global
!        if (tx <= N) then
!           do j = ty, N, blockDim%y
!              if (tx >= j) then
!                 T(tx, j) = T_s(IJ2TRI(tx, j))
!              endif
!           end do
!        end if
! 
!     end subroutine finish_T_block_kernel
! 
function test_launch_finish_t_block_kernel()
  ! errorCode > 0 implies that the test has failed
  use iso_c_binding
  use hip
  use zheevd_gpu_kernels
  implicit none
  integer :: errorCode = 1
  ! TODO fix parameters
  ! - Add missing arguments
  ! - Determine size of arrays (typically indicated by 'type(c_ptr)' type)
  ! - Add target where we need a pointer
  type(dim3,,intent(IN, :: grid
  type(dim3,,intent(IN, :: block
  integer(c_int,,intent(IN, :: sharedMem
  type(c_ptr,,value,intent(IN, :: stream
  INTEGER(kind=),value :: n
  INTEGER(kind=),value :: ldt
  type(c_ptr),value :: _t
  integer(c_int),value,intent(IN) :: t_n1
  integer(c_int),value,intent(IN) :: t_n2
  integer(c_int),value,intent(IN) :: t_lb1
  integer(c_int),value,intent(IN) :: t_lb2
  type(c_ptr),value :: _tau
  integer(c_int),value,intent(IN) :: tau_n1
  integer(c_int),value,intent(IN) :: tau_lb1
  ! TODO Create initial data on host
  ! TODO Copy data to device ! (dest,src,size,direction)
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  CALL hipCheck(hipMemcpy(???,c_loc(???),C_SIZEOF(???),hipMemcpyHostToDevice)) ! 
  ! ... might be more (or less) than two memcopies 
  ! TODO run the test
  CALL launch_finish_t_block_kernel(0,c_null_ptr,grid,block,sharedMem,stream,n,ldt,t,t_n1,t_n2,t_lb1,t_lb2,tau,tau_n1,tau_lb1) ! Modify sharedMem if other than default 0
  CALL hipCheck(hipDeviceSynchronize())
  CALL launch_finish_t_block_kernel_cpu(0,c_null_ptr,grid,block,sharedMem,stream,n,ldt,t,t_n1,t_n2,t_lb1,t_lb2,tau,tau_n1,tau_lb1)

  ! TODO Copy results back to host
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  CALL hipCheck(hipMemcpy(c_loc(???),???,C_SIZEOF(???),hipMemcpyDeviceToHost)
  ! ... might be more (or less) than two memcopies 
  ! TODO Compare results
  ! TODO Update error code if the results do not match
  return errorCode
end function


program test_zheevd_gpu_kernels
  implicit none
  integer :: globalErrorCode = 0, errorCode, fails = 0, tests = 0
  ! declare test functions and return type
  integer :: test_launch_krnl_e26a05_0_auto
  integer :: test_launch_krnl_b1ccc1_1_auto
  integer :: test_launch_krnl_b95769_2_auto
  integer :: test_launch_finish_t_block_kernel
  write(*,*) "SUITE test_zheevd_gpu_kernels run ..."
  errorCode = test_launch_krnl_e26a05_0_auto()
  IF (errorCode > 0) THEN
    fails = fails + 1
    write(*,*) "TEST test_launch_krnl_e26a05_0_auto ... FAILURE"
  ELSE 
    write(*,*) "TEST test_launch_krnl_e26a05_0_auto ... SUCCESS"
  END IF
  tests = tests + 1
  globalErrorCode = globalErrorCode + errorCode
  errorCode = test_launch_krnl_b1ccc1_1_auto()
  IF (errorCode > 0) THEN
    fails = fails + 1
    write(*,*) "TEST test_launch_krnl_b1ccc1_1_auto ... FAILURE"
  ELSE 
    write(*,*) "TEST test_launch_krnl_b1ccc1_1_auto ... SUCCESS"
  END IF
  tests = tests + 1
  globalErrorCode = globalErrorCode + errorCode
  errorCode = test_launch_krnl_b95769_2_auto()
  IF (errorCode > 0) THEN
    fails = fails + 1
    write(*,*) "TEST test_launch_krnl_b95769_2_auto ... FAILURE"
  ELSE 
    write(*,*) "TEST test_launch_krnl_b95769_2_auto ... SUCCESS"
  END IF
  tests = tests + 1
  globalErrorCode = globalErrorCode + errorCode
  errorCode = test_launch_finish_t_block_kernel()
  IF (errorCode > 0) THEN
    fails = fails + 1
    write(*,*) "TEST test_launch_finish_t_block_kernel ... FAILURE"
  ELSE 
    write(*,*) "TEST test_launch_finish_t_block_kernel ... SUCCESS"
  END IF
  tests = tests + 1
  globalErrorCode = globalErrorCode + errorCode

  IF (globalErrorCode > 0) THEN
    write(*,*) "SUITE test_zheevd_gpu_kernels ... FAILURE passed:",(tests-fails)," failed:",fails," total:",tests
  ELSE 
    write(*,*) "SUITE test_zheevd_gpu_kernels ... SUCCESS passed:",(tests-fails)," failed:",fails," total:",tests
  END IF
end program test_zheevd_gpu_kernels