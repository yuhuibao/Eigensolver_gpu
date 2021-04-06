module funcs
contains

    ! Creates pseudo-random positive-definite symmetric matrix
    subroutine create_random_symmetric_pd(A, N)
        use hipfort
        use hipfort_hipblas
        use hipfort_check
        real(8), allocatable, dimension(:, :), target         :: A, temp
        real(8), pointer, dimension(:,:) :: A_d, temp_d
        !type(c_ptr) :: A_d, temp_d
        real(8)                                         :: rv
        integer                                         :: i, j, N, istat
        type(c_ptr) :: hipblasHandle = c_null_ptr

        allocate (A(N, N))
        allocate (temp(N, N))

        ! Create general symmetric temp
        do j = 1, N
            do i = 1, N
                if (i > j) then
                    call random_number(rv)
                    temp(i, j) = rv
                    temp(j, i) = rv
                else if (i == j) then
                    call random_number(rv)
                    temp(i, j) = rv
                end if
            end do
        end do

        !allocate(A_d, source = A)
        !allocate(temp_d, source = temp)
        call hipCheck(hipMalloc(A_d, N,N))
        call hipCheck(hipMemcpy(A_d, A, N*N, hipMemcpyHostToDevice))
        call hipCheck(hipMalloc(temp_d, N,N))
        call hipCheck(hipMemcpy(temp_d, temp, N*N, hipMemcpyHostToDevice))
        ! Multiply temp by transpose of temp to get positive definite A
        istat = hipblasCreate(hipblasHandle)
        istat = hipblasdgemm(hipblasHandle, HIPBLAS_OP_T, HIPBLAS_OP_N, N, N, N, 1.d0, temp_d, N, temp_d, N, 0.d0, A_d, N)

        call hipCheck(hipMemcpy(A, A_d, N*N, hipMemcpyHostToDevice))
        deallocate (temp)
        call hipCheck(hipFree(A_d))
        call hipCheck(hipFree(temp_d))
        istat = hipblasDestroy(hipblasHandle)
    end subroutine
end module funcs
program main
    use hipfort
    use hipfort_hipblas
    use hipfort_rocsolver
    use eigsolve_vars, ONLY: init_eigsolve_gpu
    use dsygvdx_gpu
    use funcs
    use hipfort_check
    use utils
    implicit none
    integer(c_int), parameter :: bytes_per_element = 8 !double precision
    integer                                         :: N, M, i, j, info, lda, istat
    integer                                         :: n1, n2, m1, m2, lda1, lda2
    integer                                         :: lwork_d, lwork, lrwork, liwork, il, iu,lwork1,liwork1
    character(len=20)                               :: arg
    character(:),allocatable                        :: file1,file2
    real(8)                                         :: ts, te, wallclock
    real(8), dimension(:, :), allocatable, target            :: A1, A2, Aref
    real(8), dimension(:, :), allocatable, target            :: B1, B2, Bref
    real(8),pointer, dimension(:,:)    :: Z2
    
    real(8),pointer, dimension(:,:)    :: A2_d, B2_d, Z2_d
    
    real(8),pointer, dimension(:)      :: work
    real(8),pointer, dimension(:)      :: w1, w2
    integer,pointer, dimension(:)      :: iwork
    real(8),pointer, dimension(:)      :: work_d
    
    real(8),pointer, dimension(:)      :: w2_d

    
    i = command_argument_count()

    if (i == 1) then
        ! If N is provided, generate random symmetric matrices for A and B
        print *, "Using randomly-generated matrices..."
        call get_command_argument(1, arg)
        read (arg, *) N
        lda = N

        ! Create random positive-definite hermetian matrices on host
        call create_random_symmetric_pd(Aref, N)
        !call print_matrix(Aref)
        call create_random_symmetric_pd(Bref, N)

    
    elseif (i ==2) then
            print*, "Reading  matrices from files ..."
            print*, "Unformatted files with n,m,lda "
            print*, "A(lda,n) B(lda,n)"
            call get_command_argument(1, arg)
            file1=trim(arg)
            call get_command_argument(2, arg)
            file2=trim(arg)
            open(UNIT=13, FILE=file1, ACTION="read")
            open(UNIT=14, FILE=file2, ACTION="read")
            read(13,*) n1,m1,lda1
            read(14,*) n2,m2,lda2
            print *,n1,m1,lda1,n2,m2,lda2
            if( n1/=n2 .or. m1/=m2 .or. lda1 /= lda2) then
              print *,"expecting A and B to have same N,M,LDA"
              call exit
            end if
            N=n1
            M=m1
            LDA=lda1
            print *,"n,m,lda from files:",n,m,lda
            allocate(Aref(n,m))
            allocate(Bref(n,m))
            read(13,*)Aref(1:n,1:n)
            read(14,*)Bref(1:n,1:n)
            close(13)
            close(14)
            print*,"Aref"
            call print_matrix(Aref)
            print*,"Bref"
            call print_matrix(Bref)
        
          else
            print*, "Usage:\n\t ./main [N]"
            call exit
          endif

    print *, "Running with N = ", N
    
   ! Allocate/Copy matrices to device
    allocate (A1, source=Aref)
    allocate (A2, source=Aref)
    
    call hipCheck(hipMalloc(A2_d, lda,N))
    call hipCheck(hipMemcpy(A2_d, Aref, lda*N, hipMemcpyHostToDevice))
    allocate (B1, source=Bref)
    allocate (B2, source=Bref)
    
    call hipCheck(hipMalloc(B2_d, lda,N))
    call hipCheck(hipMemcpy(B2_d, Bref, lda*N, hipMemcpyHostToDevice))

    call hipCheck(hipHostMalloc(Z2, lda,N,0))
    Z2 = Aref

    call hipCheck(hipMalloc(Z2_d, lda,N))
    call hipCheck(hipMemcpy(Z2_d, Z2, lda*N, hipMemcpyHostToDevice))

    call hipCheck(hipHostMalloc(w1, N,0))
    call hipCheck(hipHostMalloc(w2, N,0))
    
    call hipCheck(hipMalloc(w2_d, N))
    call hipCheck(hipMemset(c_loc(w2_d), 0, N*8_8))

    ! Initialize solvers
  call init_eigsolve_gpu()

  !! Solving generalized eigenproblem using DSYGVD
  ! CASE 1: CPU _____________________________________________
  print*
  print*, "CPU_____________________"
  lwork = 1 + 6*N + 2*N*N
  liwork = 3 + 5*N
  call hipCheck(hipHostMalloc(work, lwork,0))
  call hipCheck(hipHostMalloc(iwork, liwork,0))
  call dsygvd(1, 'V', 'U', N, A1, lda, B1, lda, w1, work, -1, iwork, -1, istat)
  if (istat /= 0) write(*,*) 'CPU dsygvd worksize failed'
  lwork = work(1);; liwork = iwork(1)
  
  !call print_vector(iwork,liwork)
  call hipCheck(hipHostFree(work))
  call hipCheck(hipHostFree(iwork))
  call hipCheck(hipHostMalloc(work, lwork,0))
  call hipCheck(hipHostMalloc(iwork, liwork,0))

  A1 = Aref
  B1 = Bref
  ! Run once before timing
  call dsygvd(1, 'V', 'U', N, A1, lda, B1, lda, w1, work, lwork, iwork, liwork, istat)
  if (istat /= 0) write(*,*) 'CPU dsygvd failed. istat = ', istat

  A1 = Aref
  B1 = Bref
  ts = wallclock()
  call dsygvd(1, 'V', 'U', N, A1, lda, B1, lda, w1, work, lwork, iwork, liwork, istat)
  te = wallclock()
  if (istat /= 0) write(*,*) 'CPU dsygvd failed. istat = ', istat

  print*, "\tTime for CPU dsygvd = ", (te - ts)*1000.0
  print*

! CASE 4: using CUSTOM ____________________________________________________________________
  print*
  print*, "CUSTOM_____________________"
  iu = N
  il = 1
  lwork = 1 + 6*N + 2*N*N
  liwork = 3 + 5*N
  call hipCheck(hipHostFree(work))
  call hipCheck(hipHostFree(iwork))

  call hipCheck(hipHostMalloc(work, lwork,0))
  call hipCheck(hipHostMalloc(iwork, liwork,0))

  lwork_d = 2*64*64 + 66*N
  call hipCheck(hipMalloc(work_d, lwork_d))
  ts = wallclock()
  call dsygvdx_gpu_h(N, A2_d,A2, lda, B2_d, B2, lda, Z2_d, lda, il, iu, w2_d, work_d, lwork_d, &
                     work, lwork, iwork, liwork, Z2, lda, w2, istat)
  te = wallclock()
  print *, "Time for CUSTOM dsygvd/x = ", (te - ts)*1000.0
  if (istat /= 0) write (*, *) 'dsygvdx_gpu failed'

  call print_matrix(A1)
  call print_matrix(Z2)
end program
