program main
    use hipfort
    use hipfort_check
    use utils
    use dsyevd_gpu
    implicit none

    integer                                         :: N, lwork, liwork_h,info,il,iu
    real(8), dimension(:,:), pointer                :: A,Z
    real(8), dimension(:,:), allocatable            :: A_h,Z_h
    real(8), dimension(:), pointer                  :: w, work
    real(8), dimension(:), allocatable              :: work_h, w_h
    integer, dimension(:), allocatable              :: iwork_h

    N = 3
    lwork = 6*N
    liwork_h = 10*N
    iu = n
    il = 1

    allocate(A_h(N,N))
    allocate(Z_h(N,N))
    A_h(1, 1) = 1
    A_h(1, 2) = 2
    A_h(1, 3) = 3
    A_h(2, 1) = 2
    A_h(2, 2) = 4
    A_h(2, 3) = 1
    A_h(3, 1) = 3
    A_h(3, 2) = 1
    A_h(3, 3) = 1

    ! A_h(1, 1) = 1
    ! A_h(1, 2) = 2
    ! A_h(1, 3) = 3
    ! A_h(1, 4) = 4
    ! A_h(2, 1) = 2
    ! A_h(2, 2) = 4
    ! A_h(2, 3) = 1
    ! A_h(2, 4) = 1
    ! A_h(3, 1) = 3
    ! A_h(3, 2) = 1
    ! A_h(3, 3) = 1
    ! A_h(3, 4) = 2
    ! A_h(4, 1) = 4
    ! A_h(4, 2) = 1
    ! A_h(4, 3) = 2
    ! A_h(4, 4) = 2

    ! A_h(1, 1) = 1
    ! A_h(1, 2) = 2
    ! A_h(1, 3) = 3
    ! A_h(1, 4) = 4
    ! A_h(1, 5) = 5
    ! A_h(2, 1) = 2
    ! A_h(2, 2) = 4
    ! A_h(2, 3) = 1
    ! A_h(2, 4) = 1
    ! A_h(2, 5) = 5
    ! A_h(3, 1) = 3
    ! A_h(3, 2) = 1
    ! A_h(3, 3) = 1
    ! A_h(3, 4) = 2
    ! A_h(3, 5) = 3
    ! A_h(4, 1) = 4
    ! A_h(4, 2) = 1
    ! A_h(4, 3) = 2
    ! A_h(4, 4) = 2
    ! A_h(4, 5) = 3
    ! A_h(5, 1) = 5
    ! A_h(5, 2) = 5
    ! A_h(5, 3) = 3
    ! A_h(5, 4) = 3
    ! A_h(5, 1) = 1

    Z_h = A_h
    !allocate(A, source=A_h)
    call hipCheck(hipMalloc(A,N,N))
    call hipCheck(hipMemcpy(A, A_h, N*N, hipMemcpyHostToDevice))

    call hipCheck(hipMalloc(Z,N,N))
    call hipCheck(hipMemcpy(Z, Z_h, N*N, hipMemcpyHostToDevice)) 
    allocate(work_h(lwork))
    !allocate(work, source=work_h)
    call hipCheck(hipMalloc(work,lwork))
    call hipCheck(hipMemset(c_loc(work), 0, lwork*8_8))
    allocate(w_h(N))
    !allocate(d, source=d_h)
    call hipCheck(hipMalloc(w,N))
    call hipCheck(hipMemset(c_loc(w), 0, N*8_8))
    allocate(iwork_h(liwork_h))
    

    call dsyevd_gpu_h('V', 'U', il, iu, N, A, A_h, N, Z, N, w, work, lwork, &
                          work_h, lwork, iwork_h, liwork_h, Z_h, N, w_h, info)

end program