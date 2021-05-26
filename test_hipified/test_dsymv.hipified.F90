program main
    use hipfort
    use hipfort_check
    use utils
    !use dsytrd_gpu
    implicit none

    integer                                         :: N,lda
    real(8), dimension(:,:), pointer                :: A
    real(8), dimension(:,:), allocatable            :: A_h
    real(8), dimension(:), pointer                  :: x,y
    real(8), dimension(:), allocatable              :: x_h, y_h
    character(len=40)                               :: f1,f2
    type(dim3)                                      :: threads2D, blocks2D

    f1 = "mat1_32.dat"
    f2 = "vec_32.dat"
    
    call read_matrix_from_file(f1,A_h)
    call read_vector_from_file(f2,x_h)
    N= size(x_h)
    lda=N
    allocate(y_h(N))
    
    call hipCheck(hipMalloc(A,N,N))
    call hipCheck(hipMemcpy(A, A_h, N*N, hipMemcpyHostToDevice))

    call hipCheck(hipMalloc(x,N))
    call hipCheck(hipMemcpy(x, x_h, N, hipMemcpyHostToDevice))
    
    call hipCheck(hipMalloc(y,N))
    call hipCheck(hipMemset(c_loc(y), 0, N*8_8))
    
    threads2D = dim3(16,16,1)
    blocks2D = dim3(10, ceiling(real(N - 1)/16),1)
    call launch_dsymv_gpu_m(blocks2D, threads2D, (16+1)*16*8 + 16*8 + 16*16*8, c_null_ptr, N - 1, lda, A, x, y)

    call hipCheck(hipMemcpy(y_h, y, N, hipMemcpyDeviceToHost))
    call print_vector(y_h)

end program