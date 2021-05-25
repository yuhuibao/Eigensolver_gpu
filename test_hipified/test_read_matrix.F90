program main
    use utils
    implicit none

    character(len=40)                       :: f1, f2
    real(8), dimension(:,:), allocatable    :: A
    real(8), dimension(:), allocatable      :: vec

    f1 = "mat1.dat"
    call read_matrix_from_file(f1, A)
    call print_matrix(A)

    f2 = "vec.dat"
    call read_vector_from_file(f2, vec)
    call print_vector(vec)

end program