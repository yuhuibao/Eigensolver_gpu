module utils
  contains

  subroutine print_matrix(A)
    real(8),dimension(:,:) :: A

    print*, ""

    do i = 1,size(A,1)

      do j = 1,size(A,2)
        write(*, fmt='(1X, A, F0.2)', advance="no") " ", a(i,j)
      end do

      print*, ""
    end do

    print*, ""

  end subroutine

  subroutine print_vector(A)
    real(8),dimension(:) :: A
    do j = 1,size(A)
      write(*, fmt='(1X, A, F0.2)', advance="no") " ", a(j)
    end do
    print*, ""
  end subroutine

  subroutine read_matrix_from_file(filename, matrix)
    integer                            :: n, m, lda
    real(8),dimension(:,:),allocatable :: matrix
    character(len=40)                  :: filename

    open(UNIT=13, FILE=filename, ACTION="read")

    read(13,*) n,m,lda

    allocate(matrix(n,m))

    read(13,*) matrix(1:n,1:n)

    close(13)
  end subroutine read_matrix_from_file

  subroutine read_vector_from_file(filename, vector)
    integer                            :: n
    real(8),dimension(:),allocatable :: vector 
    character(len=40)                  :: filename

    open(UNIT=13, FILE=filename, ACTION="read")

    read(13,*) n

    allocate(vector(n))

    read(13,*) vector(1:n)

    close(13)
  end subroutine read_vector_from_file

end module utils
