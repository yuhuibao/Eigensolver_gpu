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
      print*, " ",A(j)
    end do
  end subroutine

end module utils
