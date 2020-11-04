module funcs
    contains
    subroutine print_matrix(A)
             
             real(8),dimension(:,:) :: A
             do j = 1,size(A,2)
              do i = 1,size(A,1)
                 write(*,*) "a(",i,",",j,") = ", a(i,j)
               end do
             end do
           end subroutine
           subroutine print_vector(A)
        
             integer,dimension(:) :: A
             do j = 1,size(A)
               print*, " ",A(j)
             end do
           end subroutine
end module funcs