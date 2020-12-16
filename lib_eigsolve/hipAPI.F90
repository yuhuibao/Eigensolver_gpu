program test_api
    use hipfort

    !type(c_ptr) :: handle
    type(c_ptr) :: event2, event1
    type(c_ptr) :: stream2, stream1
    integer :: istat
    !call hipblasCheck(hipblasCreate(handle))
    istat = hipStreamCreate(stream1)
    istat = hipStreamCreate(stream2)
    !call hipCheck(hipEventCreate(event1))
    istat = hipEventCreate(event2)

    istat = hipEventRecord(event2, stream1)
    ! write (*, *), "reach here"
    ! if (.not. c_associated(handle)) then
    !     write (*, *), "handle changes"
    ! endif
    ! call hipblasCheck(hipblasSetStream(handle, stream1))
end program
