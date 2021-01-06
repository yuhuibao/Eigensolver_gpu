!
! Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
!
!
! Permission is hereby granted, free of charge, to any person obtaining a
! copy of this software and associated documentation files (the "Software"),
! to deal in the Software without restriction, including without limitation
! the rights to use, copy, modify, merge, publish, distribute, sublicense,
! and/or sell copies of the Software, and to permit persons to whom the
! Software is furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
! THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
! FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
! DEALINGS IN THE SOFTWARE.
!

! Module containing various handles used for GPU eigensolver
module eigsolve_vars
    use hipfort
    use iso_c_binding
    use iso_c_binding_ext
    use hipfort_hipblas
    use hipfort_check

    use hipfort_rocblas
    integer                        :: initialized = 0
    type(c_ptr)             :: hipblasHandle
    type(c_ptr)         :: rocsolverHandle
    type(c_ptr) :: event1, event2, event3
    type(c_ptr) :: stream1, stream2, stream3
    integer, pointer :: devInfo_d(:)
    integer, pointer :: finished(:)
    integer, target :: hfinished(1)

contains

    subroutine init_eigsolve_gpu()

        implicit none
        integer :: istat
        hipblasHandle = c_null_ptr
        if (initialized == 0) then
            ! Configure shared memory to use 8 byte banks
            call hipCheck(hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte))

            call hipblasCheck(hipblasCreate(hipblasHandle))
            call rocsolverCheck(rocblas_create_handle(rocsolverHandle))
            call hipCheck(hipStreamCreate(stream1))
            call hipCheck(hipStreamCreate(stream2))
            call hipCheck(hipStreamCreate(stream3))
            call hipCheck(hipEventCreate(event1))
            call hipCheck(hipEventCreate(event2))

            initialized = 1
            CALL hipCheck(hipMalloc(finished, 1))
            call hipCheck(hipMalloc(devInfo_d, 1))
            hfinished(1) = 0
            call hipCheck(hipMemcpy(finished, hfinished, 1, hipMemcpyHostToDevice))
        endif
    end subroutine init_eigsolve_gpu

end module eigsolve_vars
