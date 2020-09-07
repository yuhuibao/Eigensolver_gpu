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
   use hip
   use iso_c_binding
   use iso_c_binding_ext
   use hipblas

   use cusolverDn
   integer                        :: initialized = 0
   type(cublasHandle)             :: cuHandle
   type(cusolverDnHandle)         :: cusolverHandle
   type(cudaEvent)                :: event1, event2, event3
   type(c_ptr) :: stream1 = c_null_ptr, stream2 = c_null_ptr, stream3 = c_null_ptr
   type(c_ptr) :: devInfo_d = c_null_ptr
   type(c_ptr) :: finished = c_null_ptr
   integer(c_int) :: finished_n1, finished_lb1

contains

   subroutine init_eigsolve_gpu()
      use hip
      use iso_c_binding
      use iso_c_binding_ext
      use hipblas

      type(c_ptr) :: hipblasHandle = c_null_ptr

      implicit none
      integer istat
      if (initialized == 0) then
         ! Configure shared memory to use 8 byte banks
         istat = hipDeviceSetSharedMemConfig(hipSharedMemBankSizeEightByte)

         hipblasCreate(hipblasHandle)
         istat = hipblasCreate(hipblasHandle, cuHandle)
         istat = cusolverDnCreate(cusolverHandle)
         istat = hipStreamCreate(stream1)
         istat = hipStreamCreate(stream2)
         istat = hipStreamCreate(stream3)
         istat = hipEventCreate(event1)
         istat = hipEventCreate(event2)
         hipblasDestroy(hipblasHandle)

         initialized = 1
         CALL hipCheck(hipMalloc(finished, 1_8*(4)*(1)))
         finished_n1 = 1
         finished_lb1 = 1
         finished(1) = 0
      endif
   end subroutine init_eigsolve_gpu

end module eigsolve_vars
