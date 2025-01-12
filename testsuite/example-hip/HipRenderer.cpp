#if defined(__HIP_DEVICE_COMPILE__)
#warning "This file should not be compiled by HIP"
#endif 

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/imagebuf.h>
#include <OpenImageIO/sysutil.h>

#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/oslexec.h>

#include <OSL/device_string.h>

#include "HipRenderer.hpp"



int
HipRenderer::supports(OIIO::string_view feature) const
{
    // Change here between OptiX and HIP
    if (feature == "HIP") 
    {
        return true;
    }
    return false;
}

void*
HipRenderer::device_alloc(size_t size)
{
     void* ptr       = nullptr;
    hipError_t res = hipMalloc(reinterpret_cast<void**>(&ptr), size);
    if (res != hipSuccess) {
        this->errhandler().errorfmt("hipMalloc({}) failed with error: {}\n", size,
                              hipGetErrorString(res));
    }
    
    return ptr;
}

void
HipRenderer::device_free(void* ptr)
{
    hipError_t res = hipFree(ptr);
    if (res != hipSuccess) {
        errhandler().errorfmt("hipFree() failed with error: {}\n",
                              hipGetErrorString(res));
    }
}

void*
HipRenderer::copy_to_device(void* dst_device, const void* src_host, size_t size)
{
    hipError_t res = hipMemcpy(dst_device, src_host, size,
                                 hipMemcpyHostToDevice);
    if (res != hipSuccess) {
        errhandler().errorfmt(
            "hipMemcpy host->device of size {} failed with error: {}\n", size,
            hipGetErrorString(res));
    }
    return dst_device;
}
