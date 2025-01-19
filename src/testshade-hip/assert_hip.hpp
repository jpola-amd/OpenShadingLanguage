
#ifndef ASSERT_HIP_HPP
#define ASSERT_HIP_HPP

#include <hip/hip_runtime.h>

#define HIP_CHECK(call)                                               \
    {                                                                 \
        hipError_t error = call;                                      \
        if (error != hipSuccess) {                                    \
            std::stringstream ss;                                     \
            ss << "HIP call (" << #call << " ) failed with error: '"  \
               << hipGetErrorString(error) << "' (" __FILE__ << ":"   \
               << __LINE__ << ")\n";                                  \
            fprintf(stderr, "[HIP ERROR]  %s", ss.str().c_str());     \
            exit(1);                                                  \
        }                                                             \
    }


#define HIP_SYNC_CHECK()                                                     \
    {                                                                        \
        hipError_t error_ = hipDeviceSynchronize();                          \
        hipError_t error = hipGetLastError();                                \
        if (error != hipSuccess) {                                           \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(error));                               \
            exit(1);                                                         \
        }                                                                    \
    }

#define HIPRTC_CHECK(call)                                              \
    {                                                                  \
        hiprtcResult error = call;                                      \
        if (error != HIPRTC_SUCCESS) {                                  \
            std::stringstream ss;                                      \
            ss << "HIPRTC call (" << #call << " ) failed with error: '" \
               << hiprtcGetErrorString(error) << "' (" __FILE__ << ":"  \
               << __LINE__ << ")\n";                                   \
            fprintf(stderr, "[HIPRTC ERROR]  %s", ss.str().c_str());    \
            exit(1);                                                   \
        }                                                              \
    }

#endif // ASSERT_HIP_HPP