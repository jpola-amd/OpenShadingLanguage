#ifndef HIPRENDERER_HPP
#define HIPRENDERER_HPP

#include "simplerenderer.hpp"
#include <hip/hip_runtime.h>


class HIPRenderer : public SimpleRenderer
{
public:
    HIPRenderer();
    virtual ~HIPRenderer();

    int supports(OIIO::string_view feature) const override;

private:
    hipDeviceProp_t m_deviceProperties;
    hipStream_t m_stream;
    hipModule_t m_module;
    hipFunction_t m_function;

   

};


#endif // HIPRENDERER_HPP