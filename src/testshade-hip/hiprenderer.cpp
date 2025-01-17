#include "hiprenderer.hpp"

//TODO: Add assert to check HIP CALLS
HIPRenderer::HIPRenderer()
{
    hipInit(0);
    hipGetDeviceProperties(&m_deviceProperties, 0);
    hipStreamCreate(&m_stream);
}

HIPRenderer::~HIPRenderer()
{
    hipStreamDestroy(m_stream);
}


int HIPRenderer::supports(OIIO::string_view feature) const
{
    if (feature == "HIP")
        return true;
    return SimpleRenderer::supports(feature);
}

