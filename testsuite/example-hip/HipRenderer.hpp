#pragma once

#include <hip/hip_runtime.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>


class HipRenderer final : public OSL::RendererServices
{
private:
    OIIO::ErrorHandler m_errorHandler {OIIO::ErrorHandler::default_handler()};
    OIIO::ErrorHandler& errhandler() { return m_errorHandler; }
public:
    HipRenderer() = default;
    virtual ~HipRenderer() = default;

    virtual int supports(OIIO::string_view feature) const override;

    virtual void* device_alloc(size_t size) override;
    virtual void device_free(void* ptr) override;
    virtual void* copy_to_device(void* dst_device, const void* src_host,
                                 size_t size) override;
};
