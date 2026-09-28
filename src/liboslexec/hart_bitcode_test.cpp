// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <algorithm>
#include <string>

#include <OpenImageIO/filesystem.h>
#include <OpenImageIO/unittest.h>

#include "hart_bitcode.h"


class TestErrorHandler final : public OSL::ErrorHandler {
public:
    void operator()(int code, const std::string& message) override
    {
        OIIO_CHECK_EQUAL(code, EH_ERROR);
        ++errors;
        last_message = message;
    }

    int errors = 0;
    std::string last_message;
};



int
main(int argc, char* argv[])
{
    if (argc != 2 && argc != 3) {
        OSL::print(stderr,
                   "Usage: hart_bitcode_test arch [expected-bitcode]\n");
        return 1;
    }

    TestErrorHandler errhandler;
    const auto bitcode = OSL::pvt::hart_shadeops_bitcode(argv[1], errhandler);
    if (argc == 3) {
        OIIO_CHECK_EQUAL(errhandler.errors, 0);
        OIIO_CHECK_ASSERT(!bitcode.empty());
        OIIO_CHECK_EQUAL(bitcode.size(), OIIO::Filesystem::file_size(argv[2]));
        if (!bitcode.empty()) {
            std::string expected(bitcode.size(), '\0');
            OIIO_CHECK_EQUAL(OIIO::Filesystem::read_bytes(argv[2],
                                                          expected.data(),
                                                          expected.size()),
                             expected.size());
            OIIO_CHECK_ASSERT(std::equal(bitcode.begin(), bitcode.end(),
                                         reinterpret_cast<const unsigned char*>(
                                             expected.data())));
            const auto again = OSL::pvt::hart_shadeops_bitcode(argv[1],
                                                               errhandler);
            OIIO_CHECK_EQUAL(bitcode.data(), again.data());
            OIIO_CHECK_EQUAL(errhandler.errors, 0);
        }
    } else {
        OIIO_CHECK_ASSERT(bitcode.empty());
        OIIO_CHECK_EQUAL(errhandler.errors, 1);
        OIIO_CHECK_EQUAL(
            errhandler.last_message,
            OSL::fmtformat("No embedded HART shadeops for architecture '{}'",
                           argv[1]));
    }
    for (OSL::string_view arch : { "", "gfx9999" }) {
        const int before = errhandler.errors;
        OIIO_CHECK_ASSERT(
            OSL::pvt::hart_shadeops_bitcode(arch, errhandler).empty());
        OIIO_CHECK_EQUAL(errhandler.errors, before + 1);
        OIIO_CHECK_EQUAL(
            errhandler.last_message,
            OSL::fmtformat("No embedded HART shadeops for architecture '{}'",
                           arch));
    }
    return unit_test_failures;
}
