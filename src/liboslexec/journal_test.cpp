// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <OSL/encodedtypes.h>
#include <OpenImageIO/unittest.h>

#include <array>
#include <cstring>
#include <vector>

using namespace OSL;

template<typename... Args>
void
check(string_view format, string_view expected, bool expected_valid,
      Args... args)
{
    const std::array<EncodedType, sizeof...(Args)> types {
        pvt::TypeEncoder<Args>::value...
    };
    std::vector<uint8_t> bytes;
    auto append = [&](auto arg) {
        const auto value  = pvt::TypeEncoder<decltype(arg)>::Encode(arg);
        const auto offset = bytes.size();
        bytes.resize(offset + sizeof(value));
        memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    (append(args), ...);
    bool valid          = false;
    std::string message = "stale";
    const auto hash     = ustring(format).hash();
    const int consumed  = decode_message(hash, int(types.size()), types.data(),
                                         bytes.data(), message, valid);
    OIIO_CHECK_EQUAL(message, expected);
    OIIO_CHECK_EQUAL(valid, expected_valid);
    if (valid)
        OIIO_CHECK_EQUAL(consumed, int(bytes.size()));
    std::string legacy;
    OIIO_CHECK_EQUAL(decode_message(hash, int(types.size()), types.data(),
                                    bytes.data(), legacy),
                     consumed);
    OIIO_CHECK_EQUAL(legacy, message);
}



int
main()
{
    check("", "", true);
    check("plain {{literal}} 100%", "plain {literal} 100%", true);
    check("{:d}|{:s}|{:.2f}|{:x}|{:o}", "-7|MiXeD|0.25|ffffffff|377", true,
          int32_t(-7), ustring("MiXeD"), .25f, uint32_t(-1), uint32_t(255));
    check("[{:s}][{:s}]", "[][]", true, ustring(), ustring(""));
    check("{:s}", std::string(1024, 'x'), true, std::string(1024, 'x'));
    check("{:s}", std::string(1024, 'x'), false, std::string(1025, 'x'));
    check("{:1024d}", std::string(1023, ' ') + "7", true, int32_t(7));
    check("{:1025d}", std::string(1024, ' '), false, int32_t(7));
    check("{:d}", "", false);
    check("unused", "unused", false, int32_t(7));
    check("unclosed {:d", "unclosed {:d", false, int32_t(7));
    check("}", "}", false);

    const EncodedType type = EncodedType::kInt32;
    const int32_t value    = 7;
    bool valid             = true;
    std::string message;
    decode_message(ustring("{:q}").hash(), 1, &type,
                   reinterpret_cast<const uint8_t*>(&value), message, valid);
    OIIO_CHECK_ASSERT(!valid);
    OIIO_CHECK_ASSERT(message.find("MIS-FORMAT") != std::string::npos);
    OIIO_CHECK_ASSERT(message.size() <= 1024);
    return unit_test_failures;
}
