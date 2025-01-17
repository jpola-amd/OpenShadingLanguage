#include "utils.hpp"
#include <iostream>


bool 
setup_crash_stacktrace(OIIO::string_view filename)
{
    bool result = false;
#if defined(OIIO_HAS_STACKTRACE)
    result = OIIO::Sysutil::setup_crash_stacktrace(filename);
#else
    (void)filename;
    std::cerr << "setup_crash_stacktrace() OIIO_HAS_STACKTRACE is not defined\n";
#endif
    return result;
}