#ifndef OSLCOMPILER_HPP
#define OSLCOMPILER_HPP

#include <OpenImageIO/string_view.h>

bool compile_shader(std::string& outputOsoBuffer, const char* oslFilename, const std::vector<std::string>& options, OIIO::string_view stdoslpath = {});

#endif // OSLCOMPILER_HPP