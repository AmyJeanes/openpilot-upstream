cmake_minimum_required(VERSION 3.20)
project(gta5op CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")  # no VC++ runtime DLLs needed in the game folder
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
add_compile_options(/W3 /EHsc /permissive- /utf-8)

# import library for ScriptHookV.dll, from its export names
set(SHV_LIB ${CMAKE_BINARY_DIR}/ScriptHookV.lib)
add_custom_command(OUTPUT ${SHV_LIB}
  COMMAND lib /nologo /machine:x64 /def:${CMAKE_SOURCE_DIR}/ScriptHookV.def /out:${SHV_LIB}
  DEPENDS ${CMAKE_SOURCE_DIR}/ScriptHookV.def)
add_custom_target(shv_lib DEPENDS ${SHV_LIB})

add_library(gta5op SHARED src/loader.cpp)
add_dependencies(gta5op shv_lib)
target_link_libraries(gta5op PRIVATE ${SHV_LIB})
set_target_properties(gta5op PROPERTIES SUFFIX ".asi")

add_library(gta5op_core SHARED src/core.cpp src/capture.cpp src/net.cpp src/present_hook.cpp)
