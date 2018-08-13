cmake_minimum_required(VERSION 3.12)
project(bonded_molecular_conformation)

set(CMAKE_CXX_STANDARD 11)

include(ExternalProject)
ExternalProject_Add(lbfgs
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/liblbfgs-1.10
        CONFIGURE_COMMAND ${CMAKE_SOURCE_DIR}/liblbfgs-1.10/configure --prefix=${CMAKE_BINARY_DIR}
        BUILD_COMMAND ${MAKE})
add_library(liblbfgs STATIC IMPORTED GLOBAL)
set_property(TARGET liblbfgs PROPERTY IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/lib/liblbfgs.a)
include_directories(${CMAKE_SOURCE_DIR}/liblbfgs-1.10/include)

add_executable(bonded_molecular_conformation main.cpp)
target_link_libraries(bonded_molecular_conformation liblbfgs)
add_dependencies(liblbfgs lbfgs)
