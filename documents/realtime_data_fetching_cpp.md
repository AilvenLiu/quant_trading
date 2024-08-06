cmake_minimum_required(VERSION 3.25)

# Project name
project(IB_TWS_CPP)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set compiler to Clang (adjust path if needed)
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)

# Compile and link flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -arch arm64")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -arch arm64")

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Find OpenSSL libraries
find_package(OpenSSL REQUIRED)

# Verify Boost presence
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Boost not found!")
endif()

# Verify OpenSSL presence
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found!")
    message(STATUS "OpenSSL include dirs: ${OPENSSL_INCLUDE_DIR}")
    message(STATUS "OpenSSL libraries: ${OPENSSL_LIBRARIES}")
else()
    message(FATAL_ERROR "OpenSSL not found!")
endif()

# Add source files
file(GLOB SOURCES "src/*.cpp")

# Specify output directory for the library
set(LIBRARY_OUTPUT_PATH "lib")

# Create shared library (dynamic library)
add_library(ib_tws_cpp SHARED ${SOURCES})

# Include directories
include_directories(${Boost_INCLUDE_DIRS} ${OPENSSL_INCLUDE_DIR})

# Link Boost and OpenSSL libraries
target_link_libraries(ib_tws_cpp 
    ${Boost_LIBRARIES} 
    ${OPENSSL_LIBRARIES}
)