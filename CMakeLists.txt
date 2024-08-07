cmake_minimum_required(VERSION 3.25)
project(OpenSTX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set include directories for project headers and API headers
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/api/include)

# Add source files for the main project
file(GLOB_RECURSE PROJECT_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Specify the output directories for binaries and libraries
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

# Add executable target for the project
add_executable(QuantTrading ${PROJECT_SOURCES})

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

# Specify the architecture for M1
if (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    message(STATUS "Building for Apple Silicon (arm64)")
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Set the path to the TWS API dynamic library
set(TWS_LIB "${PROJECT_SOURCE_DIR}/api/lib/libib_tws.dylib")

# Link libraries to the executable
target_link_libraries(QuantTrading
    ${TWS_LIB}               # Link TWS API library
    ${Boost_LIBRARIES}       # Link Boost libraries
    ${OPENSSL_LIBRARIES}     # Link OpenSSL libraries
)

# Installation rules
install(TARGETS QuantTrading
    RUNTIME DESTINATION bin
)

# Set compiler options for macOS
if (APPLE)
    # Enforce use of RPATH to locate shared libraries on macOS
    set_target_properties(QuantTrading PROPERTIES
        INSTALL_RPATH "@loader_path/../lib"
    )
endif()