cmake_minimum_required(VERSION 3.25)
project(APILibrary)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Set include directories for API headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add API source files
file(GLOB API_SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Specify the output directory for the dynamic library
set(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

# Create a shared library (dynamic library)
add_library(ib_tws SHARED ${API_SOURCES})

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

# Link Boost and OpenSSL libraries
target_link_libraries(ib_tws 
    ${Boost_LIBRARIES} 
    ${OPENSSL_LIBRARIES}
)

# Optionally, you can define the visibility of symbols in the shared library
set_target_properties(ib_tws PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN 1
)

# Installation rules
install(TARGETS ib_tws
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)