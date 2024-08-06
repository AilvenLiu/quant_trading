cmake_minimum_required(VERSION 3.25)
project(QuantTrading)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/api/include
)

# Library directories
link_directories(
    ${PROJECT_SOURCE_DIR}/api/lib
)

# Add source files
file(GLOB PROJECT_SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/data/*.cpp"
    "${PROJECT_SOURCE_DIR}/src/logger/*.cpp"
)

# Add executable
add_executable(QuantTrading ${PROJECT_SOURCES})

# Find Boost libraries
find_package(Boost REQUIRED COMPONENTS system filesystem)

# Verify Boost presence
if(Boost_FOUND)
    message(STATUS "Boost found: ${Boost_VERSION}")
    message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
    message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
else()
    message(FATAL_ERROR "Boost not found!")
endif()

# Link Boost and IB TWS libraries
target_link_libraries(QuantTrading 
    ${Boost_LIBRARIES}
    ib_tws  # Ensure that libib_tws.dylib is present in api/lib
)

# Set visibility of symbols in the executable
set_target_properties(QuantTrading PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN 1
)

# Installation rules
install(TARGETS QuantTrading
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)