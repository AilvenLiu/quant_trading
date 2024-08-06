cmake_minimum_required(VERSION 3.10)

project(QuantTrading)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Define paths for output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    include
    api/include
)

# Library directories
link_directories(
    api/lib
)

# Source files
set(SOURCE_FILES
    src/main.cpp
    src/Logger.cpp
    modules/data/cpp/ReadTimeData.cpp
    modules/data/cpp/SharedMemoryServer.cpp
    modules/data/cpp/Logger.cpp
)

# Create executable
add_executable(QuantTrading ${SOURCE_FILES})

# Link libraries (assuming libib_tws.dylib is the IB API library)
target_link_libraries(QuantTrading PRIVATE ib_tws boost_system)

# Include Boost library paths if needed (assuming Boost is installed)
find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(QuantTrading ${Boost_LIBRARIES})

# Create a shared library (if needed, example here)
# Uncomment the following lines if you want to build a shared library
# add_library(MyLibrary SHARED ${SOURCE_FILES})
# target_link_libraries(MyLibrary PRIVATE ib_tws boost_system)
