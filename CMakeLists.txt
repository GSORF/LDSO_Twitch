cmake_minimum_required(VERSION 2.8)
project(ldso_twitch)

# Check for the Build Type and set it if necessary
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of the build." FORCE)
    message(STATUS "Setting the build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Check for the C++ Standard
if(${CMAKE_VERSION} VERSION_LESS "3.2.0")
    set( CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
else()
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Custom modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Binary and library
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

# required libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
find_package(Pangolin REQUIRED)

# optional libraries
find_package(LibZip QUIET)

# Set additional compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wno-deprecated -march=native -Wno-duplicate-decl-specifier -Wno-ignored-qualifiers -Wno-reorder -Wno-missing-braces")

if(NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

# Decision if we have ZipLib:
if(LIBZIP_LIBRARY)
    message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
    add_definitions(-DHAS_ZIPLIB=1)
    include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} )
else()
    message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
    set(LIBZIP_LIBRARY "")
endif()

# Add the include directories
include_directories(
    ${EIGEN3_INCLUDE_DIR} # Linear algebra library
    ${OpenCV_INCLUDE_DIR} # Image processing library
    ${Glog_INCLUDE_DIR}  # Logging library
    ${Pangolin_INCLUDE_DIR}  # Visualization library

    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/
    ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/
)

set( THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${GLOG_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/thirdparty/DBoW3/build/src/libDBoW3${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${PROJECT_SOURCE_DIR}/thirdparty/g2o/lib/libg2o${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${LIBZIP_LIBRARY}
)

# The main source code of LDSO_Twitch
add_subdirectory(src) # Contains a separate CMakeLists.txt
add_subdirectory(examples) # Contains a separate CMakeLists.txt



