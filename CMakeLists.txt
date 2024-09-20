cmake_minimum_required(VERSION 3.11)

project(optimizemethod)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(BUILD_SHARED_LIBS OFF)

# assume built-in pthreads on MacOS
IF(APPLE)
    set(CMAKE_THREAD_LIBS_INIT "-lpthread")
    set(CMAKE_HAVE_THREADS_LIBRARY 1)
    set(CMAKE_USE_WIN32_THREADS_INIT 0)
    set(CMAKE_USE_PTHREADS_INIT 1)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
ENDIF()

# include dir
include_directories(
    include
    3rdparty/eigen
)

# add log module
add_subdirectory(3rdparty/spdlog)

# add optimize method lib
add_library(${PROJECT_NAME} STATIC src/OptimizeMethod.cpp)
target_link_libraries(${PROJECT_NAME} spdlog)

# add executable 
add_executable(assignment_1 src/assignment_1.cpp)
target_link_libraries(assignment_1 ${PROJECT_NAME})

add_executable(assignment_2 src/assignment_2.cpp)
target_link_libraries(assignment_2 ${PROJECT_NAME})
