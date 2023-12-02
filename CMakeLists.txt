cmake_minimum_required(VERSION 3.11)

project(optimizemethod)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(BUILD_SHARED_LIBS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static")
set(CMAKE_EXE_LINKER_FLAGS "-static")

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
