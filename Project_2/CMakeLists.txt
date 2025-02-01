cmake_minimum_required(VERSION 3.10.0)
project(project_2 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(project_2 readfiles-1.cpp)
target_link_libraries(project_2 ${OpenCV_LIBS})

# Passing the argument to run the timeBlue.cpp
add_custom_command(TARGET project_2 POST_BUILD
    COMMAND project_2 ${CMAKE_SOURCE_DIR}/olympus/test   # Pass the filename as an argument here
    COMMENT "Running the executable with a directory as an argument"
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
