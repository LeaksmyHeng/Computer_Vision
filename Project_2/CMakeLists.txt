cmake_minimum_required(VERSION 3.10.0)
project(project_2 VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(CTest)
enable_testing()

add_executable(main main.cpp features.cpp)
target_link_libraries(main ${OpenCV_LIBS})

# Define a new target to run the executable with arguments
add_custom_command(TARGET main POST_BUILD
    COMMAND $<TARGET_FILE:main> ${CMAKE_SOURCE_DIR}/olympus/test/pic.009.jpg
    COMMENT "Running the target image"
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
