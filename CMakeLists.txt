cmake_minimum_required(VERSION 3.10.0)
project(project_1 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# # add_executable(project_1 imgDisplay.cpp)
# This is to run the video project
# add_executable(project_1 vidDisplay.cpp filters.cpp)
# target_link_libraries(project_1 ${OpenCV_LIBS})

# # to run timeBlur.cpp use this
# add_executable(timeblurr timeBlur.cpp filters.cpp)
# target_link_libraries(timeblurr ${OpenCV_LIBS})

# # Passing the argument to run the timeBlue.cpp
# add_custom_command(TARGET timeblurr POST_BUILD
#     COMMAND timeblurr ${CMAKE_SOURCE_DIR}/Module_1/photos/original.JPG   # Pass the filename as an argument here
#     COMMENT "Running the executable with an image file as an argument"
# )

# To run face detection use this
add_executable(project_1 vidDisplay.cpp filters.cpp faceDetect.cpp haarcascade_frontalface_alt2.xml)
target_link_libraries(project_1 ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
