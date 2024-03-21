cmake_minimum_required(VERSION 3.14)
project (head_pose)

set(ROOT_PATH ${CMAKE_SOURCE_DIR})
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")

include(FetchContent)
FetchContent_Declare(dlib
    GIT_REPOSITORY https://github.com/davisking/dlib.git
    GIT_TAG        v19.24
)
FetchContent_MakeAvailable(dlib)

find_package(OpenCV REQUIRED)

include_directories(${ROOT_PATH}/inc)
file(GLOB_RECURSE SOURCES ${ROOT_PATH}/src/*.cpp)
add_library(headposelib STATIC ${SOURCES})
target_link_libraries(headposelib dlib::dlib)
target_include_directories(headposelib PUBLIC ${OpenCV_INCLUDE_DIRS})

add_executable(demo ${ROOT_PATH}/main.cpp)
target_link_libraries(demo PUBLIC headposelib)
target_link_libraries(demo PUBLIC ${OpenCV_LIBS})
target_link_libraries(demo PUBLIC stdc++fs)