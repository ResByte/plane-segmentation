cmake_minimum_required(VERSION 2.8.3)

project(opencv_test)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "-stdlib=c++11")
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)
include_directories(
	${OpenCV_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
)
add_executable(run src/main.cpp)
target_link_libraries(run
	${OpenCV_LIBRARIES}
	${Boost_LIBRARIES}
)
