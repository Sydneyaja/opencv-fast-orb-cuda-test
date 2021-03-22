cmake_minimum_required ( VERSION 2.8 )

project( OpenCVGpuTestProject )

set( EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin )
set( LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib )

find_package( OpenCV REQUIRED )

set(srcfiles
        ${CMAKE_SOURCE_DIR}/src/main.cpp
)

add_executable( GpuTest ${CMAKE_SOURCE_DIR}/src/main.cpp )

target_link_libraries( GpuTest ${OpenCV_LIBS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-ignored-attributes -Wno-deprecated -Wno-unused-result -pthread")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
