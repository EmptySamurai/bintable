cmake_minimum_required(VERSION 3.9)
project(bintable)

set(CMAKE_CXX_STANDARD 11)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(C:\\Users\\emptysamurai\\Anaconda2\\lib\\site-packages\\numpy\\core\\include)


add_executable(bintable main.cpp utils.h bintablestring.h bintablestring.cpp bintableheader.cpp bintableheader.h bintable.h bintabletypes.h bintablepython.cpp)