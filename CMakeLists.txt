cmake_minimum_required(VERSION 3.17.1)

project(HelloWorld LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Trilinos_DIR "/home/mraeth@ipp.mpg.de/phd/codes/trilinos16/Trilinos-trilinos-release-16-1-0/install/lib/cmake/Trilinos")

# MPI
find_package(MPI REQUIRED)

# Trilinos
find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra MueLu)

add_executable(HelloWorld src/hello_world.cpp)

target_include_directories(HelloWorld PRIVATE
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

target_link_libraries(HelloWorld PRIVATE
  MPI::MPI_CXX
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)
