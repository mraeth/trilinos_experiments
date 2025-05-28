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

# Add src/matrix.hpp to the Poisson executable's source files
add_executable(Poisson src/main.cpp src/matrix.cpp)


target_include_directories(HelloWorld PRIVATE
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

target_link_libraries(HelloWorld PRIVATE
  MPI::MPI_CXX
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)



# Configuration for the Poisson executable
target_include_directories(Poisson PRIVATE
  # Add the 'src' directory so CMake can find matrix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

target_link_libraries(Poisson PRIVATE
  MPI::MPI_CXX
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)