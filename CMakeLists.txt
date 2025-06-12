cmake_minimum_required(VERSION 3.17.1)

project(HelloWorld LANGUAGES C CXX)

# Optional CUDA activation via preset
option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
  enable_language(CUDA)
  message(STATUS "CUDA support enabled")
  find_package(CUDAToolkit REQUIRED)
else()
  message(STATUS "CUDA support disabled")
endif()

# C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MPI
find_package(MPI REQUIRED)

# Trilinos
find_package(Trilinos REQUIRED COMPONENTS Teuchos Tpetra MueLu)

# === HelloWorld Target ===
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

if(USE_CUDA)
  target_link_libraries(HelloWorld PRIVATE
    CUDAToolkit::cusolver
    CUDAToolkit::cusparse
  )
endif()

# === Poisson Target ===
add_executable(Poisson
  src/main.cpp
  src/matrix.cpp
)

target_include_directories(Poisson PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${Trilinos_INCLUDE_DIRS}
  ${Trilinos_TPL_INCLUDE_DIRS}
)

target_link_libraries(Poisson PRIVATE
  MPI::MPI_CXX
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)

if(USE_CUDA)
  target_link_libraries(Poisson PRIVATE
    CUDAToolkit::cusolver
    CUDAToolkit::cusparse
  )
  set_target_properties(Poisson PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
