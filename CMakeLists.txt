cmake_minimum_required(VERSION 3.0)
project(MinervaSegs)

include_directories(
  basic
  /opt/mshadow/
  /home/chihyang_teng/2.research/mshan_seg/LibN3L/
)

add_definitions(-DUSE_CUDA=0)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O0" )
ELSE()
    SET( CMAKE_CXX_FLAGS  "-w -msse3 -funroll-loops -O3" )
ENDIF()
####for   openblas
add_definitions(-DMSHADOW_USE_CUDA=0)
add_definitions(-DMSHADOW_USE_CBLAS=1)
add_definitions(-DMSHADOW_USE_MKL=0)

SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lopenblas")
####endfor openblas

####for   cuda
#add_definitions(-DMSHADOW_USE_CUDA=1)
#add_definitions(-DMSHADOW_USE_CBLAS=1)
#add_definitions(-DMSHADOW_USE_MKL=0)

#SET( CMAKE_SHARED_LINKER_FLAGS  "-lm -lcudart -lcublas -lcurand" )
#include_directories(
#  $(USE_CUDA_PATH)/include
#)
#LINK_DIRECTORIES($(USE_CUDA_PATH)/lib64)
####endfor cuda

#add_subdirectory(basic)

#aux_source_directory(. DIR_SRCS)


add_executable(GatedLabeler GatedLabeler.cpp)
add_executable(GatedCRFMMLabeler GatedCRFMMLabeler.cpp)
add_executable(GatedCRFMLLabeler GatedCRFMLLabeler.cpp)
add_executable(TNNLabeler TNNLabeler.cpp)
add_executable(TNNCRFMMLabeler TNNCRFMMLabeler.cpp)
add_executable(TNNCRFMLLabeler TNNCRFMLLabeler.cpp)
add_executable(RNNLabeler RNNLabeler.cpp)
add_executable(RNNCRFMMLabeler RNNCRFMMLabeler.cpp)
add_executable(RNNCRFMLLabeler RNNCRFMLLabeler.cpp)
add_executable(GRNNLabeler GRNNLabeler.cpp)
add_executable(GRNNCRFMMLabeler GRNNCRFMMLabeler.cpp)
add_executable(GRNNCRFMLLabeler GRNNCRFMLLabeler.cpp)
add_executable(LSTMLabeler LSTMLabeler.cpp)
add_executable(LSTMCRFMLLabeler LSTMCRFMLLabeler.cpp)
add_executable(LSTMCRFMMLabeler LSTMCRFMMLabeler.cpp)
add_executable(SparseLabeler SparseLabeler.cpp)
add_executable(SparseCRFMMLabeler SparseCRFMMLabeler.cpp)
add_executable(SparseCRFMLLabeler SparseCRFMLLabeler.cpp)
add_executable(SparseGatedLabeler SparseGatedLabeler.cpp)
add_executable(SparseGatedCRFMMLabeler SparseGatedCRFMMLabeler.cpp)
add_executable(SparseGatedCRFMLLabeler SparseGatedCRFMLLabeler.cpp)
add_executable(SparseTNNLabeler SparseTNNLabeler.cpp)
add_executable(SparseTNNCRFMMLabeler SparseTNNCRFMMLabeler.cpp)
add_executable(SparseTNNCRFMLLabeler SparseTNNCRFMLLabeler.cpp)
add_executable(SparseRNNLabeler SparseRNNLabeler.cpp)
add_executable(SparseRNNCRFMMLabeler SparseRNNCRFMMLabeler.cpp)
add_executable(SparseRNNCRFMLLabeler SparseRNNCRFMLLabeler.cpp)
add_executable(SparseGRNNLabeler SparseGRNNLabeler.cpp)
add_executable(SparseGRNNCRFMMLabeler SparseGRNNCRFMMLabeler.cpp)
add_executable(SparseGRNNCRFMLLabeler SparseGRNNCRFMLLabeler.cpp)
add_executable(SparseLSTMLabeler SparseLSTMLabeler.cpp)
add_executable(SparseLSTMCRFMLLabeler SparseLSTMCRFMLLabeler.cpp)
add_executable(SparseLSTMCRFMMLabeler SparseLSTMCRFMMLabeler.cpp)







target_link_libraries(GatedLabeler openblas)
target_link_libraries(GatedCRFMMLabeler openblas)
target_link_libraries(GatedCRFMLLabeler openblas)
target_link_libraries(TNNLabeler openblas)
target_link_libraries(TNNCRFMMLabeler openblas)
target_link_libraries(TNNCRFMLLabeler openblas)
target_link_libraries(RNNLabeler openblas)
target_link_libraries(RNNCRFMMLabeler openblas)
target_link_libraries(RNNCRFMLLabeler openblas)
target_link_libraries(GRNNLabeler openblas)
target_link_libraries(GRNNCRFMMLabeler openblas)
target_link_libraries(GRNNCRFMLLabeler openblas)
target_link_libraries(LSTMLabeler openblas)
target_link_libraries(LSTMCRFMLLabeler openblas)
target_link_libraries(LSTMCRFMMLabeler openblas)
target_link_libraries(SparseLabeler openblas)
target_link_libraries(SparseCRFMMLabeler openblas)
target_link_libraries(SparseCRFMLLabeler openblas)
target_link_libraries(SparseGatedLabeler openblas)
target_link_libraries(SparseGatedCRFMMLabeler openblas)
target_link_libraries(SparseGatedCRFMLLabeler openblas)
target_link_libraries(SparseTNNLabeler openblas)
target_link_libraries(SparseTNNCRFMMLabeler openblas)
target_link_libraries(SparseTNNCRFMLLabeler openblas)
target_link_libraries(SparseRNNLabeler openblas)
target_link_libraries(SparseRNNCRFMMLabeler openblas)
target_link_libraries(SparseRNNCRFMLLabeler openblas)
target_link_libraries(SparseGRNNLabeler openblas)
target_link_libraries(SparseGRNNCRFMMLabeler openblas)
target_link_libraries(SparseGRNNCRFMLLabeler openblas)
target_link_libraries(SparseLSTMLabeler openblas)
target_link_libraries(SparseLSTMCRFMLLabeler openblas)
target_link_libraries(SparseLSTMCRFMMLabeler openblas)


