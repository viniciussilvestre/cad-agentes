cmake_minimum_required(VERSION 3.25)
project(GameC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cada versao compila da propria pasta, sem fonte compartilhada: se uma quebrar,
# a outra continua de pe.

add_executable(GameC
        base/main.cpp
        base/game.cpp
        base/game.h
        base/player.cpp
        base/player.h
        base/enums.h
        base/constants.h)
target_include_directories(GameC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_compile_options(GameC PRIVATE -Wall -Wextra -O2)

add_executable(genGraph tools/genGraph.cpp)
target_compile_options(genGraph PRIVATE -Wall -Wextra -O2)

# O mesmo tests.cpp roda contra as duas versoes; quem escolhe e o include path.
add_executable(runTests tests/tests.cpp base/game.cpp base/player.cpp)
target_include_directories(runTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_compile_options(runTests PRIVATE -Wall -Wextra -O2)

# No macOS o Apple clang precisa do libomp do Homebrew; em outros sistemas o
# find_package resolve sozinho.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(GameC_omp
            openmp/main.cpp
            openmp/game.cpp
            openmp/game.h
            openmp/player.cpp
            openmp/player.h
            openmp/enums.h
            openmp/constants.h)
    target_include_directories(GameC_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/openmp)
    target_compile_options(GameC_omp PRIVATE -Wall -Wextra -O2)
    target_link_libraries(GameC_omp PRIVATE OpenMP::OpenMP_CXX)

    add_executable(runTests_omp tests/tests.cpp openmp/game.cpp openmp/player.cpp)
    target_include_directories(runTests_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/openmp)
    target_compile_options(runTests_omp PRIVATE -Wall -Wextra -O2)
    target_link_libraries(runTests_omp PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP nao encontrado: GameC_omp e runTests_omp nao serao construidos. Use o Makefile, que trata o libomp do Homebrew.")
endif()
