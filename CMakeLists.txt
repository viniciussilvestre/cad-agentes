cmake_minimum_required(VERSION 3.25)
project(GameC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

add_executable(runTests tests/tests.cpp base/game.cpp base/player.cpp)
target_include_directories(runTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/base)
target_compile_options(runTests PRIVATE -Wall -Wextra -O2)
