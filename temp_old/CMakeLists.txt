cmake_minimum_required(VERSION 3.25)
project(GameC)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(GameC main.cpp PlagueGame.cpp PlagueGame.h Player.cpp Player.h)
target_compile_options(GameC PRIVATE -Wall -Wextra -O2)

add_executable(genGraph tools/genGraph.cpp)
target_compile_options(genGraph PRIVATE -Wall -Wextra -O2)

add_executable(runTests tests/tests.cpp PlagueGame.cpp Player.cpp)
target_include_directories(runTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(runTests PRIVATE -Wall -Wextra -O2)
