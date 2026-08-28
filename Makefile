CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I.
GENERATOR= genGraph

# OpenMP: o Apple clang do macOS nao aceita -fopenmp direto e precisa do libomp
# do Homebrew. Com GCC ou no Linux basta -fopenmp nos dois lugares. Para
# sobrescrever: make OMPFLAGS=-fopenmp OMPLIBS=-fopenmp
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  OMP_PREFIX = $(shell brew --prefix libomp 2>/dev/null || echo /opt/homebrew/opt/libomp)
  OMPFLAGS   = -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMPLIBS    = -L$(OMP_PREFIX)/lib -lomp
else
  OMPFLAGS   = -fopenmp
  OMPLIBS    = -fopenmp
endif

BASE_SRCS = base/main.cpp base/game.cpp base/player.cpp
BASE_HDRS = base/game.h base/player.h base/enums.h base/constants.h
OMP_SRCS  = openmp/main.cpp openmp/game.cpp openmp/player.cpp
OMP_HDRS  = openmp/game.h openmp/player.h openmp/enums.h openmp/constants.h

all: GameC GameC_omp $(GENERATOR)

# Cada versao compila da propria pasta, sem fonte compartilhada: se uma quebrar,
# a outra continua de pe.
GameC: $(BASE_SRCS) $(BASE_HDRS)
	$(CXX) $(CXXFLAGS) -Ibase -o $@ $(BASE_SRCS)

GameC_omp: $(OMP_SRCS) $(OMP_HDRS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Iopenmp -o $@ $(OMP_SRCS) $(OMPLIBS)

$(GENERATOR): tools/genGraph.cpp
	$(CXX) $(CXXFLAGS) -o $@ tools/genGraph.cpp

# O mesmo tests.cpp roda contra as duas versoes; quem escolhe e o -I.
runTests: tests/tests.cpp base/game.cpp base/player.cpp $(BASE_HDRS)
	$(CXX) $(CXXFLAGS) -Ibase -o $@ tests/tests.cpp base/game.cpp base/player.cpp

runTests_omp: tests/tests.cpp openmp/game.cpp openmp/player.cpp $(OMP_HDRS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Iopenmp -o $@ tests/tests.cpp openmp/game.cpp openmp/player.cpp $(OMPLIBS)

run: GameC
	./GameC

test: runTests runTests_omp
	./runTests
	./runTests_omp

# Bit-a-bit contra o golden, nas duas versoes e em varias contagens de thread.
# Divergencia com N threads e race.
golden: GameC GameC_omp
	./scripts/check_golden.sh ./GameC
	./scripts/check_golden.sh ./GameC_omp

clean:
	rm -f GameC GameC_omp runTests runTests_omp $(GENERATOR) base/*.o openmp/*.o

.PHONY: all run test golden clean
