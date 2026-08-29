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

MTL_SRCS  = metal/main.cpp metal/game.cpp
MTL_HDRS  = metal/game.h metal/enums.h metal/constants.h metal/rng.h metal/gpu.h
MTL_FRAME = -framework Metal -framework Foundation

BASE_SRCS = base/main.cpp base/game.cpp base/player.cpp
BASE_HDRS = base/game.h base/player.h base/enums.h base/constants.h
OMP_SRCS  = openmp/main.cpp openmp/game.cpp openmp/player.cpp
OMP_HDRS  = openmp/game.h openmp/player.h openmp/enums.h openmp/constants.h

all: GameC GameC_omp GameC_metal kernels.metallib $(GENERATOR)

# Cada versao compila da propria pasta, sem fonte compartilhada: se uma quebrar,
# a outra continua de pe.
GameC: $(BASE_SRCS) $(BASE_HDRS)
	$(CXX) $(CXXFLAGS) -Ibase -o $@ $(BASE_SRCS)

GameC_omp: $(OMP_SRCS) $(OMP_HDRS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Iopenmp -o $@ $(OMP_SRCS) $(OMPLIBS)

# O shader compila offline: erro de kernel vira erro de build, e nao surpresa em
# tempo de execucao. Alem disso e o que habilita o profiler do Xcode.
kernels.metallib: metal/kernels.metal $(MTL_HDRS)
	xcrun -sdk macosx metal -O2 -I metal -c metal/kernels.metal -o metal/kernels.air
	xcrun -sdk macosx metallib metal/kernels.air -o $@
	rm -f metal/kernels.air

# gpu.mm e Objective-C++ (unico arquivo que enxerga o Metal); o resto e C++ puro.
GameC_metal: $(MTL_SRCS) metal/gpu.mm $(MTL_HDRS)
	$(CXX) $(CXXFLAGS) -Imetal -o $@ $(MTL_SRCS) metal/gpu.mm -fobjc-arc $(MTL_FRAME)

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
golden: GameC GameC_omp GameC_metal kernels.metallib
	./scripts/check_golden.sh ./GameC
	./scripts/check_golden.sh ./GameC_omp
	./scripts/check_golden.sh ./GameC_metal

clean:
	rm -f GameC GameC_omp GameC_metal kernels.metallib runTests runTests_omp \
	      $(GENERATOR) base/*.o openmp/*.o metal/*.air

.PHONY: all run test golden clean
