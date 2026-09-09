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
  HAVE_METAL = 1
else
  OMPFLAGS   = -fopenmp
  OMPLIBS    = -fopenmp
  HAVE_METAL =
endif

# CUDA: o nvcc raramente esta no PATH. CUDA_ARCH precisa bater com a placa -
# sm_61 Pascal (Quadro P6000), sm_80 Ampere (A100), sm_90 Hopper. Toolkits a
# partir do 13 nao geram mais codigo para Pascal nem Volta.
#
# Por isso a arquitetura nao pode ser um numero fixo: com o 13 instalado,
# -arch=sm_61 nao e aviso, e "nvcc fatal: Unsupported gpu architecture", e o
# `make` inteiro morria ali. Quem sabe o que o toolkit gera e o proprio
# toolkit, entao a lista sai de --list-gpu-code em vez de ser adivinhada pela
# versao.
# Para sobrescrever: make genGraph_cuda CUDA_ARCH=sm_80
NVCC      ?= $(shell command -v nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
HAVE_NVCC := $(shell test -x "$(NVCC)" && echo 1)
ifeq ($(HAVE_NVCC),1)
  NVCC_CODES := $(shell "$(NVCC)" --list-gpu-code 2>/dev/null)
endif

# sm_61 continua o padrao onde o toolkit ainda o suporta - e a placa antiga do
# lab. Onde nao suporta, cai na mais antiga que ele gera, que e a que cobre
# mais placa.
ifneq ($(filter sm_61,$(NVCC_CODES)),)
  CUDA_ARCH ?= sm_61
else ifneq ($(NVCC_CODES),)
  CUDA_ARCH ?= $(firstword $(NVCC_CODES))
else
  CUDA_ARCH ?= sm_61
endif

# CUDA e OPCIONAL. Ter o nvcc no disco nao quer dizer que ele compila para a
# arquitetura pedida, entao o alvo so entra no `all` quando os dois batem -
# do contrario sai de fora, como ja acontece com Metal, MPI e ROCm.
#
# Pedir `make GameC_cuda` na mao continua valendo e continua falhando alto: se
# alguem pediu CUDA por nome, o erro do nvcc e exatamente o que se quer ver.
CUDA_TARGET =
ifeq ($(HAVE_NVCC),1)
ifneq ($(filter $(CUDA_ARCH),$(NVCC_CODES)),)
  CUDA_TARGET = GameC_cuda genGraph_cuda
endif
endif

# ROCm/HIP: mesma ideia do nvcc, com um detalhe a mais. O hipcc precisa saber a
# arquitetura EXATA da placa - gfx90a (MI200), gfx942 (MI300), gfx1030 (RDNA2),
# gfx1100 (RDNA3) - porque, ao contrario do PTX da NVIDIA, nao ha um formato
# intermediario que o driver compile na hora. Binario de gfx errado nao roda.
#
# Por isso a arquitetura sai do rocminfo da propria maquina, e nao de um padrao
# fixo: nao existe um gfx que sirva de aposta razoavel para todo mundo.
# Para sobrescrever: make GameC_rocm ROCM_ARCH=gfx90a
ROCM_PATH  ?= /opt/rocm
HIPCC      ?= $(shell command -v hipcc 2>/dev/null || echo $(ROCM_PATH)/bin/hipcc)
HAVE_HIPCC := $(shell test -x "$(HIPCC)" && echo 1)
ROCM_FOUND := $(shell $(ROCM_PATH)/bin/rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
ROCM_ARCH  ?= $(if $(ROCM_FOUND),$(ROCM_FOUND),native)

# ROCm e OPCIONAL pelo mesmo motivo que CUDA. Ter o hipcc instalado nao quer
# dizer que ha placa da AMD nesta maquina: sem uma, --offload-arch=native nao
# tem o que detectar e o alvo derrubaria o `all` inteiro.
#
# A condicao aceita os dois caminhos - achou GPU aqui, ou alguem disse a
# arquitetura na mao. O segundo e o que permite compilar num login node sem
# placa para rodar depois no no de computo: make GameC_rocm ROCM_ARCH=gfx90a.
ROCM_TARGET =
ifeq ($(HAVE_HIPCC),1)
ifneq ($(ROCM_FOUND)$(filter-out native,$(ROCM_ARCH)),)
  ROCM_TARGET = GameC_rocm
endif
endif

# MPI: o mpicxx raramente esta no PATH em instalacao local, entao procura
# tambem o prefixo comum. Para sobrescrever: make genGraph_mpi MPICXX=mpicxx
MPICXX     ?= $(shell command -v mpicxx 2>/dev/null || echo /opt/openmpi/bin/mpicxx)
HAVE_MPICXX := $(shell test -x "$(MPICXX)" && echo 1)
ifeq ($(HAVE_MPICXX),1)
  MPI_TARGET = genGraph_mpi
else
  MPI_TARGET =
endif

MTL_SRCS  = metal/main.cpp metal/simulation.cpp metal/game.cpp
MTL_HDRS  = metal/simulation.h metal/game.h metal/enums.h metal/constants.h metal/rng.h metal/gpu.h
MTL_FRAME = -framework Metal -framework Foundation

# Metal so existe no macOS: fora dele o alvo sai do `all`/`golden` em vez de
# quebrar o build inteiro.
ifdef HAVE_METAL
  METAL_TARGETS = GameC_metal kernels.metallib
else
  METAL_TARGETS =
endif

BASE_SRCS = base/main.cpp base/simulation.cpp base/game.cpp base/player.cpp
BASE_HDRS = base/simulation.h base/game.h base/player.h base/pool.h base/enums.h base/constants.h
OMP_SRCS  = openmp/main.cpp openmp/simulation.cpp openmp/game.cpp openmp/player.cpp
OMP_HDRS  = openmp/simulation.h openmp/game.h openmp/player.h openmp/pool.h openmp/enums.h openmp/constants.h

all: GameC GameC_omp $(METAL_TARGETS) $(GENERATOR) $(MPI_TARGET) $(CUDA_TARGET) $(ROCM_TARGET)

# Cada versao compila da propria pasta, sem fonte compartilhada: se uma quebrar,
# a outra continua de pe.
GameC: $(BASE_SRCS) $(BASE_HDRS)
	$(CXX) $(CXXFLAGS) -Ibase -o $@ $(BASE_SRCS)

GameC_omp: $(OMP_SRCS) $(OMP_HDRS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Iopenmp -o $@ $(OMP_SRCS) $(OMPLIBS)

ifdef HAVE_METAL

# O shader compila offline: erro de kernel vira erro de build, e nao surpresa em
# tempo de execucao. Alem disso e o que habilita o profiler do Xcode.
kernels.metallib: metal/kernels.metal $(MTL_HDRS)
	xcrun -sdk macosx metal -O2 -I metal -c metal/kernels.metal -o metal/kernels.air
	xcrun -sdk macosx metallib metal/kernels.air -o $@
	rm -f metal/kernels.air

# gpu.mm e Objective-C++ (unico arquivo que enxerga o Metal); o resto e C++ puro.
#
# O -fopenmp nao e para o laco de agentes, que e da GPU: e para ler os mapas em
# paralelo, como na versao CUDA. Com dezenas de arquivos grandes, em serie a
# leitura leva minutos de um core so enquanto a GPU espera parada.
GameC_metal: $(MTL_SRCS) metal/gpu.mm $(MTL_HDRS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Imetal -o $@ $(MTL_SRCS) metal/gpu.mm \
	       -fobjc-arc $(MTL_FRAME) $(OMPLIBS)

else

# Pedir o alvo Metal na mao fora do macOS falha com recado claro, e nao com um
# erro de compilador sobre -fobjc-arc / -framework.
kernels.metallib GameC_metal:
	@echo "$@: alvo Metal so compila no macOS (uname -s = $(UNAME_S))." >&2
	@exit 1

endif

# Os N grafos sao independentes, entao o gerador sai em duas versoes a partir
# do mesmo fonte: threads OpenMP, ou um processo MPI por grafo. A escolha e de
# quem roda, e nao do codigo - o laco e o mesmo nos dois casos.
$(GENERATOR): tools/genGraph.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ tools/genGraph.cpp $(OMPLIBS)

genGraph_mpi: tools/genGraph.cpp
	$(MPICXX) $(CXXFLAGS) -DUSE_MPI -o $@ tools/genGraph.cpp

# A construcao do grafo vai para a GPU; a emissao do arquivo continua na CPU,
# que e onde esta a maior parte do tempo. Ver docs/medicoes.md.
#
# CUDA_CCBIN existe porque o nvcc so aceita GCC ate certa versao (14, no 12.9)
# e a distribuicao pode ter uma mais nova como padrao. Se g++-14 estiver
# instalado, usa ele; senao deixa o nvcc escolher.
CUDA_CCBIN ?= $(shell command -v g++-14 2>/dev/null)
ifneq ($(CUDA_CCBIN),)
  NVCCBIN = -ccbin $(CUDA_CCBIN)
else
  NVCCBIN =
endif

# tools/cuda_compat entra na frente do <math.h> do sistema: a glibc 2.41+
# declara cospi/sinpi/rsqrt com noexcept e o cabecalho do CUDA declara sem, o
# que impede compilar qualquer .cu. Ver o comentario la dentro.
CUDA_GAME_SRCS = cuda/main.cu cuda/simulation.cu cuda/game.cu
CUDA_GAME_HDRS = cuda/simulation.h cuda/game.h cuda/enums.h cuda/constants.h cuda/rng.h

# O laco de agentes de TODOS os jogos cabe num kernel so: cada agente carrega o
# indice do jogo, entao nao ha lista de presentes para manter. Ver simulation.h.
#
# O -fopenmp nao e para o laco de agentes, que e da GPU: e para ler os mapas em
# paralelo. Com 24 arquivos de 1,5 GB, em serie a leitura leva minutos de um
# core so enquanto a GPU espera parada.
GameC_cuda: $(CUDA_GAME_SRCS) $(CUDA_GAME_HDRS) tools/cuda_compat/math.h
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) $(NVCCBIN) -Itools/cuda_compat -Icuda \
	        -o $@ $(CUDA_GAME_SRCS) -Xcompiler "-Wall -Wextra -fopenmp" -lgomp

# Porte HIP da versao CUDA, para GPU da AMD. Mesmo algoritmo e mesma saida: o
# CSV de GameC_rocm tem de bater com o de GameC_cuda linha a linha.
#
# O -x hip existe porque nem toda versao do hipcc reconhece a extensao .hip
# sozinha; dizer a linguagem na linha de comando funciona em todas.
#
# O -fopenmp nao e para o laco de agentes, que e da GPU: e para ler os mapas em
# paralelo. Aqui ele vai direto na linha, e nao dentro de -Xcompiler, porque o
# hipcc E o compilador do host - nao ha um segundo compilador atras dele como
# no nvcc.
ROCM_GAME_SRCS = rocm/main.hip rocm/simulation.hip rocm/game.hip
ROCM_GAME_HDRS = rocm/simulation.h rocm/game.h rocm/enums.h rocm/constants.h rocm/rng.h

GameC_rocm: $(ROCM_GAME_SRCS) $(ROCM_GAME_HDRS)
	$(HIPCC) -std=c++17 -O2 --offload-arch=$(ROCM_ARCH) -Irocm -x hip \
	        -o $@ $(ROCM_GAME_SRCS) -Wall -Wextra -fopenmp

genGraph_cuda: tools/genGraph.cu tools/cuda_compat/math.h
	$(NVCC) -std=c++17 -O2 -arch=$(CUDA_ARCH) $(NVCCBIN) -Itools/cuda_compat \
	        -o $@ tools/genGraph.cu -Xcompiler "-Wall -Wextra"

# O mesmo tests.cpp roda contra as duas versoes; quem escolhe e o -I.
runTests: tests/tests.cpp base/simulation.cpp base/game.cpp base/player.cpp $(BASE_HDRS)
	$(CXX) $(CXXFLAGS) -Ibase -o $@ tests/tests.cpp base/simulation.cpp base/game.cpp base/player.cpp

runTests_omp: tests/tests.cpp openmp/simulation.cpp openmp/game.cpp openmp/player.cpp $(OMP_HDRS)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -Iopenmp -o $@ tests/tests.cpp openmp/simulation.cpp openmp/game.cpp openmp/player.cpp $(OMPLIBS)

run: GameC
	./GameC

test: runTests runTests_omp
	./runTests
	./runTests_omp

# Bit-a-bit contra o golden, nas duas versoes e em varias contagens de thread.
# Divergencia com N threads e race.
golden: GameC GameC_omp $(METAL_TARGETS)
	./scripts/check_golden.sh ./GameC
	./scripts/check_golden.sh ./GameC_omp
ifdef HAVE_METAL
	./scripts/check_golden.sh ./GameC_metal
endif

clean:
	rm -f GameC GameC_omp GameC_metal GameC_cuda GameC_rocm kernels.metallib runTests runTests_omp \
	      $(GENERATOR) genGraph_mpi genGraph_cuda base/*.o openmp/*.o metal/*.air cuda/*.o rocm/*.o

.PHONY: all run test golden clean
