// genGraph_cuda - o mesmo gerador de mapas, com a construcao do grafo na GPU.
//
// O que muda em relacao a tools/genGraph.cpp:
//
// 1. RNG. O mt19937 e sequencial e carrega 2,5 KB de estado - inviavel com uma
//    thread por no. Aqui cada sorteio e funcao pura de (semente, no, indice),
//    pelo mesmo splitmix64 que o jogo usa. Isso significa que ESTA VERSAO NAO
//    PRODUZ OS MESMOS ARQUIVOS que a versao de CPU com a mesma semente: os
//    grafos tem as mesmas propriedades, mas nao sao os mesmos grafos.
//
// 2. Resolucao dos atalhos. Na CPU os nos sao processados em ordem e cada
//    aresta ja criada muda o grau visto pelos proximos. Em paralelo isso nao
//    existe, entao a disputa e resolvida em rodadas com atomicMin: numa
//    rodada, cada no destino aceita a proposta do menor id que o quer, e quem
//    recebe nao envia. Assim o grau cresce no maximo 1 por no por rodada, e o
//    resultado nao depende da ordem em que as threads rodam - a mesma semente
//    da o mesmo grafo em qualquer execucao.
//
// O que NAO muda: o formato do arquivo, o teto de grau, a distribuicao de
// Kleinberg dos atalhos, e o escritor rapido - a emissao continua sendo a
// parte mais cara, e ela e de CPU.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using namespace std;

static const int MAX_DEGREE = 6;
static const int SHORTCUT_SLOTS = 2;
static const int MAX_ATTEMPTS = 30;
static const int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr, "Erro de CUDA em %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(status));           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct GraphConfig
{
    int width = 10000;
    int height = 10000;
    double shortcuts = 0.01;
    int regionsX = 5;
    int regionsY = 2;
    int spacing = 100;
    int jitter = 40;
    unsigned int seed = 42;
};

// ---------------------------------------------------------------------------
// RNG: o mesmo splitmix64 do jogo, indexado por (semente, no, indice).
// ---------------------------------------------------------------------------
__host__ __device__ inline uint64_t splitMix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

struct NodeRng
{
    uint64_t seed;
    uint32_t index;

    __host__ __device__ uint32_t next()
    {
        uint64_t hash = splitMix64(seed ^ (uint64_t) index);
        index++;
        return (uint32_t) (hash ^ (hash >> 32));
    }

    __host__ __device__ uint32_t below(uint32_t bound)
    {
        return (uint32_t) (((uint64_t) next() * (uint64_t) bound) >> 32);
    }

    // Uniforme em [0, 1) com 24 bits de mantissa - suficiente para comparar
    // contra uma taxa e sem double no laco.
    __host__ __device__ float unit()
    {
        return (float) (next() >> 8) * (1.0f / 16777216.0f);
    }
};

__host__ __device__ inline NodeRng rngFor(unsigned int seed, int node, int stream)
{
    NodeRng rng;
    uint64_t hash = splitMix64((uint64_t) seed);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) node);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) stream);
    rng.seed = hash;
    rng.index = 0;
    return rng;
}

__device__ inline int neighborSlot(int node, int slot)
{
    return node * MAX_DEGREE + slot;
}

__device__ inline bool hasEdgeDevice(const int* adjacency, const int* degree, int from, int to)
{
    for (int slot = 0; slot < degree[from]; slot++)
    {
        if (adjacency[neighborSlot(from, slot)] == to)
        {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Coordenadas e regiao. Funcao pura do indice do no - o caso mais simples de
// paralelizar que existe, e a fase mais cara das tres na CPU.
__global__ void buildNodesKernel(int* coordinateX, int* coordinateY, int* nodeRegion,
                                 int numberOfNodes, int width, int height,
                                 int regionsX, int regionsY, int spacing, int jitter,
                                 unsigned int seed)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        int row = i / width;
        int column = i % width;

        NodeRng rng = rngFor(seed, i, 0);
        int offsetX = (int) rng.below((uint32_t) (2 * jitter + 1)) - jitter;
        int offsetY = (int) rng.below((uint32_t) (2 * jitter + 1)) - jitter;

        coordinateX[i] = column * spacing + offsetX;
        coordinateY[i] = row * spacing + offsetY;

        int regionColumn = (int) ((long long) column * regionsX / width);
        int regionRow = (int) ((long long) row * regionsY / height);
        nodeRegion[i] = regionRow * regionsX + regionColumn;
    }
}

// Arestas de grade. Cada no escreve os PROPRIOS vizinhos - direita, esquerda,
// baixo, cima - sem tocar na linha de ninguem. Na CPU a aresta era inserida nos
// dois extremos de uma vez; aqui cada lado deduz a sua metade, e o resultado e
// o mesmo grafo sem precisar de atomico nenhum.
__global__ void buildGridKernel(int* adjacency, int* degree,
                                int numberOfNodes, int width)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        int column = i % width;
        int count = 0;

        if (column != width - 1)         { adjacency[neighborSlot(i, count++)] = i + 1; }
        if (column != 0)                 { adjacency[neighborSlot(i, count++)] = i - 1; }
        if (i + width < numberOfNodes)   { adjacency[neighborSlot(i, count++)] = i + width; }
        if (i - width >= 0)              { adjacency[neighborSlot(i, count++)] = i - width; }

        degree[i] = count;
    }
}

// Sorteia o destino do atalho de um no, na mesma distribuicao da versao de CPU:
// amostra a distancia da cumulativa harmonica e depois um ponto do losango
// aquela distancia. Devolve -1 se as MAX_ATTEMPTS tentativas cairem fora da
// grade.
__device__ int drawShortcutTarget(const double* cumulative, int cumulativeSize,
                                  NodeRng& rng, int node, int width, int height)
{
    int row = node / width;
    int column = node % width;

    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
    {
        // Busca binaria na cumulativa, igual ao upper_bound da CPU.
        double target = (double) rng.unit() * cumulative[cumulativeSize - 1];
        int low = 0;
        int high = cumulativeSize - 1;
        while (low < high)
        {
            int middle = (low + high) / 2;
            if (cumulative[middle] <= target) { low = middle + 1; }
            else { high = middle; }
        }
        int distance = low + 1;

        int deltaX = (int) rng.below((uint32_t) (distance + 1));
        int deltaY = distance - deltaX;
        if (rng.below(2) == 1) { deltaX = -deltaX; }
        if (rng.below(2) == 1) { deltaY = -deltaY; }

        int targetColumn = column + deltaX;
        int targetRow = row + deltaY;
        if (targetColumn < 0 || targetColumn >= width || targetRow < 0 || targetRow >= height)
        {
            continue;
        }
        int candidate = targetRow * width + targetColumn;
        if (candidate == node)
        {
            continue;
        }
        return candidate;
    }
    return -1;
}

// Uma proposta de atalho por no para esta rodada.
__global__ void proposeKernel(int* proposal, const double* cumulative, int cumulativeSize,
                              int numberOfNodes, int width, int height,
                              double shortcuts, unsigned int seed, int round)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        // Fluxo proprio por (no, rodada): a proposta de um no nao depende do
        // que os outros fizeram, entao a rodada e reproduzivel.
        NodeRng rng = rngFor(seed, i, round + 1);

        if ((double) rng.unit() >= shortcuts)
        {
            proposal[i] = -1;
            continue;
        }
        proposal[i] = drawShortcutTarget(cumulative, cumulativeSize, rng, i, width, height);
    }
}

// Cada destino fica com a proposta do MENOR id que o quer. atomicMin nao
// depende da ordem em que as threads chegam, entao a rodada inteira e
// deterministica.
__global__ void electKernel(const int* proposal, const int* degree, int* winner,
                            const int* adjacency, int numberOfNodes)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        int target = proposal[i];
        if (target < 0)
        {
            continue;
        }
        if (degree[i] >= MAX_DEGREE || degree[target] >= MAX_DEGREE)
        {
            continue;
        }
        if (hasEdgeDevice(adjacency, degree, i, target))
        {
            continue;
        }
        atomicMin(&winner[target], i);
    }
}

// Um no ou envia, ou recebe, ou nada - nunca as duas coisas na mesma rodada.
// E o que garante que o grau sobe no maximo 1 por rodada e nunca estoura o
// teto, sem precisar reservar os dois extremos de forma atomica.
__device__ inline bool sendsThisRound(int node, const int* proposal, const int* winner,
                                      const int* degree, const int* adjacency)
{
    int target = proposal[node];
    if (target < 0) { return false; }
    if (winner[node] != INT_MAX) { return false; }        // esta recebendo
    if (winner[target] != node) { return false; }         // perdeu a disputa
    if (degree[node] >= MAX_DEGREE || degree[target] >= MAX_DEGREE) { return false; }
    if (hasEdgeDevice(adjacency, degree, node, target)) { return false; }
    return true;
}

// A decisao vem ANTES da aplicacao, em kernel separado, e nao junto com ela.
//
// Se o mesmo kernel decidisse e escrevesse, uma thread estaria atualizando
// degree[i] enquanto outra le degree[i] para avaliar o predicado do proprio
// emissor - e as duas pontas da mesma aresta poderiam chegar a respostas
// diferentes, uma gravando o vizinho e a outra nao. Com a decisao congelada
// num vetor, emissor e receptor leem o mesmo valor.
__global__ void decideKernel(unsigned char* acceptedOut, const int* proposal,
                             const int* winner, const int* degree,
                             const int* adjacency, int numberOfNodes)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        acceptedOut[i] = sendsThisRound(i, proposal, winner, degree, adjacency) ? 1 : 0;
    }
}

__global__ void applyKernel(int* adjacency, int* degree, const int* proposal,
                            const int* winner, const unsigned char* acceptedOut,
                            int numberOfNodes, unsigned long long* accepted)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        bool sends = (acceptedOut[i] != 0);

        // Recebe se o vencedor da sua disputa de fato enviou. Os dois lados
        // consultam a MESMA decisao ja congelada, entao concordam sempre.
        int sender = winner[i];
        bool receives = (sender != INT_MAX) && (acceptedOut[sender] != 0);

        int slot = degree[i];
        if (sends)
        {
            adjacency[neighborSlot(i, slot)] = proposal[i];
            slot++;
            atomicAdd(accepted, 1ULL);
        }
        if (receives)
        {
            adjacency[neighborSlot(i, slot)] = sender;
            slot++;
        }
        degree[i] = slot;
    }
}

__global__ void resetWinnerKernel(int* winner, int numberOfNodes)
{
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numberOfNodes; i += stride)
    {
        winner[i] = INT_MAX;
    }
}

// ---------------------------------------------------------------------------
// Escrita do arquivo - identica a da versao de CPU, e pelo mesmo motivo: o
// ostream formatando inteiro era 65% do programa.
// ---------------------------------------------------------------------------
static inline char* writeInt(char* cursor, int value)
{
    if (value < 0)
    {
        *cursor++ = '-';
        value = -value;
    }
    if (value == 0)
    {
        *cursor++ = '0';
        return cursor;
    }
    char digits[12];
    int count = 0;
    while (value > 0)
    {
        digits[count++] = (char) ('0' + value % 10);
        value /= 10;
    }
    while (count > 0)
    {
        *cursor++ = digits[--count];
    }
    return cursor;
}

static bool writeGraph(const string& outputFile,
                       const vector<int>& coordinateX, const vector<int>& coordinateY,
                       const vector<int>& nodeRegion, const vector<int>& adjacency,
                       const vector<int>& degree, int numberOfNodes, int numberOfRegions,
                       int numberOfEdges)
{
    FILE* file = fopen(outputFile.c_str(), "wb");
    if (file == nullptr)
    {
        return false;
    }

    const size_t CAPACITY = 1u << 22;
    vector<char> storage(CAPACITY + 64);
    char* cursor = storage.data();

    #define FLUSH_IF_FULL()                                                    \
        if ((size_t) (cursor - storage.data()) >= CAPACITY) {                  \
            fwrite(storage.data(), 1, (size_t) (cursor - storage.data()), file); \
            cursor = storage.data();                                           \
        }

    cursor = writeInt(cursor, numberOfNodes);  *cursor++ = '\t';
    cursor = writeInt(cursor, numberOfEdges);  *cursor++ = '\t';
    cursor = writeInt(cursor, numberOfRegions); *cursor++ = '\n';
    *cursor++ = '\n';

    for (int i = 0; i < numberOfNodes; i++)
    {
        cursor = writeInt(cursor, coordinateX[i]); *cursor++ = '\t';
        cursor = writeInt(cursor, coordinateY[i]); *cursor++ = '\t';
        cursor = writeInt(cursor, nodeRegion[i]);  *cursor++ = '\n';
        FLUSH_IF_FULL();
    }
    *cursor++ = '\n';

    for (int i = 0; i < numberOfNodes; i++)
    {
        int targets[MAX_DEGREE];
        int count = 0;
        for (int slot = 0; slot < degree[i]; slot++)
        {
            int neighbor = adjacency[(size_t) i * MAX_DEGREE + slot];
            if (neighbor > i)
            {
                targets[count++] = neighbor;
            }
        }
        for (int a = 1; a < count; a++)
        {
            int value = targets[a];
            int b = a - 1;
            while (b >= 0 && targets[b] > value) { targets[b + 1] = targets[b]; b--; }
            targets[b + 1] = value;
        }
        for (int index = 0; index < count; index++)
        {
            cursor = writeInt(cursor, i);              *cursor++ = '\t';
            cursor = writeInt(cursor, targets[index]); *cursor++ = '\n';
            FLUSH_IF_FULL();
        }
    }

    fwrite(storage.data(), 1, (size_t) (cursor - storage.data()), file);
    fclose(file);
    #undef FLUSH_IF_FULL
    return true;
}

// ---------------------------------------------------------------------------

static int blocksFor(int items)
{
    int blocks = (items + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks < 1) { blocks = 1; }
    if (blocks > 65535) { blocks = 65535; }
    return blocks;
}

static bool generateGraphOnGpu(const GraphConfig& config, const string& outputFile, string& report)
{
    ostringstream out;

    int numberOfNodes = config.width * config.height;
    int numberOfRegions = config.regionsX * config.regionsY;
    int blocks = blocksFor(numberOfNodes);

    // Cumulativa das distancias: P(d) proporcional a 1/d, igual a versao de
    // CPU. Cabe em algumas centenas de KB e fica residente na L2.
    int maxDistance = config.width + config.height - 2;
    if (maxDistance < 1) { maxDistance = 1; }
    vector<double> cumulative((size_t) maxDistance);
    double total = 0.0;
    for (int d = 1; d <= maxDistance; d++)
    {
        total += 1.0 / (double) d;
        cumulative[(size_t) d - 1] = total;
    }

    int* deviceCoordinateX = nullptr;
    int* deviceCoordinateY = nullptr;
    int* deviceNodeRegion = nullptr;
    int* deviceAdjacency = nullptr;
    int* deviceDegree = nullptr;
    int* deviceProposal = nullptr;
    int* deviceWinner = nullptr;
    unsigned char* deviceAcceptedOut = nullptr;
    double* deviceCumulative = nullptr;
    unsigned long long* deviceAccepted = nullptr;

    size_t nodeBytes = (size_t) numberOfNodes * sizeof(int);
    CUDA_CHECK(cudaMalloc(&deviceCoordinateX, nodeBytes));
    CUDA_CHECK(cudaMalloc(&deviceCoordinateY, nodeBytes));
    CUDA_CHECK(cudaMalloc(&deviceNodeRegion, nodeBytes));
    CUDA_CHECK(cudaMalloc(&deviceAdjacency, nodeBytes * MAX_DEGREE));
    CUDA_CHECK(cudaMalloc(&deviceDegree, nodeBytes));
    CUDA_CHECK(cudaMalloc(&deviceProposal, nodeBytes));
    CUDA_CHECK(cudaMalloc(&deviceWinner, nodeBytes));
    CUDA_CHECK(cudaMalloc(&deviceAcceptedOut, (size_t) numberOfNodes));
    CUDA_CHECK(cudaMalloc(&deviceCumulative, cumulative.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&deviceAccepted, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(deviceCumulative, cumulative.data(),
                          cumulative.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(deviceAccepted, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(deviceAdjacency, -1, nodeBytes * MAX_DEGREE));

    buildNodesKernel<<<blocks, BLOCK_SIZE>>>(deviceCoordinateX, deviceCoordinateY, deviceNodeRegion,
                                             numberOfNodes, config.width, config.height,
                                             config.regionsX, config.regionsY,
                                             config.spacing, config.jitter, config.seed);
    CUDA_CHECK(cudaGetLastError());

    buildGridKernel<<<blocks, BLOCK_SIZE>>>(deviceAdjacency, deviceDegree,
                                            numberOfNodes, config.width);
    CUDA_CHECK(cudaGetLastError());

    // Uma rodada por slot de atalho. Cada rodada e uma eleicao completa, e o
    // grau so muda entre rodadas - por isso nada precisa ser reservado nos dois
    // extremos ao mesmo tempo.
    for (int round = 0; round < SHORTCUT_SLOTS; round++)
    {
        resetWinnerKernel<<<blocks, BLOCK_SIZE>>>(deviceWinner, numberOfNodes);
        proposeKernel<<<blocks, BLOCK_SIZE>>>(deviceProposal, deviceCumulative,
                                              (int) cumulative.size(), numberOfNodes,
                                              config.width, config.height,
                                              config.shortcuts, config.seed, round);
        electKernel<<<blocks, BLOCK_SIZE>>>(deviceProposal, deviceDegree, deviceWinner,
                                            deviceAdjacency, numberOfNodes);
        decideKernel<<<blocks, BLOCK_SIZE>>>(deviceAcceptedOut, deviceProposal, deviceWinner,
                                             deviceDegree, deviceAdjacency, numberOfNodes);
        applyKernel<<<blocks, BLOCK_SIZE>>>(deviceAdjacency, deviceDegree, deviceProposal,
                                            deviceWinner, deviceAcceptedOut,
                                            numberOfNodes, deviceAccepted);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<int> coordinateX((size_t) numberOfNodes);
    vector<int> coordinateY((size_t) numberOfNodes);
    vector<int> nodeRegion((size_t) numberOfNodes);
    vector<int> adjacency((size_t) numberOfNodes * MAX_DEGREE);
    vector<int> degree((size_t) numberOfNodes);
    unsigned long long acceptedShortcuts = 0;

    CUDA_CHECK(cudaMemcpy(coordinateX.data(), deviceCoordinateX, nodeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(coordinateY.data(), deviceCoordinateY, nodeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(nodeRegion.data(), deviceNodeRegion, nodeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(adjacency.data(), deviceAdjacency, nodeBytes * MAX_DEGREE, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(degree.data(), deviceDegree, nodeBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&acceptedShortcuts, deviceAccepted, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    cudaFree(deviceCoordinateX);
    cudaFree(deviceCoordinateY);
    cudaFree(deviceNodeRegion);
    cudaFree(deviceAdjacency);
    cudaFree(deviceDegree);
    cudaFree(deviceProposal);
    cudaFree(deviceWinner);
    cudaFree(deviceAcceptedOut);
    cudaFree(deviceCumulative);
    cudaFree(deviceAccepted);

    long long degreeTotal = 0;
    int minimumDegree = MAX_DEGREE;
    int maximumDegree = 0;
    for (int value : degree)
    {
        degreeTotal += value;
        if (value < minimumDegree) { minimumDegree = value; }
        if (value > maximumDegree) { maximumDegree = value; }
    }
    int numberOfEdges = (int) (degreeTotal / 2);

    if (!writeGraph(outputFile, coordinateX, coordinateY, nodeRegion, adjacency, degree,
                    numberOfNodes, numberOfRegions, numberOfEdges))
    {
        report = "Erro: nao foi possivel escrever em '" + outputFile + "'.\n";
        return false;
    }

    char line[256];
    out << "arquivo: " << outputFile << "\n";
    snprintf(line, sizeof(line), "grade: %dx%d  spacing=%d  jitter=%d  seed=%u\n",
             config.width, config.height, config.spacing, config.jitter, config.seed);
    out << line;
    out << "nos: " << numberOfNodes << "\n";
    out << "arestas: " << numberOfEdges << "\n";
    snprintf(line, sizeof(line), "grau medio: %.3f\n",
             numberOfNodes > 0 ? (double) degreeTotal / (double) numberOfNodes : 0.0);
    out << line;
    out << "grau minimo: " << minimumDegree << "  grau maximo: " << maximumDegree << "\n";
    out << "atalhos criados: " << acceptedShortcuts << "\n";
    out << "regioes: " << numberOfRegions << " (" << config.regionsX << "x" << config.regionsY << ")\n";

    report = out.str();
    return true;
}

int main(int argc, char** argv)
{
    GraphConfig config;
    int numberOfGraphs = 1;
    string outputPrefix = "map_";

    for (int i = 1; i < argc; i++)
    {
        string flag = argv[i];
        bool hasValue = (i + 1 < argc);
        if (flag == "--width" && hasValue) { config.width = atoi(argv[++i]); }
        else if (flag == "--height" && hasValue) { config.height = atoi(argv[++i]); }
        else if (flag == "--shortcuts" && hasValue) { config.shortcuts = atof(argv[++i]); }
        else if (flag == "--regions-x" && hasValue) { config.regionsX = atoi(argv[++i]); }
        else if (flag == "--regions-y" && hasValue) { config.regionsY = atoi(argv[++i]); }
        else if (flag == "--spacing" && hasValue) { config.spacing = atoi(argv[++i]); }
        else if (flag == "--jitter" && hasValue) { config.jitter = atoi(argv[++i]); }
        else if (flag == "--seed" && hasValue) { config.seed = (unsigned int) strtoul(argv[++i], nullptr, 10); }
        else if (flag == "--graphs" && hasValue) { numberOfGraphs = atoi(argv[++i]); }
        else if (flag == "--out-prefix" && hasValue) { outputPrefix = argv[++i]; }
        else
        {
            fprintf(stderr, "Erro: flag desconhecida ou sem valor: '%s'.\n", flag.c_str());
            return EXIT_FAILURE;
        }
    }

    if (config.width < 1 || config.height < 1)
    {
        fprintf(stderr, "Erro: --width e --height precisam ser >= 1.\n");
        return EXIT_FAILURE;
    }
    if (config.regionsX < 1 || config.regionsY < 1 ||
        config.regionsX > config.width || config.regionsY > config.height)
    {
        fprintf(stderr, "Erro: --regions-x/--regions-y precisam estar entre 1 e a dimensao correspondente.\n");
        return EXIT_FAILURE;
    }
    if (config.shortcuts < 0.0 || config.shortcuts > 1.0)
    {
        fprintf(stderr, "Erro: --shortcuts precisa estar em [0, 1].\n");
        return EXIT_FAILURE;
    }
    if (config.jitter < 0)
    {
        fprintf(stderr, "Erro: --jitter precisa ser >= 0.\n");
        return EXIT_FAILURE;
    }
    if (numberOfGraphs < 1)
    {
        fprintf(stderr, "Erro: --graphs precisa ser >= 1.\n");
        return EXIT_FAILURE;
    }

    // Os grafos saem um de cada vez: um so ja ocupa a GPU inteira, e a memoria
    // do device e o limite real - 53 bytes por no.
    for (int graph = 0; graph < numberOfGraphs; graph++)
    {
        GraphConfig local = config;
        local.seed = config.seed + (unsigned int) graph;
        string report;
        if (!generateGraphOnGpu(local, outputPrefix + to_string(graph) + ".txt", report))
        {
            fputs(report.c_str(), stderr);
            return EXIT_FAILURE;
        }
        fputs(report.c_str(), stderr);
    }

    return EXIT_SUCCESS;
}
