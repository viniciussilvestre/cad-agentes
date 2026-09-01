#include "simulation.h"
#include "constants.h"
#include "rng.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t status = (call);                                           \
        if (status != cudaSuccess) {                                           \
            fprintf(stderr, "Erro de CUDA em %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(status));           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

namespace
{
    constexpr int BLOCK_SIZE = 256;

    // Grade limitada com passo grid-stride: com centenas de milhoes de agentes,
    // um bloco por agente daria milhoes de blocos e o escalonador passaria mais
    // tempo criando e destruindo bloco do que rodando agente.
    constexpr int MAX_BLOCKS = 65535;

    int blocksFor(long long items)
    {
        long long blocks = (items + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks < 1) { blocks = 1; }
        if (blocks > MAX_BLOCKS) { blocks = MAX_BLOCKS; }
        return (int) blocks;
    }
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__device__ inline uint64_t seedFor(unsigned int gameSeed, int id, int turn)
{
    uint64_t hash = splitMix64((uint64_t) gameSeed);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) id);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) turn);
    return hash;
}

// Semeadura. Turno -1 e o sorteio uniforme; turno -2 e o da semeadura por
// regiao, separado de proposito para que trocar o modo nao desloque as
// posicoes dos suscetiveis.
__global__ void seedAgentsKernel(int* positions, unsigned char* states,
                                 unsigned char* counters, unsigned char* travelLock,
                                 unsigned char* gameOf,
                                 const int* firstAgent, const int* healthyOf,
                                 const int* nodeCount, const unsigned int* seeds,
                                 const int* seedMode, const int* seedTarget,
                                 const int* candidates, const int* candidateOffset,
                                 const int* candidateCount,
                                 int numberOfGames, int numberOfAgents)
{
    long long stride = (long long) blockDim.x * gridDim.x;
    for (long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
         index < numberOfAgents; index += stride)
    {
        int id = (int) index;

        // Qual jogo semeia este agente. As faixas sao contiguas, entao uma
        // busca linear sobre poucos jogos e mais barata que subir uma tabela.
        int game = numberOfGames - 1;
        for (int candidate = 0; candidate < numberOfGames; candidate++)
        {
            if (id < firstAgent[candidate])
            {
                game = candidate - 1;
                break;
            }
        }

        bool infected = (id - firstAgent[game]) >= healthyOf[game];
        unsigned int gameSeed = seeds[game];
        int position;

        if (infected && seedMode[game] == (int) SeedMode::SingleNode)
        {
            position = seedTarget[game];
        }
        else if (infected && seedMode[game] == (int) SeedMode::SingleRegion)
        {
            AgentRng rng;
            rng.seed = seedFor(gameSeed, id, -2);
            rng.index = 0;
            position = candidates[candidateOffset[game] + (int) rng.below((uint32_t) candidateCount[game])];
        }
        else
        {
            AgentRng rng;
            rng.seed = seedFor(gameSeed, id, -1);
            rng.index = 0;
            position = (int) rng.below((uint32_t) nodeCount[game]);
        }

        positions[id] = position;
        states[id] = (unsigned char) (infected ? States::INFECTED : States::SUSCEPTIBLE);
        // O contador mede turnos no estado atual e zera em toda transicao,
        // entao vale zero tambem para quem nasce INFECTED.
        counters[id] = 0;
        travelLock[id] = 0;
        gameOf[id] = (unsigned char) game;
    }
}

// Marca os nos com pelo menos um infectado.
//
// Grava 1, e nao incrementa: o unico consumidor testa "> 0". Todos os
// escritores gravam o MESMO valor, entao nao ha leitura-modificacao-escrita e
// nao ha necessidade de atomico. Com centenas de milhoes de agentes sobre
// alguns milhoes de nos, um fetch_add seria o gargalo do turno.
__global__ void markInfectedKernel(const int* positions, const unsigned char* states,
                                   const unsigned char* gameOf, const int* nodeOffset,
                                   unsigned char* infectedAtNode, int numberOfAgents)
{
    long long stride = (long long) blockDim.x * gridDim.x;
    for (long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
         index < numberOfAgents; index += stride)
    {
        int id = (int) index;
        if (states[id] != (unsigned char) States::INFECTED)
        {
            continue;
        }
        infectedAtNode[nodeOffset[gameOf[id]] + positions[id]] = 1;
    }
}

// Um turno de todos os agentes de todos os jogos.
//
// Uma thread por agente, nenhuma sincronizacao. infectedAtNode e so leitura
// durante o kernel (foi congelado pelo anterior), cada thread escreve
// exclusivamente nos proprios slots, e o RNG e funcao pura de (semente do jogo,
// id, turno). A viagem tambem: mudar de jogo e escrever gameOf[id] e
// positions[id], os dois do proprio agente.
__global__ void stepKernel(int* positions, unsigned char* states,
                           unsigned char* counters, unsigned char* travelLock,
                           unsigned char* gameOf,
                           const int* neighbors, const unsigned char* infectedAtNode,
                           const int* nodeOffset, const int* nodeCount,
                           const unsigned int* seeds,
                           int numberOfGames, int numberOfAgents, int turn)
{
    long long stride = (long long) blockDim.x * gridDim.x;
    for (long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
         index < numberOfAgents; index += stride)
    {
        int id = (int) index;
        States state = (States) states[id];

        // DEAD e terminal: nao se move, nao transmite, nao transiciona, nao
        // viaja. O early return tambem economiza os sorteios.
        if (state == States::DEAD)
        {
            continue;
        }

        int game = gameOf[id];
        int offset = nodeOffset[game];
        int nodes = nodeCount[game];

        AgentRng rng;
        rng.seed = seedFor(seeds[game], id, turn);
        rng.index = 0;

        // Grau fixo: todo no tem exatamente SLOTS slots e os vazios apontam
        // para o proprio no. Sorteia-se um slot e pronto - sem laco de
        // rejeicao, sem checagem de limites, sem branch, o que na GPU quer
        // dizer sem divergencia de warp.
        int slot = (int) rng.below(Constants::SLOTS);
        int position = neighbors[(long long) offset * Constants::SLOTS + (long long) slot * nodes + positions[id]];
        positions[id] = position;

        // Contato. Cada agente faz no maximo uma transicao por turno: quem
        // acaba de ser exposto nao avanca a incubacao no mesmo turno, e por
        // isso sai daqui sem gastar o segundo sorteio.
        if (state == States::SUSCEPTIBLE || state == States::RECOVERED)
        {
            if (infectedAtNode[offset + position] > 0)
            {
                states[id] = (unsigned char) States::EXPOSED;
                counters[id] = 0;
            }
        }
        else if (state == States::EXPOSED || state == States::INFECTED)
        {
            // O contador mede turnos NO ESTADO ATUAL e zera em toda transicao.
            // Nunca passa de EXPOSED_TIME, entao cabe num byte.
            unsigned char counter = (unsigned char) (counters[id] + 1);

            // Sorteio como inteiro de 32 bits contra o limiar equivalente a
            // taxa. Sem double no laco: contracao FMA daria resultados
            // diferentes em CPU e GPU para a mesma conta.
            uint32_t probability = rng.next();

            if (state == States::EXPOSED)
            {
                if (probability < Constants::INFECTION_THRESHOLD)
                {
                    states[id] = (unsigned char) States::INFECTED;
                    counters[id] = 0;
                }
                else if (counter == Constants::EXPOSED_TIME)
                {
                    // A janela fechou sem a infeccao pegar: volta a suscetivel.
                    states[id] = (unsigned char) States::SUSCEPTIBLE;
                    counters[id] = 0;
                }
                else
                {
                    counters[id] = counter;
                }
            }
            else
            {
                // INFECTED. A ordem importa: sem sair aqui depois da morte,
                // quem morre no ultimo turno da janela cairia no ramo de
                // RECOVERED e voltaria a vida.
                if (probability < Constants::MORTALITY_THRESHOLD)
                {
                    states[id] = (unsigned char) States::DEAD;
                    counters[id] = 0;
                    continue;
                }
                if (counter == Constants::DEATH_TIME)
                {
                    states[id] = (unsigned char) States::RECOVERED;
                    counters[id] = 0;
                }
                else
                {
                    counters[id] = counter;
                }
            }
        }

        // Viagem entre jogos, depois do turno normal. Os sorteios comecam no
        // indice 2 porque o passeio consome o 0 e a transicao as vezes o 1 -
        // fixar o indice faz a viagem nao depender do caminho que o agente
        // tomou acima.
        if (numberOfGames > 1)
        {
            if (travelLock[id] > 0)
            {
                travelLock[id]--;
            }
            else
            {
                rng.index = 2;
                if (rng.next() < Constants::TRAVEL_THRESHOLD)
                {
                    // Sorteia entre os OUTROS jogos: tira este do baralho e
                    // desloca, em vez de sortear e repetir se cair em si mesmo.
                    int destination = (int) rng.below((uint32_t) (numberOfGames - 1));
                    if (destination >= game)
                    {
                        destination++;
                    }
                    // O destino tem o proprio mapa, entao a posicao antiga nao
                    // vale la: o agente chega num no sorteado do grafo novo.
                    positions[id] = (int) rng.below((uint32_t) nodeCount[destination]);
                    gameOf[id] = (unsigned char) destination;
                    travelLock[id] = (unsigned char) Constants::TRAVEL_LOCK_TURNS;
                }
            }
        }
    }
}

// Contagem por (jogo, estado) com privatizacao em memoria compartilhada: o
// bloco acumula localmente e so o agregado vai para a global. Sem isso seriam
// centenas de milhoes de atomicAdd disputando poucas dezenas de enderecos.
__global__ void countKernel(const unsigned char* states, const unsigned char* gameOf,
                            int* counts, int numberOfAgents, int bins)
{
    extern __shared__ int localCounts[];
    for (int bin = threadIdx.x; bin < bins; bin += blockDim.x)
    {
        localCounts[bin] = 0;
    }
    __syncthreads();

    long long stride = (long long) blockDim.x * gridDim.x;
    for (long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
         index < numberOfAgents; index += stride)
    {
        int id = (int) index;
        atomicAdd(&localCounts[gameOf[id] * 5 + (int) states[id]], 1);
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < bins; bin += blockDim.x)
    {
        if (localCounts[bin] != 0)
        {
            atomicAdd(&counts[bin], localCounts[bin]);
        }
    }
}

// ---------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------

Simulation::Simulation(int numberOfGames, int healthy, int infected)
{
    totalHealthy = healthy;
    totalInfected = infected;
    numberOfAgents = healthy + infected;

    if (numberOfGames < 1)
    {
        cerr << "Erro: numero de jogos invalido: " << numberOfGames << "." << endl;
        exit(EXIT_FAILURE);
    }
    // gameOf tem um byte por agente, o que economiza centenas de MB na escala
    // de 600 milhoes - e limita o numero de jogos a 256.
    if (numberOfGames > 256)
    {
        cerr << "Erro: no maximo 256 jogos (gameOf tem um byte por agente)." << endl;
        exit(EXIT_FAILURE);
    }

    games.resize((size_t) numberOfGames);
    mapFiles.assign((size_t) numberOfGames, "A1100a.txt");
    seeds.assign((size_t) numberOfGames, Constants::DEFAULT_SEED);
}

Simulation::~Simulation()
{
    releaseDevice();
}

void Simulation::setSeed(unsigned int seed)
{
    seeds.assign(games.size(), seed);
}

void Simulation::setSeeds(const vector<unsigned int>& newSeeds)
{
    for (size_t index = 0; index < games.size(); index++)
    {
        seeds[index] = newSeeds[index];
    }
}

void Simulation::setSeedMode(SeedMode mode, int regionOrNode)
{
    for (Game& game : games)
    {
        game.setSeedMode(mode, regionOrNode);
    }
}

void Simulation::setMaps(const vector<string>& files)
{
    mapFiles = files;
}

void Simulation::initialize(const string& fileName)
{
    setMaps(vector<string>(games.size(), fileName));
    initialize();
}

void Simulation::initialize()
{
    int numberOfGames = (int) games.size();
    firstAgentOf.assign((size_t) numberOfGames, 0);
    agentsOf.assign((size_t) numberOfGames, 0);

    // As faixas de id sao calculadas em serie porque cada uma depende da
    // anterior.
    int firstAgent = 0;
    for (int index = 0; index < numberOfGames; index++)
    {
        // Divisao igual, com a sobra distribuida um agente por jogo entre os
        // primeiros, para a soma bater exatamente com o que foi pedido.
        int healthy = totalHealthy / numberOfGames + (index < totalHealthy % numberOfGames ? 1 : 0);
        int infected = totalInfected / numberOfGames + (index < totalInfected % numberOfGames ? 1 : 0);
        firstAgentOf[(size_t) index] = firstAgent;
        agentsOf[(size_t) index] = healthy + infected;
        firstAgent += healthy + infected;
    }

    // Ler o arquivo, nao: um jogo nao toca no mapa de outro.
    //
    // Isto e CPU, e e o unico trecho pesado que sobrou nela. Com 24 mapas de
    // 1,5 GB, em serie sao minutos de um core sozinho enquanto a GPU - que faz
    // 20 turnos em 0,4 s - espera parada. E o mesmo motivo pelo qual a versao
    // OpenMP paraleliza este laco.
    #pragma omp parallel for schedule(dynamic)
    for (int index = 0; index < numberOfGames; index++)
    {
        games[index].initializeGame(mapFiles[index]);
        games[index].validateSeeding();
    }
}

void Simulation::uploadMaps()
{
    int numberOfGames = (int) games.size();

    // Mapas concatenados: um vetor de vizinhanca so, com deslocamento por jogo.
    // Um vetor de ponteiros seria uma indireccao extra por acesso, e o acesso
    // aqui ja e aleatorio.
    vector<int> nodeOffset((size_t) numberOfGames, 0);
    vector<int> nodeCount((size_t) numberOfGames, 0);
    totalNodes = 0;
    for (int index = 0; index < numberOfGames; index++)
    {
        nodeOffset[(size_t) index] = (int) totalNodes;
        nodeCount[(size_t) index] = games[index].numberOfNodes;
        totalNodes += games[index].numberOfNodes;
    }

    CUDA_CHECK(cudaMalloc(&deviceNeighbors, (size_t) totalNodes * Constants::SLOTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceNodeRegion, (size_t) totalNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceInfectedAtNode, (size_t) totalNodes));

    for (int index = 0; index < numberOfGames; index++)
    {
        const Game& game = games[index];
        size_t slots = (size_t) game.numberOfNodes * Constants::SLOTS;
        CUDA_CHECK(cudaMemcpy(deviceNeighbors + (size_t) nodeOffset[(size_t) index] * Constants::SLOTS,
                              game.neighbors.data(), slots * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(deviceNodeRegion + nodeOffset[(size_t) index],
                              game.nodeRegion.data(),
                              (size_t) game.numberOfNodes * sizeof(int), cudaMemcpyHostToDevice));
    }

    // Candidatos da semeadura por regiao, concatenados do mesmo jeito. Ficam
    // vazios para os jogos que nao usam SingleRegion.
    vector<int> candidates;
    vector<int> candidateOffset((size_t) numberOfGames, 0);
    vector<int> candidateCount((size_t) numberOfGames, 0);
    vector<int> seedModeOf((size_t) numberOfGames, 0);
    vector<int> seedTargetOf((size_t) numberOfGames, 0);

    for (int index = 0; index < numberOfGames; index++)
    {
        seedModeOf[(size_t) index] = (int) games[index].seedModeOf();
        seedTargetOf[(size_t) index] = games[index].seedTargetOf();
        candidateOffset[(size_t) index] = (int) candidates.size();
        if (games[index].seedModeOf() == SeedMode::SingleRegion)
        {
            const vector<int>& list = games[index].regionNodes[games[index].seedTargetOf()];
            candidates.insert(candidates.end(), list.begin(), list.end());
            candidateCount[(size_t) index] = (int) list.size();
        }
    }
    if (candidates.empty())
    {
        candidates.push_back(0);
    }

    auto upload = [](int** target, const vector<int>& source) {
        CUDA_CHECK(cudaMalloc(target, source.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(*target, source.data(), source.size() * sizeof(int), cudaMemcpyHostToDevice));
    };

    upload(&deviceNodeOffset, nodeOffset);
    upload(&deviceNodeCount, nodeCount);
    upload(&deviceFirstAgent, firstAgentOf);
    upload(&deviceSeedMode, seedModeOf);
    upload(&deviceSeedTarget, seedTargetOf);
    upload(&deviceCandidates, candidates);
    upload(&deviceCandidateOffset, candidateOffset);
    upload(&deviceCandidateCount, candidateCount);

    vector<int> healthyOf((size_t) numberOfGames, 0);
    for (int index = 0; index < numberOfGames; index++)
    {
        healthyOf[(size_t) index] = totalHealthy / numberOfGames +
                                    (index < totalHealthy % numberOfGames ? 1 : 0);
    }
    upload(&deviceHealthyOf, healthyOf);

    CUDA_CHECK(cudaMalloc(&deviceSeeds, seeds.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceSeeds, seeds.data(), seeds.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&deviceCounts, (size_t) numberOfGames * 5 * sizeof(int)));
}

void Simulation::start()
{
    uploadMaps();

    size_t agents = (size_t) numberOfAgents;
    CUDA_CHECK(cudaMalloc(&devicePositions, agents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceStates, agents));
    CUDA_CHECK(cudaMalloc(&deviceCounters, agents));
    CUDA_CHECK(cudaMalloc(&deviceTravelLock, agents));
    CUDA_CHECK(cudaMalloc(&deviceGameOf, agents));
    ready = true;

    int blocks = blocksFor(numberOfAgents);
    seedAgentsKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceCounters, deviceTravelLock, deviceGameOf,
        deviceFirstAgent, deviceHealthyOf, deviceNodeCount, deviceSeeds,
        deviceSeedMode, deviceSeedTarget,
        deviceCandidates, deviceCandidateOffset, deviceCandidateCount,
        (int) games.size(), numberOfAgents);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void Simulation::step()
{
    int blocks = blocksFor(numberOfAgents);

    CUDA_CHECK(cudaMemsetAsync(deviceInfectedAtNode, 0, (size_t) totalNodes));

    markInfectedKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceGameOf, deviceNodeOffset,
        deviceInfectedAtNode, numberOfAgents);
    CUDA_CHECK(cudaGetLastError());

    // Mesma stream, entao o passo so comeca quando a marcacao terminou:
    // infectedAtNode ja esta congelado quando o laco de agentes o le.
    stepKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceCounters, deviceTravelLock, deviceGameOf,
        deviceNeighbors, deviceInfectedAtNode, deviceNodeOffset, deviceNodeCount,
        deviceSeeds, (int) games.size(), numberOfAgents, turns);
    CUDA_CHECK(cudaGetLastError());

    turns++;
}

vector<vector<int>> Simulation::countStatesPerGame()
{
    int numberOfGames = (int) games.size();
    int bins = numberOfGames * 5;

    CUDA_CHECK(cudaMemset(deviceCounts, 0, (size_t) bins * sizeof(int)));
    countKernel<<<blocksFor(numberOfAgents), BLOCK_SIZE, (size_t) bins * sizeof(int)>>>(
        deviceStates, deviceGameOf, deviceCounts, numberOfAgents, bins);
    CUDA_CHECK(cudaGetLastError());

    vector<int> flat((size_t) bins, 0);
    CUDA_CHECK(cudaMemcpy(flat.data(), deviceCounts, (size_t) bins * sizeof(int),
                          cudaMemcpyDeviceToHost));

    vector<vector<int>> counts((size_t) numberOfGames, vector<int>(5, 0));
    for (int game = 0; game < numberOfGames; game++)
    {
        for (int state = 0; state < 5; state++)
        {
            counts[(size_t) game][(size_t) state] = flat[(size_t) game * 5 + state];
        }
    }
    return counts;
}

void Simulation::checkConservation(const vector<vector<int>>& counts) const
{
    long long total = 0;
    for (const vector<int>& perGame : counts)
    {
        for (int value : perGame)
        {
            total += value;
        }
    }
    if (total != (long long) numberOfAgents)
    {
        cerr << "Invariante de conservacao violada no turno " << turns << ": S+E+I+R+D = "
             << total << " != " << numberOfAgents << endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::run(int numberOfTurns)
{
    chrono::steady_clock::time_point before = chrono::steady_clock::now();
    for (int turn = 0; turn < numberOfTurns; turn++)
    {
        step();
    }
    // Os lancamentos sao assincronos: sem esperar aqui, o cronometro mediria o
    // tempo de enfileirar os kernels, e nao o de executa-los.
    CUDA_CHECK(cudaDeviceSynchronize());
    chrono::steady_clock::time_point after = chrono::steady_clock::now();

    vector<vector<int>> counts = countStatesPerGame();
    checkConservation(counts);

    cout << "jogo,turno,S,E,I,R,D" << endl;
    for (size_t game = 0; game < counts.size(); game++)
    {
        cout << game << "," << turns
             << "," << counts[game][static_cast<int>(States::SUSCEPTIBLE)]
             << "," << counts[game][static_cast<int>(States::EXPOSED)]
             << "," << counts[game][static_cast<int>(States::INFECTED)]
             << "," << counts[game][static_cast<int>(States::RECOVERED)]
             << "," << counts[game][static_cast<int>(States::DEAD)] << "\n";
    }
    cout.flush();

    double time = chrono::duration<double>(after - before).count();
    double elapsed = (time > 1e-9) ? time : 1e-9;
    cerr << "tempo: " << fixed << setprecision(3) << elapsed << " s   " << endl;
}

void Simulation::releaseDevice()
{
    if (!ready)
    {
        return;
    }
    cudaFree(devicePositions);
    cudaFree(deviceStates);
    cudaFree(deviceCounters);
    cudaFree(deviceTravelLock);
    cudaFree(deviceGameOf);
    cudaFree(deviceNeighbors);
    cudaFree(deviceNodeRegion);
    cudaFree(deviceInfectedAtNode);
    cudaFree(deviceNodeOffset);
    cudaFree(deviceNodeCount);
    cudaFree(deviceSeeds);
    cudaFree(deviceFirstAgent);
    cudaFree(deviceHealthyOf);
    cudaFree(deviceSeedMode);
    cudaFree(deviceSeedTarget);
    cudaFree(deviceCandidates);
    cudaFree(deviceCandidateOffset);
    cudaFree(deviceCandidateCount);
    cudaFree(deviceCounts);
    ready = false;
}
