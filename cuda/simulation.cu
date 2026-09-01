#include "simulation.h"
#include "constants.h"
#include "rng.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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

    // Pedaco da subida por DMA.
    //
    // Copiar direto de um vector<int> e copia de memoria PAGINAVEL: o driver
    // ainda assim passa por um buffer proprio dele, so que escondido e sincrono,
    // e a taxa cai para algo perto de 5 GB/s. Fixar os 13 GB dos mapas tambem
    // nao serve - cudaHostAlloc registra a ~1 GB/s, gastaria mais tempo do que
    // economiza. Um par de buffers pequenos reciclados resolve os dois: o DMA
    // sai de memoria fixada e a memoria fixada e barata de criar.
    //
    // 8 MB ja satura o barramento; o que importa e haver DOIS por thread, para
    // o memcpy do pedaco n+1 rodar enquanto o pedaco n ainda esta no PCIe.
    constexpr size_t STAGE_CHUNK = (size_t) 8 << 20;

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

// Auto-laco nos slots vazios, no device.
//
// O host sobe o mapa ainda com os -1 que claimSlot usa como sentinela de slot
// livre, e a correcao acontece aqui. A passada e uma leitura e uma escrita
// sobre SLOTS * nodes inteiros - centenas de MB por jogo - e depois da subida
// ninguem mais le esse vetor no host, entao nao ha motivo para a CPU gastar
// uma segunda passada nele.
//
// gridDim.y = SLOTS, para o indice do no sair direto de blockIdx.x sem divisao.
__global__ void fillSelfLoopsKernel(int* neighbors, long long base, int nodes)
{
    long long slot = (long long) blockIdx.y;
    long long stride = (long long) blockDim.x * gridDim.x;
    for (long long node = (long long) blockIdx.x * blockDim.x + threadIdx.x;
         node < nodes; node += stride)
    {
        long long at = base + slot * (long long) nodes + node;
        if (neighbors[at] == -1)
        {
            neighbors[at] = (int) node;
        }
    }
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

// Marca os nos com pelo menos um infectado, um BIT por no.
//
// A versao em bytes gravava 1 sem atomico, e o argumento era correto: todos os
// escritores gravavam o MESMO valor, entao nao havia leitura-modificacao-escrita.
// Com bits esse argumento morre - dois agentes acendendo bits DIFERENTES da
// mesma palavra sao leitura-modificacao-escrita de verdade, e sem atomicOr um
// dos dois se perde.
//
// A troca compensa porque o consumidor nao e este kernel. Quem le o vetor e o
// laco de agentes, uma leitura aleatoria por agente vivo por turno, e leituras
// superam escritas em varias vezes: densificar o lado da leitura em 8x paga o
// atomico do lado da escrita com folga. O memset do turno tambem encolhe 8x.
//
// O resultado e identico ao da versao em bytes, bit a bit.
__global__ void markInfectedKernel(const int* positions, const unsigned char* states,
                                   const unsigned char* gameOf, const int* nodeOffset,
                                   unsigned int* infectedMask, int numberOfAgents)
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
        long long node = (long long) nodeOffset[gameOf[id]] + positions[id];
        atomicOr(&infectedMask[node >> 5], 1u << (unsigned int) (node & 31));
    }
}

// Um turno de todos os agentes de todos os jogos.
//
// Uma thread por agente, nenhuma sincronizacao. infectedMask e so leitura
// durante o kernel (foi congelado pelo anterior), cada thread escreve
// exclusivamente nos proprios slots, e o RNG e funcao pura de (semente do jogo,
// id, turno). A viagem tambem: mudar de jogo e escrever gameOf[id] e
// positions[id], os dois do proprio agente.
__global__ void stepKernel(int* positions, unsigned char* states,
                           unsigned char* counters, unsigned char* travelLock,
                           unsigned char* gameOf,
                           const int* neighbors, const unsigned int* infectedMask,
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
            long long node = (long long) offset + position;
            unsigned int bit = (infectedMask[node >> 5] >> (unsigned int) (node & 31)) & 1u;
            if (bit != 0u)
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
// Subida em pipeline
// ---------------------------------------------------------------------------

namespace
{
    // Buffers fixados (pinned) e stream por thread de carga.
    struct Stager
    {
        cudaStream_t stream = nullptr;
        unsigned char* buffer[2] = { nullptr, nullptr };
        cudaEvent_t done[2] = { nullptr, nullptr };
    };

    void createStagers(vector<Stager>& stagers)
    {
        for (Stager& stager : stagers)
        {
            CUDA_CHECK(cudaStreamCreate(&stager.stream));
            for (int slot = 0; slot < 2; slot++)
            {
                CUDA_CHECK(cudaHostAlloc(&stager.buffer[slot], STAGE_CHUNK, cudaHostAllocDefault));
                CUDA_CHECK(cudaEventCreateWithFlags(&stager.done[slot], cudaEventDisableTiming));
            }
        }
    }

    void destroyStagers(vector<Stager>& stagers)
    {
        for (Stager& stager : stagers)
        {
            for (int slot = 0; slot < 2; slot++)
            {
                cudaFreeHost(stager.buffer[slot]);
                cudaEventDestroy(stager.done[slot]);
            }
            cudaStreamDestroy(stager.stream);
        }
    }

    // Copia bytes do host para o device passando pelos dois buffers fixados,
    // alternando entre eles. O evento e o que autoriza a reutilizacao: enquanto
    // o DMA do pedaco anterior nao terminou, aquele buffer ainda esta sendo
    // lido pelo motor de copia.
    //
    // Quando esta funcao retorna, todos os bytes do host ja foram lidos para os
    // buffers fixados - o vetor de origem pode ser liberado na hora, mesmo com
    // DMA ainda em voo.
    void stagedUpload(Stager& stager, void* device, const void* host, size_t bytes)
    {
        const unsigned char* source = (const unsigned char*) host;
        unsigned char* target = (unsigned char*) device;
        int slot = 0;

        for (size_t done = 0; done < bytes; done += STAGE_CHUNK)
        {
            size_t piece = bytes - done;
            if (piece > STAGE_CHUNK) { piece = STAGE_CHUNK; }

            // Um evento nunca gravado conta como concluido, entao as duas
            // primeiras voltas passam direto.
            CUDA_CHECK(cudaEventSynchronize(stager.done[slot]));
            memcpy(stager.buffer[slot], source + done, piece);
            CUDA_CHECK(cudaMemcpyAsync(target + done, stager.buffer[slot], piece,
                                       cudaMemcpyHostToDevice, stager.stream));
            CUDA_CHECK(cudaEventRecord(stager.done[slot], stager.stream));
            slot ^= 1;
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

// initialize ja sobe mapas e configuracao e libera o grafo do host logo depois,
// entao mexer em qualquer um dos tres dali em diante nao teria efeito nenhum no
// device. Falhar alto e melhor que rodar com uma configuracao que o usuario
// acha que aplicou.
void Simulation::refuseAfterInitialize(const char* what) const
{
    if (initialized)
    {
        cerr << "Erro: " << what << " precisa ser chamado antes de initialize()." << endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::setSeed(unsigned int seed)
{
    refuseAfterInitialize("setSeed");
    seeds.assign(games.size(), seed);
}

void Simulation::setSeeds(const vector<unsigned int>& newSeeds)
{
    refuseAfterInitialize("setSeeds");
    if (newSeeds.size() < games.size())
    {
        cerr << "Erro: " << newSeeds.size() << " sementes para " << games.size()
             << " jogos." << endl;
        exit(EXIT_FAILURE);
    }
    for (size_t index = 0; index < games.size(); index++)
    {
        seeds[index] = newSeeds[index];
    }
}

void Simulation::setSeedMode(SeedMode mode, int regionOrNode)
{
    refuseAfterInitialize("setSeedMode");
    for (Game& game : games)
    {
        game.setSeedMode(mode, regionOrNode);
    }
}

void Simulation::setMaps(const vector<string>& files)
{
    refuseAfterInitialize("setMaps");
    if (files.size() < games.size())
    {
        cerr << "Erro: " << files.size() << " mapas para " << games.size()
             << " jogos." << endl;
        exit(EXIT_FAILURE);
    }
    mapFiles = files;
}

void Simulation::initialize(const string& fileName)
{
    setMaps(vector<string>(games.size(), fileName));
    initialize();
}

// Le so os cabecalhos, para saber onde cada mapa comeca no vetor concatenado
// antes de parsear qualquer um deles. Um getline por arquivo.
void Simulation::prescanMaps()
{
    int numberOfGames = (int) games.size();
    nodeOffset.assign((size_t) numberOfGames, 0);
    nodeCount.assign((size_t) numberOfGames, 0);
    totalNodes = 0;

    for (int index = 0; index < numberOfGames; index++)
    {
        int nodes = Game::declaredNodesOf(mapFiles[(size_t) index]);
        nodeOffset[(size_t) index] = (int) totalNodes;
        nodeCount[(size_t) index] = nodes;
        totalNodes += nodes;
    }

    // nodeOffset e int porque vai para o kernel como int. Estourar isso daria
    // deslocamento negativo em silencio.
    if (totalNodes > 2147483647LL)
    {
        cerr << "Erro: " << totalNodes << " nos no total, acima do teto de "
             << "2^31-1 imposto por nodeOffset." << endl;
        exit(EXIT_FAILURE);
    }
}

// Parse e subida em pipeline. Cada thread parseia um mapa e o sobe na sua
// propria stream assim que termina, enquanto as outras ainda estao lendo os
// seus - o tempo de PCIe some atras do tempo de parse em vez de vir depois
// dele.
void Simulation::loadAndUploadMaps()
{
    int numberOfGames = (int) games.size();

    int workers = numberOfGames;
#ifdef _OPENMP
    if (omp_get_max_threads() < workers) { workers = omp_get_max_threads(); }
#else
    workers = 1;
#endif
    if (workers < 1) { workers = 1; }

    vector<Stager> stagers((size_t) workers);
    createStagers(stagers);

    candidatesOf.assign((size_t) numberOfGames, vector<int>());

    #pragma omp parallel for schedule(dynamic) num_threads(workers)
    for (int index = 0; index < numberOfGames; index++)
    {
        int worker = 0;
#ifdef _OPENMP
        worker = omp_get_thread_num();
#endif
        Game& game = games[(size_t) index];
        game.initializeGame(mapFiles[(size_t) index]);
        game.validateSeeding();

        // Candidatos da semeadura por regiao, copiados AQUI porque regionNodes
        // morre no fim desta iteracao.
        if (game.seedModeOf() == SeedMode::SingleRegion)
        {
            candidatesOf[(size_t) index] = game.regionNodes[(size_t) game.seedTargetOf()];
        }

        size_t bytes = (size_t) game.numberOfNodes * Constants::SLOTS * sizeof(int);
        stagedUpload(stagers[(size_t) worker],
                     deviceNeighbors + (size_t) nodeOffset[(size_t) index] * Constants::SLOTS,
                     game.neighbors.data(), bytes);

        // Na mesma stream da copia, entao roda depois dela.
        dim3 grid((unsigned int) blocksFor(game.numberOfNodes), (unsigned int) Constants::SLOTS);
        fillSelfLoopsKernel<<<grid, BLOCK_SIZE, 0, stagers[(size_t) worker].stream>>>(
            deviceNeighbors,
            (long long) nodeOffset[(size_t) index] * Constants::SLOTS,
            game.numberOfNodes);
        CUDA_CHECK(cudaGetLastError());

        // stagedUpload ja leu tudo para os buffers fixados: o grafo do host nao
        // serve mais para nada, e sao centenas de MB por jogo.
        game.releaseHostMap();
    }

    // Espera o DMA em voo antes de devolver os buffers fixados de onde ele le.
    CUDA_CHECK(cudaDeviceSynchronize());
    destroyStagers(stagers);
}

void Simulation::uploadMetadata()
{
    int numberOfGames = (int) games.size();

    // As faixas de id sao calculadas em serie porque cada uma depende da
    // anterior.
    firstAgentOf.assign((size_t) numberOfGames, 0);
    vector<int> healthyOf((size_t) numberOfGames, 0);
    int firstAgent = 0;
    for (int index = 0; index < numberOfGames; index++)
    {
        // Divisao igual, com a sobra distribuida um agente por jogo entre os
        // primeiros, para a soma bater exatamente com o que foi pedido.
        int healthy = totalHealthy / numberOfGames + (index < totalHealthy % numberOfGames ? 1 : 0);
        int infected = totalInfected / numberOfGames + (index < totalInfected % numberOfGames ? 1 : 0);
        firstAgentOf[(size_t) index] = firstAgent;
        healthyOf[(size_t) index] = healthy;
        firstAgent += healthy + infected;
    }

    // Candidatos concatenados. Ficam vazios para os jogos que nao usam
    // SingleRegion.
    vector<int> candidates;
    vector<int> candidateOffset((size_t) numberOfGames, 0);
    vector<int> candidateCount((size_t) numberOfGames, 0);
    vector<int> seedModeOf((size_t) numberOfGames, 0);
    vector<int> seedTargetOf((size_t) numberOfGames, 0);

    for (int index = 0; index < numberOfGames; index++)
    {
        seedModeOf[(size_t) index] = (int) games[(size_t) index].seedModeOf();
        seedTargetOf[(size_t) index] = games[(size_t) index].seedTargetOf();
        candidateOffset[(size_t) index] = (int) candidates.size();
        const vector<int>& list = candidatesOf[(size_t) index];
        candidates.insert(candidates.end(), list.begin(), list.end());
        candidateCount[(size_t) index] = (int) list.size();
    }
    if (candidates.empty())
    {
        candidates.push_back(0);
    }
    vector<vector<int>>().swap(candidatesOf);

    auto upload = [](int** target, const vector<int>& source) {
        CUDA_CHECK(cudaMalloc(target, source.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(*target, source.data(), source.size() * sizeof(int), cudaMemcpyHostToDevice));
    };

    upload(&deviceNodeOffset, nodeOffset);
    upload(&deviceNodeCount, nodeCount);
    upload(&deviceFirstAgent, firstAgentOf);
    upload(&deviceHealthyOf, healthyOf);
    upload(&deviceSeedMode, seedModeOf);
    upload(&deviceSeedTarget, seedTargetOf);
    upload(&deviceCandidates, candidates);
    upload(&deviceCandidateOffset, candidateOffset);
    upload(&deviceCandidateCount, candidateCount);

    CUDA_CHECK(cudaMalloc(&deviceSeeds, seeds.size() * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceSeeds, seeds.data(), seeds.size() * sizeof(unsigned int),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&deviceCounts, (size_t) numberOfGames * 5 * sizeof(int)));
}

void Simulation::initialize()
{
    prescanMaps();

    // Alocar antes do parse e o que permite a subida em pipeline: com os
    // deslocamentos ja fixados, cada mapa sabe onde escrever no momento em que
    // fica pronto.
    infectedMaskWords = (totalNodes + 31) / 32;
    CUDA_CHECK(cudaMalloc(&deviceNeighbors, (size_t) totalNodes * Constants::SLOTS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceInfectedMask, (size_t) infectedMaskWords * sizeof(unsigned int)));
    ready = true;

    loadAndUploadMaps();
    uploadMetadata();
    initialized = true;
}

void Simulation::start()
{
    if (!initialized)
    {
        cerr << "Erro: start() antes de initialize()." << endl;
        exit(EXIT_FAILURE);
    }

    size_t agents = (size_t) numberOfAgents;
    CUDA_CHECK(cudaMalloc(&devicePositions, agents * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&deviceStates, agents));
    CUDA_CHECK(cudaMalloc(&deviceCounters, agents));
    CUDA_CHECK(cudaMalloc(&deviceTravelLock, agents));
    CUDA_CHECK(cudaMalloc(&deviceGameOf, agents));

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

    CUDA_CHECK(cudaMemsetAsync(deviceInfectedMask, 0,
                               (size_t) infectedMaskWords * sizeof(unsigned int)));

    markInfectedKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceGameOf, deviceNodeOffset,
        deviceInfectedMask, numberOfAgents);
    CUDA_CHECK(cudaGetLastError());

    // Mesma stream, entao o passo so comeca quando a marcacao terminou:
    // infectedMask ja esta congelado quando o laco de agentes o le.
    stepKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceCounters, deviceTravelLock, deviceGameOf,
        deviceNeighbors, deviceInfectedMask, deviceNodeOffset, deviceNodeCount,
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
    cudaFree(deviceInfectedMask);
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
