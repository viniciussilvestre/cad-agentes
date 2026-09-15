// Porte HIP/ROCm da versao CUDA, para GPU da AMD.
//
// A forma do algoritmo nao muda: um kernel unico cobre os agentes de TODOS os
// jogos, cada agente carrega o indice do jogo em que esta, e o sorteio e funcao
// pura de (semente do jogo, id, turno). E justamente por isso que o porte pode
// ser verificado - com os mesmos argumentos, este binario tem de imprimir o
// MESMO CSV que o CUDA e que o sequencial, linha a linha. Se divergir, o porte
// esta errado, e nao "diferente porque a GPU e outra".
//
// O que muda e o que era especifico da NVIDIA:
//
//   - a grade deixa de ter teto fixo de 65535 blocos (um numero herdado do
//     limite de gridDim das capacidades antigas) e passa a ser dimensionada
//     pelo numero de CUs da placa;
//   - os kernels declaram __launch_bounds__, porque sem isso o compilador da
//     AMD assume workgroup de 1024 threads e corta o orcamento de VGPR por
//     thread para caber nesse pior caso que nunca acontece aqui;
//   - os comentarios sobre warp de 32 valem para wavefront de 64 nas CDNA
//     (MI100/MI200/MI300) e de 32 nas RDNA - o codigo serve as duas porque
//     nenhum kernel depende da largura do wavefront.

#include "simulation.h"
#include "constants.h"
#include "rng.h"
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <hip/hip_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define HIP_CHECK(call)                                                       \
    do {                                                                       \
        hipError_t status = (call);                                           \
        if (status != hipSuccess) {                                           \
            fprintf(stderr, "Erro de HIP em %s:%d: %s\n",                     \
                    __FILE__, __LINE__, hipGetErrorString(status));           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Contraparte do HIP_CHECK para as chamadas de LIBERACAO.
//
// O HIP marca a API inteira como [[nodiscard]], coisa que o CUDA nao faz -
// entao o que compilava limpo com nvcc gera vinte avisos com hipcc. O retorno
// aqui e ignorado de proposito, e nao por descuido: releaseDevice roda no
// destrutor e destroyStagers roda depois que a subida ja terminou. Nos dois
// casos nao ha mais o que salvar, e abortar dentro de um destrutor trocaria uma
// falha de limpeza por um terminate().
//
// Uma funcao, e nao um (void), porque o (void) some no meio da linha e esconde
// que a decisao foi tomada.
static inline void ignoreOnCleanup(hipError_t) {}

namespace
{
    // Serve as duas familias da AMD sem desperdicio: 256 e multiplo de 64
    // (wavefront das CDNA) e de 32 (wavefront das RDNA), entao nenhum wavefront
    // sai pela metade em nenhuma das duas.
    constexpr int BLOCK_SIZE = 256;

    // Propriedades da placa, consultadas uma vez.
    //
    // A grade continua com passo grid-stride - com centenas de milhoes de
    // agentes, um bloco por agente daria milhoes de blocos e o escalonador
    // passaria mais tempo criando e destruindo bloco do que rodando agente -
    // mas o TETO deixa de ser um numero fixo e passa a sair da placa: blocos
    // suficientes para ocupar a placa inteira uma vez. Numa 7900 XT dao 336
    // blocos de 256 threads, ou 86016 threads, contra os 65535 blocos fixos que
    // a versao CUDA lancaria.
    //
    // CUIDADO COM multiProcessorCount: ele NAO conta CUs em toda placa da AMD.
    // Nas RDNA (gfx10/11/12) duas CUs formam um WGP e o runtime conta WGPs -
    // numa 7900 XT devolve 42, e a placa tem 84 CUs. Nas CDNA e GCN nao ha WGP
    // e o numero ja e o de CUs.
    //
    // Para a GRADE isso nao importa, e e por isso que a conta abaixo usa os
    // dois valores crus: maxThreadsPerMultiProcessor vem na MESMA unidade
    // (2048 por WGP, isto e 1024 por CU), entao o produto da a capacidade certa
    // nas duas familias. Conferido contra o rocminfo desta placa: 84 CUs x 32
    // ondas x 32 threads = 86016, o mesmo que 42 x 2048. Dobrar a grade "para
    // corrigir os CUs" lancaria o dobro do que a placa comporta.
    //
    // Quem precisa da correcao e so o numero MOSTRADO, em computeUnits.
    //
    // Nada disso altera a saida. Como o sorteio de cada agente e funcao pura de
    // (semente, id, turno), o resultado independe de quantos blocos rodaram -
    // que e o que permite comparar este binario com o CUDA linha a linha.
    struct DeviceInfo
    {
        int blockCap = 1024;
        int threadCap = 0;
        int computeUnits = 0;
        int wavefront = 0;
        size_t memory = 0;
        size_t sharedPerBlock = 0;
        char name[256] = "desconhecida";
        char arch[64] = "desconhecida";
    };

    const DeviceInfo& gpu()
    {
        // Estatico local: a consulta acontece uma vez e a inicializacao e
        // thread-safe, o que importa porque blocksFor tambem e chamado de
        // dentro da regiao OpenMP que carrega os mapas.
        static const DeviceInfo info = [] {
            DeviceInfo probed;
            int device = 0;
            hipDeviceProp_t properties;
            if (hipGetDevice(&device) == hipSuccess &&
                hipGetDeviceProperties(&properties, device) == hipSuccess)
            {
                // Por unidade contada pelo runtime (WGP nas RDNA, CU nas
                // CDNA) - a unidade se cancela no produto.
                int blocksPerUnit = properties.maxThreadsPerMultiProcessor / BLOCK_SIZE;
                if (blocksPerUnit < 1) { blocksPerUnit = 1; }
                probed.blockCap = properties.multiProcessorCount * blocksPerUnit;
                probed.threadCap = properties.multiProcessorCount *
                                   properties.maxThreadsPerMultiProcessor;

                // Duas CUs por WGP nas RDNA; nas outras familias a contagem do
                // runtime ja e em CU.
                const char* arch = properties.gcnArchName;
                bool rdna = strncmp(arch, "gfx10", 5) == 0 ||
                            strncmp(arch, "gfx11", 5) == 0 ||
                            strncmp(arch, "gfx12", 5) == 0;
                probed.computeUnits = properties.multiProcessorCount * (rdna ? 2 : 1);
                probed.wavefront = properties.warpSize;
                probed.memory = properties.totalGlobalMem;
                probed.sharedPerBlock = properties.sharedMemPerBlock;
                snprintf(probed.name, sizeof(probed.name), "%s", properties.name);
                snprintf(probed.arch, sizeof(probed.arch), "%s", properties.gcnArchName);
            }
            if (probed.blockCap < 1) { probed.blockCap = 1024; }
            return probed;
        }();
        return info;
    }

    // Pedaco da subida por DMA.
    //
    // Copiar direto de um vector<int> e copia de memoria PAGINAVEL: o driver
    // ainda assim passa por um buffer proprio dele, so que escondido e sincrono,
    // e a taxa cai para algo perto de 5 GB/s. Fixar os 13 GB dos mapas tambem
    // nao serve - hipHostMalloc registra a ~1 GB/s, gastaria mais tempo do que
    // economiza. Um par de buffers pequenos reciclados resolve os dois: o DMA
    // sai de memoria fixada e a memoria fixada e barata de criar.
    //
    // Numa MI300A, que e APU e compartilha a memoria com a CPU, esse cuidado
    // todo e desnecessario - mas tambem nao custa nada la, e e o que faz o
    // mesmo binario ser decente numa placa discreta no PCIe.
    //
    // 8 MB ja satura o barramento; o que importa e haver DOIS por thread, para
    // o memcpy do pedaco n+1 rodar enquanto o pedaco n ainda esta no PCIe.
    constexpr size_t STAGE_CHUNK = (size_t) 8 << 20;

    int blocksFor(long long items)
    {
        long long blocks = (items + BLOCK_SIZE - 1) / BLOCK_SIZE;
        long long cap = gpu().blockCap;
        if (blocks < 1) { blocks = 1; }
        if (blocks > cap) { blocks = cap; }
        return (int) blocks;
    }
}

// ---------------------------------------------------------------------------
// Kernels
//
// Todos declaram __launch_bounds__(BLOCK_SIZE). Sem isso o compilador da AMD
// precisa supor que o kernel pode ser lancado com o maximo de 1024 threads por
// workgroup e reserva registradores para esse pior caso, o que derruba o
// orcamento de VGPR por thread e faz derramar para scratch - memoria global
// com outro nome. Como aqui o tamanho do bloco e sempre BLOCK_SIZE, dizer isso
// ao compilador e de graca.
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
__global__ __launch_bounds__(BLOCK_SIZE) void fillSelfLoopsKernel(int* neighbors, long long base, int nodes)
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
__global__ __launch_bounds__(BLOCK_SIZE) void seedAgentsKernel(int* positions, unsigned char* states,
                                 unsigned char* counters, unsigned char* travelLock,
                                 GameIndex* gameOf,
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

        // Qual jogo semeia este agente: o ultimo cujo primeiro id ainda nao
        // passou de id. As faixas sao contiguas e crescentes, entao da busca
        // binaria.
        //
        // Era uma busca LINEAR, e com poucos jogos isso era mais barato que
        // qualquer alternativa - o vetor cabia na cache e o laco saia cedo. O
        // que derrubou o argumento nao foi a cache, foi a conta: o custo e
        // O(jogos) POR AGENTE. A 256 jogos sao ~128 iteracoes vezes centenas de
        // milhoes de agentes, ja caro; a 4096 jogos vira meio bilhao de
        // iteracoes por milhao de agentes, e a semeadura passa a custar mais
        // que a simulacao inteira.
        //
        // A binaria troca isso por ~12 iteracoes, e o numero de voltas quase
        // nao varia entre threads da mesma onda - nao ha divergencia a pagar.
        int low = 0;
        int high = numberOfGames - 1;
        while (low < high)
        {
            int middle = (low + high + 1) >> 1;
            if (id < firstAgent[middle]) { high = middle - 1; }
            else { low = middle; }
        }
        int game = low;

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
        gameOf[id] = (GameIndex) game;
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
//
// Na AMD o atomicOr sem valor de retorno vira um global_atomic_or que a onda
// dispara e nao espera - nao ha leitura de volta, entao o custo e o da escrita,
// e nao o de uma ida e volta ate a L2.
__global__ __launch_bounds__(BLOCK_SIZE) void markInfectedKernel(const int* positions, const unsigned char* states,
                                   const GameIndex* gameOf, const int* nodeOffset,
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
__global__ __launch_bounds__(BLOCK_SIZE) void stepKernel(int* positions, unsigned char* states,
                           unsigned char* counters, unsigned char* travelLock,
                           GameIndex* gameOf,
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
        // dizer sem divergencia de wavefront - as 64 threads de uma onda CDNA
        // (ou 32 de uma RDNA) seguem o mesmo caminho e so o endereco muda.
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
                    gameOf[id] = (GameIndex) destination;
                    travelLock[id] = (unsigned char) Constants::TRAVEL_LOCK_TURNS;
                }
            }
        }
    }
}

// Contagem por (jogo, estado), por um de dois caminhos.
//
// COM POUCOS JOGOS, privatizacao na LDS: o bloco acumula localmente e so o
// agregado vai para a global. Sem isso seriam centenas de milhoes de atomicAdd
// disputando poucas dezenas de enderecos, e o ds_add_u32 da AMD resolve o
// atomico dentro da propria CU, sem sair para a cache.
//
// COM MUITOS JOGOS isso deixa de ser possivel: a LDS pedida e bins * 4 bytes, e
// a partir de ~3300 jogos ela passa dos 64 KiB que uma CU oferece - o kernel
// nem lanca. Ai a contagem vai direto para a global.
//
// A troca funciona porque os dois casos ruins sao opostos. O que mata o atomico
// global e POUCO endereco com muita thread em cima; e justamente quando ha
// poucos jogos, e ai a LDS cobre. Com muitos jogos ha bins de sobra - a 4096
// jogos sao 20480 enderecos - e a disputa se dilui sozinha.
//
// useShared e uniforme para todas as threads, entao o if abaixo nao diverge.
__global__ __launch_bounds__(BLOCK_SIZE) void countKernel(const unsigned char* states, const GameIndex* gameOf,
                            int* counts, int numberOfAgents, int bins, int useShared)
{
    extern __shared__ int localCounts[];

    if (!useShared)
    {
        long long stride = (long long) blockDim.x * gridDim.x;
        for (long long index = (long long) blockIdx.x * blockDim.x + threadIdx.x;
            index < numberOfAgents; index += stride)
        {
            int id = (int) index;
            atomicAdd(&counts[gameOf[id] * 5 + (int) states[id]], 1);
        }
        return;
    }

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
        hipStream_t stream = nullptr;
        unsigned char* buffer[2] = { nullptr, nullptr };
        hipEvent_t done[2] = { nullptr, nullptr };
    };

    void createStagers(vector<Stager>& stagers)
    {
        for (Stager& stager : stagers)
        {
            HIP_CHECK(hipStreamCreate(&stager.stream));
            for (int slot = 0; slot < 2; slot++)
            {
                HIP_CHECK(hipHostMalloc(&stager.buffer[slot], STAGE_CHUNK, hipHostMallocDefault));
                HIP_CHECK(hipEventCreateWithFlags(&stager.done[slot], hipEventDisableTiming));
            }
        }
    }

    void destroyStagers(vector<Stager>& stagers)
    {
        for (Stager& stager : stagers)
        {
            for (int slot = 0; slot < 2; slot++)
            {
                ignoreOnCleanup(hipHostFree(stager.buffer[slot]));
                ignoreOnCleanup(hipEventDestroy(stager.done[slot]));
            }
            ignoreOnCleanup(hipStreamDestroy(stager.stream));
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
            HIP_CHECK(hipEventSynchronize(stager.done[slot]));
            memcpy(stager.buffer[slot], source + done, piece);
            HIP_CHECK(hipMemcpyAsync(target + done, stager.buffer[slot], piece,
                                       hipMemcpyHostToDevice, stager.stream));
            HIP_CHECK(hipEventRecord(stager.done[slot], stager.stream));
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
    // O teto sai da largura de GameIndex, e nao de um numero escolhido aqui -
    // ver o comentario em simulation.h.
    if (numberOfGames > MAX_GAMES)
    {
        cerr << "Erro: no maximo " << MAX_GAMES << " jogos (largura de GameIndex)." << endl;
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
        HIP_CHECK(hipGetLastError());

        // stagedUpload ja leu tudo para os buffers fixados: o grafo do host nao
        // serve mais para nada, e sao centenas de MB por jogo.
        game.releaseHostMap();
    }

    // Espera o DMA em voo antes de devolver os buffers fixados de onde ele le.
    HIP_CHECK(hipDeviceSynchronize());
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
        HIP_CHECK(hipMalloc(target, source.size() * sizeof(int)));
        HIP_CHECK(hipMemcpy(*target, source.data(), source.size() * sizeof(int), hipMemcpyHostToDevice));
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

    HIP_CHECK(hipMalloc(&deviceSeeds, seeds.size() * sizeof(unsigned int)));
    HIP_CHECK(hipMemcpy(deviceSeeds, seeds.data(), seeds.size() * sizeof(unsigned int),
                          hipMemcpyHostToDevice));

    HIP_CHECK(hipMalloc(&deviceCounts, (size_t) numberOfGames * 5 * sizeof(int)));
}

void Simulation::initialize()
{
    prescanMaps();

    // Qual placa pegou e quanto a corrida vai pedir dela, no stderr.
    //
    // Vai para o stderr, e nao para o stdout, porque o stdout e o CSV que tem
    // de sair byte a byte igual ao das outras versoes.
    //
    // A linha existe porque a faixa de VRAM entre as AMD e enorme: uma MI300X
    // tem 192 GB e uma Radeon de mesa tem 16 ou 24. Com a configuracao densa
    // deste programa a diferenca decide se a corrida cabe, e ver o numero
    // antes do hipMalloc e melhor do que descobrir por um "out of memory" sem
    // contexto.
    const DeviceInfo& device = gpu();
    infectedMaskWords = (totalNodes + 31) / 32;

    long long mapBytes = totalNodes * Constants::SLOTS * (long long) sizeof(int);
    long long maskBytes = infectedMaskWords * (long long) sizeof(unsigned int);
    long long agentBytes = (long long) numberOfAgents *
                           (long long) (sizeof(int) + 3 * sizeof(unsigned char) + sizeof(GameIndex));

    cerr << "GPU: " << device.name << " (" << device.arch << "), "
         << device.computeUnits << " CUs, wavefront " << device.wavefront
         << ", " << (device.memory >> 20) << " MiB de VRAM" << endl;
    cerr << "grade: " << device.blockCap << " blocos x " << BLOCK_SIZE
         << " threads = " << device.threadCap << " (ocupacao cheia da placa)" << endl;
    cerr << "VRAM pedida: " << ((mapBytes + maskBytes + agentBytes) >> 20)
         << " MiB (mapas " << (mapBytes >> 20)
         << " + agentes " << (agentBytes >> 20)
         << " + mascara " << (maskBytes >> 20) << ")" << endl;

    // Alocar antes do parse e o que permite a subida em pipeline: com os
    // deslocamentos ja fixados, cada mapa sabe onde escrever no momento em que
    // fica pronto.
    HIP_CHECK(hipMalloc(&deviceNeighbors, (size_t) totalNodes * Constants::SLOTS * sizeof(int)));
    HIP_CHECK(hipMalloc(&deviceInfectedMask, (size_t) infectedMaskWords * sizeof(unsigned int)));
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
    HIP_CHECK(hipMalloc(&devicePositions, agents * sizeof(int)));
    HIP_CHECK(hipMalloc(&deviceStates, agents));
    HIP_CHECK(hipMalloc(&deviceCounters, agents));
    HIP_CHECK(hipMalloc(&deviceTravelLock, agents));
    HIP_CHECK(hipMalloc(&deviceGameOf, agents * sizeof(GameIndex)));

    int blocks = blocksFor(numberOfAgents);
    seedAgentsKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceCounters, deviceTravelLock, deviceGameOf,
        deviceFirstAgent, deviceHealthyOf, deviceNodeCount, deviceSeeds,
        deviceSeedMode, deviceSeedTarget,
        deviceCandidates, deviceCandidateOffset, deviceCandidateCount,
        (int) games.size(), numberOfAgents);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
}

void Simulation::step()
{
    int blocks = blocksFor(numberOfAgents);

    HIP_CHECK(hipMemsetAsync(deviceInfectedMask, 0,
                               (size_t) infectedMaskWords * sizeof(unsigned int)));

    markInfectedKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceGameOf, deviceNodeOffset,
        deviceInfectedMask, numberOfAgents);
    HIP_CHECK(hipGetLastError());

    // Mesma stream, entao o passo so comeca quando a marcacao terminou:
    // infectedMask ja esta congelado quando o laco de agentes o le.
    stepKernel<<<blocks, BLOCK_SIZE>>>(
        devicePositions, deviceStates, deviceCounters, deviceTravelLock, deviceGameOf,
        deviceNeighbors, deviceInfectedMask, deviceNodeOffset, deviceNodeCount,
        deviceSeeds, (int) games.size(), numberOfAgents, turns);
    HIP_CHECK(hipGetLastError());

    turns++;
}

vector<vector<int>> Simulation::countStatesPerGame()
{
    int numberOfGames = (int) games.size();
    int bins = numberOfGames * 5;

    HIP_CHECK(hipMemset(deviceCounts, 0, (size_t) bins * sizeof(int)));
    // Metade da LDS do workgroup, e nao ela toda: pedir os 64 KiB inteiros
    // deixaria um unico workgroup residente por CU. Acima disso a contagem cai
    // no caminho global, que a essa altura ja tem bins de sobra para diluir a
    // disputa.
    size_t sharedBudget = gpu().sharedPerBlock / 2;
    size_t sharedBytes = (size_t) bins * sizeof(int);
    bool useShared = sharedBytes <= sharedBudget;

    countKernel<<<blocksFor(numberOfAgents), BLOCK_SIZE, useShared ? sharedBytes : 0>>>(
        deviceStates, deviceGameOf, deviceCounts, numberOfAgents, bins, useShared ? 1 : 0);
    HIP_CHECK(hipGetLastError());

    vector<int> flat((size_t) bins, 0);
    HIP_CHECK(hipMemcpy(flat.data(), deviceCounts, (size_t) bins * sizeof(int),
                          hipMemcpyDeviceToHost));

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
    HIP_CHECK(hipDeviceSynchronize());
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
    ignoreOnCleanup(hipFree(devicePositions));
    ignoreOnCleanup(hipFree(deviceStates));
    ignoreOnCleanup(hipFree(deviceCounters));
    ignoreOnCleanup(hipFree(deviceTravelLock));
    ignoreOnCleanup(hipFree(deviceGameOf));
    ignoreOnCleanup(hipFree(deviceNeighbors));
    ignoreOnCleanup(hipFree(deviceInfectedMask));
    ignoreOnCleanup(hipFree(deviceNodeOffset));
    ignoreOnCleanup(hipFree(deviceNodeCount));
    ignoreOnCleanup(hipFree(deviceSeeds));
    ignoreOnCleanup(hipFree(deviceFirstAgent));
    ignoreOnCleanup(hipFree(deviceHealthyOf));
    ignoreOnCleanup(hipFree(deviceSeedMode));
    ignoreOnCleanup(hipFree(deviceSeedTarget));
    ignoreOnCleanup(hipFree(deviceCandidates));
    ignoreOnCleanup(hipFree(deviceCandidateOffset));
    ignoreOnCleanup(hipFree(deviceCandidateCount));
    ignoreOnCleanup(hipFree(deviceCounts));
    ready = false;
}
