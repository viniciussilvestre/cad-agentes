// Kernels Metal: um laco unico sobre TODOS os agentes de TODOS os jogos.
//
// rng.h e incluido aqui e no host: e literalmente o mesmo gerador dos dois
// lados, nao uma copia. Se fosse copia, CPU e GPU divergiriam em silencio no
// dia em que alguem mexesse num lado so.
//
// constants.h NAO entra aqui: o MSL nao aceita variavel constexpr em escopo de
// programa nem o tipo double, e MORTALITY e double. Em vez de duplicar as
// constantes no MSL, o host manda os valores ja calculados nos structs de
// parametros. A fonte unica continua sendo constants.h.
//
// Nao existe lista de presentes por jogo: cada agente carrega gameOf[id], e o
// mapa de qualquer agente sai de duas leituras (nodeOffset e nodeCount). Isso
// apaga a reserva atomica de slot, a lotacao e o turno duplo que as versoes de
// CPU precisam administrar - ver game.h.

#include <metal_stdlib>
using namespace metal;

#include "enums.h"
#include "rng.h"

struct FillParams
{
    ulong base;    // deslocamento deste jogo dentro de neighbors, ja x SLOTS
    uint  nodes;
};

struct SeedParams
{
    uint numberOfAgents;
    uint numberOfGames;
};

struct StepParams
{
    uint numberOfAgents;
    uint numberOfGames;
    int  turn;
    uint slots;
    int  exposedTime;
    int  deathTime;
    uint infectionThreshold;
    uint mortalityThreshold;
    uint travelThreshold;
    uint travelLockTurns;
};

// Auto-laco nos slots vazios, na GPU.
//
// O host sobe o mapa ainda com os -1 que claimSlot usa como sentinela de slot
// livre, e a correcao acontece aqui. A passada e uma leitura e uma escrita
// sobre SLOTS * nodes inteiros - centenas de MB por jogo - e depois da subida
// ninguem mais le esse vetor no host, entao nao ha motivo para a CPU gastar uma
// segunda passada nele.
//
// Grade 2D: x e o no, y e o slot. Assim o indice sai sem divisao.
kernel void fillSelfLoops(device int* neighbors [[buffer(0)]],
                          constant FillParams& params [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= params.nodes)
    {
        return;
    }
    ulong at = params.base + (ulong) gid.y * (ulong) params.nodes + (ulong) gid.x;
    if (neighbors[at] == -1)
    {
        neighbors[at] = (int) gid.x;
    }
}

// Semeadura. Turno -1 e o sorteio uniforme; turno -2 e o da semeadura por
// regiao, separado de proposito para que trocar o modo nao desloque as posicoes
// dos suscetiveis.
kernel void seedAgents(device int* positions [[buffer(0)]],
                       device uchar* states [[buffer(1)]],
                       device uchar* counters [[buffer(2)]],
                       device uchar* travelLock [[buffer(3)]],
                       device uchar* gameOf [[buffer(4)]],
                       device const int* firstAgent [[buffer(5)]],
                       device const int* healthyOf [[buffer(6)]],
                       device const int* nodeCount [[buffer(7)]],
                       device const uint* seeds [[buffer(8)]],
                       device const int* seedMode [[buffer(9)]],
                       device const int* seedTarget [[buffer(10)]],
                       device const int* candidates [[buffer(11)]],
                       device const int* candidateOffset [[buffer(12)]],
                       device const int* candidateCount [[buffer(13)]],
                       constant SeedParams& params [[buffer(14)]],
                       uint index [[thread_position_in_grid]])
{
    if (index >= params.numberOfAgents)
    {
        return;
    }
    int id = (int) index;

    // Qual jogo semeia este agente. As faixas sao contiguas, entao uma busca
    // linear sobre poucos jogos e mais barata que subir uma tabela por agente.
    int game = (int) params.numberOfGames - 1;
    for (uint candidate = 0; candidate < params.numberOfGames; candidate++)
    {
        if (id < firstAgent[candidate])
        {
            game = (int) candidate - 1;
            break;
        }
    }

    bool infected = (id - firstAgent[game]) >= healthyOf[game];
    uint gameSeed = seeds[game];
    int position;

    if (infected && seedMode[game] == (int) SeedMode::SingleNode)
    {
        position = seedTarget[game];
    }
    else if (infected && seedMode[game] == (int) SeedMode::SingleRegion)
    {
        AgentRng rng;
        rng.seed = rngSeedFor(gameSeed, id, -2);
        rng.index = 0;
        position = candidates[candidateOffset[game] + (int) rng.below((uint) candidateCount[game])];
    }
    else
    {
        AgentRng rng;
        rng.seed = rngSeedFor(gameSeed, id, -1);
        rng.index = 0;
        position = (int) rng.below((uint) nodeCount[game]);
    }

    positions[id] = position;
    states[id] = (uchar) (infected ? States::INFECTED : States::SUSCEPTIBLE);
    // O contador mede turnos no estado atual e zera em toda transicao, entao
    // vale zero tambem para quem nasce INFECTED.
    counters[id] = 0;
    travelLock[id] = 0;
    gameOf[id] = (uchar) game;
}

// Marca os nos com pelo menos um infectado, um BIT por no.
//
// A versao em bytes gravava 1 sem atomico, e o argumento era correto: todos os
// escritores gravavam o MESMO valor, entao nao havia leitura-modificacao-escrita.
// Com bits esse argumento morre - dois agentes acendendo bits DIFERENTES da
// mesma palavra sao leitura-modificacao-escrita de verdade, e sem o fetch_or um
// dos dois se perde.
//
// A troca compensa porque o consumidor nao e este kernel. Quem le o vetor e o
// laco de agentes, uma leitura aleatoria por agente vivo por turno, e leituras
// superam escritas em varias vezes: densificar o lado da leitura em 8x paga o
// atomico do lado da escrita com folga. A limpeza do turno tambem encolhe 8x.
kernel void markInfected(device const int* positions [[buffer(0)]],
                         device const uchar* states [[buffer(1)]],
                         device const uchar* gameOf [[buffer(2)]],
                         device const int* nodeOffset [[buffer(3)]],
                         device atomic_uint* infectedMask [[buffer(4)]],
                         constant StepParams& params [[buffer(5)]],
                         uint index [[thread_position_in_grid]])
{
    if (index >= params.numberOfAgents)
    {
        return;
    }
    if (states[index] != (uchar) States::INFECTED)
    {
        return;
    }
    uint node = (uint) nodeOffset[gameOf[index]] + (uint) positions[index];
    atomic_fetch_or_explicit(&infectedMask[node >> 5], 1u << (node & 31u), memory_order_relaxed);
}

// Um turno de todos os agentes de todos os jogos.
//
// Uma thread por agente, nenhuma sincronizacao. infectedMask e so leitura
// durante o kernel (foi congelado pelo anterior, e o encoder serial poe a
// barreira), cada thread escreve exclusivamente nos proprios slots, e o RNG e
// funcao pura de (semente do jogo, id, turno). A viagem tambem: mudar de jogo e
// escrever gameOf[id] e positions[id], os dois do proprio agente.
kernel void stepAgents(device int* positions [[buffer(0)]],
                       device uchar* states [[buffer(1)]],
                       device uchar* counters [[buffer(2)]],
                       device uchar* travelLock [[buffer(3)]],
                       device uchar* gameOf [[buffer(4)]],
                       device const int* neighbors [[buffer(5)]],
                       device const uint* infectedMask [[buffer(6)]],
                       device const int* nodeOffset [[buffer(7)]],
                       device const int* nodeCount [[buffer(8)]],
                       device const uint* seeds [[buffer(9)]],
                       constant StepParams& params [[buffer(10)]],
                       uint index [[thread_position_in_grid]])
{
    if (index >= params.numberOfAgents)
    {
        return;
    }
    int id = (int) index;
    uchar state = states[id];

    // DEAD e terminal: nao se move, nao transmite, nao transiciona, nao viaja.
    // O early return tambem economiza os sorteios.
    if (state == (uchar) States::DEAD)
    {
        return;
    }

    int game = (int) gameOf[id];
    int offset = nodeOffset[game];
    int nodes = nodeCount[game];

    AgentRng rng;
    rng.seed = rngSeedFor(seeds[game], id, params.turn);
    rng.index = 0;

    // Grau fixo: todo no tem exatamente SLOTS slots e os vazios apontam para o
    // proprio no. Sorteia-se um slot e pronto - sem laco de rejeicao, sem
    // checagem de limites, sem branch, o que na GPU quer dizer sem divergencia
    // de warp.
    int slot = (int) rng.below(params.slots);
    ulong at = (ulong) offset * (ulong) params.slots
             + (ulong) slot * (ulong) nodes
             + (ulong) positions[id];
    int position = neighbors[at];
    positions[id] = position;

    // Contato. Cada agente faz no maximo uma transicao por turno: quem acaba de
    // ser exposto nao avanca a incubacao no mesmo turno, e por isso sai daqui
    // sem gastar o segundo sorteio.
    if (state == (uchar) States::SUSCEPTIBLE || state == (uchar) States::RECOVERED)
    {
        uint node = (uint) offset + (uint) position;
        uint bit = (infectedMask[node >> 5] >> (node & 31u)) & 1u;
        if (bit != 0u)
        {
            states[id] = (uchar) States::EXPOSED;
            counters[id] = 0;
        }
    }
    else
    {
        // EXPOSED ou INFECTED. O contador mede turnos NO ESTADO ATUAL e zera em
        // toda transicao; nunca passa de EXPOSED_TIME, entao cabe num byte.
        uchar counter = (uchar) (counters[id] + 1);

        // Sorteio como inteiro de 32 bits contra o limiar equivalente a taxa.
        // Sem double no laco: contracao FMA daria resultados diferentes em CPU
        // e GPU para a mesma conta.
        uint probability = rng.next();

        if (state == (uchar) States::EXPOSED)
        {
            if (probability < params.infectionThreshold)
            {
                states[id] = (uchar) States::INFECTED;
                counters[id] = 0;
            }
            else if ((int) counter == params.exposedTime)
            {
                // A janela fechou sem a infeccao pegar: volta a suscetivel.
                states[id] = (uchar) States::SUSCEPTIBLE;
                counters[id] = 0;
            }
            else
            {
                counters[id] = counter;
            }
        }
        else
        {
            // INFECTED. A ordem importa: sem sair aqui depois da morte, quem
            // morre no ultimo turno da janela cairia no ramo de RECOVERED e
            // voltaria a vida.
            if (probability < params.mortalityThreshold)
            {
                states[id] = (uchar) States::DEAD;
                counters[id] = 0;
                return;
            }
            if ((int) counter == params.deathTime)
            {
                states[id] = (uchar) States::RECOVERED;
                counters[id] = 0;
            }
            else
            {
                counters[id] = counter;
            }
        }
    }

    // Viagem entre jogos, depois do turno normal. Os sorteios comecam no indice
    // 2 porque o passeio consome o 0 e a transicao as vezes o 1 - fixar o
    // indice faz a viagem nao depender do caminho que o agente tomou acima.
    if (params.numberOfGames > 1)
    {
        if (travelLock[id] > 0)
        {
            travelLock[id]--;
        }
        else
        {
            rng.index = 2;
            if (rng.next() < params.travelThreshold)
            {
                // Sorteia entre os OUTROS jogos: tira este do baralho e desloca,
                // em vez de sortear e repetir se cair em si mesmo - laco de
                // rejeicao aqui seria divergencia de warp.
                int destination = (int) rng.below(params.numberOfGames - 1u);
                if (destination >= game)
                {
                    destination++;
                }
                // O destino tem o proprio mapa, entao a posicao antiga nao vale
                // la: o agente chega num no sorteado do grafo novo.
                positions[id] = (int) rng.below((uint) nodeCount[destination]);
                gameOf[id] = (uchar) destination;
                travelLock[id] = (uchar) params.travelLockTurns;
            }
        }
    }
}
