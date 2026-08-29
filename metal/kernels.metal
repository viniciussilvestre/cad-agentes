// Kernels Metal do laco de agentes.
//
// rng.h e incluido aqui e no host: e literalmente o mesmo gerador dos dois
// lados, nao uma copia. Se fosse copia, CPU e GPU divergiriam em silencio no
// dia em que alguem mexesse num lado so - e a validacao da GPU deixaria de ser
// "reproduz o golden byte a byte" para virar "parece estatisticamente certo".
//
// constants.h NAO entra aqui: o MSL nao aceita variavel constexpr em escopo de
// programa nem o tipo double, e MORTALITY e double. Em vez de duplicar as
// constantes no MSL, o host manda os valores ja calculados no StepParams. A
// fonte unica continua sendo constants.h.

#include <metal_stdlib>
using namespace metal;

#include "enums.h"
#include "rng.h"

struct StepParams
{
    uint numberOfAgents;
    uint numberOfNodes;
    uint baseSeed;
    int  turn;
    uint slots;
    int  exposedTime;
    int  deathTime;
    uint infectionThreshold;
    uint mortalityThreshold;
};

// Marca os nos que tem pelo menos um INFECTED.
//
// Grava 1, e nao incrementa: o unico consumidor testa "> 0", entao a contagem
// exata nunca e lida. Isso troca um atomic_fetch_add por um atomic_store
// relaxado - todos os escritores gravam o MESMO valor, entao nao ha
// leitura-modificacao-escrita e nao ha contencao. Com 1 milhao de agentes sobre
// alguns milhares de nos, o fetch_add seria o gargalo do turno.
//
// O resultado da simulacao e identico ao da CPU, que conta: os dois so sao
// consultados por "> 0".
kernel void markInfected(device const int* agentState [[buffer(0)]],
                         device const int* agentPosition [[buffer(1)]],
                         device atomic_int* infectedAtNode [[buffer(2)]],
                         constant StepParams& params [[buffer(3)]],
                         uint i [[thread_position_in_grid]])
{
    if (i >= params.numberOfAgents)
    {
        return;
    }
    if (agentState[i] != (int) States::INFECTED)
    {
        return;
    }
    atomic_store_explicit(&infectedAtNode[agentPosition[i]], 1, memory_order_relaxed);
}

// Um turno de todos os agentes. Transliteracao de Game::stepAgent, linha a
// linha, inclusive na ORDEM DOS SORTEIOS - que e parte do contrato:
//   sorteio 0  sempre, o slot do passeio;
//   sorteio 1  so quando o agente chega em EXPOSED/INFECTED neste turno.
//
// Cada thread cuida de um agente e so escreve nos proprios indices. Nenhuma
// sincronizacao, nenhum atomico: os ids sao disjuntos e neighbors/infectedAtNode
// sao somente leitura durante o kernel.
kernel void stepAgents(device int* agentPosition [[buffer(0)]],
                       device int* agentState [[buffer(1)]],
                       device int* agentCounter [[buffer(2)]],
                       device const int* neighbors [[buffer(3)]],
                       device const int* infectedAtNode [[buffer(4)]],
                       constant StepParams& params [[buffer(5)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= params.numberOfAgents)
    {
        return;
    }

    int state = agentState[i];

    // DEAD e terminal: nao se move, nao transmite, nao transiciona.
    if (state == (int) States::DEAD)
    {
        return;
    }

    AgentRng rng;
    rng.seed = rngSeedFor(params.baseSeed, (int) i, params.turn);
    rng.index = 0;

    // Passeio: grau fixo, um slot sorteado, auto-laco nos slots vazios.
    int position = agentPosition[i];
    int slot = (int) rng.below(params.slots);
    position = neighbors[slot * (int) params.numberOfNodes + position];
    agentPosition[i] = position;

    // Contato. Uma transicao por turno: quem acaba de ser exposto nao avanca a
    // incubacao no mesmo turno, e por isso sai aqui sem gastar o sorteio 1.
    if (state == (int) States::SUSCEPTIBLE || state == (int) States::RECOVERED)
    {
        if (infectedAtNode[position] > 0)
        {
            agentState[i] = (int) States::EXPOSED;
            agentCounter[i] = 0;
        }
        return;
    }

    // EXPOSED ou INFECTED: avanca o cronograma da doenca.
    int counter = agentCounter[i] + 1;
    uint probability = rng.next();

    if (state == (int) States::EXPOSED)
    {
        if (probability < params.infectionThreshold)
        {
            agentState[i] = (int) States::INFECTED;
            agentCounter[i] = 0;
            return;
        }
        if (counter == params.exposedTime)
        {
            // A janela fechou sem a infeccao pegar: volta a ser suscetivel e
            // pode ser exposto de novo num turno seguinte.
            agentState[i] = (int) States::SUSCEPTIBLE;
            agentCounter[i] = 0;
            return;
        }
        agentCounter[i] = counter;
        return;
    }

    // DEAD e terminal, e o return aqui e o que impede a ressurreicao.
    if (probability < params.mortalityThreshold)
    {
        agentState[i] = (int) States::DEAD;
        agentCounter[i] = 0;
        return;
    }

    if (counter == params.deathTime)
    {
        agentState[i] = (int) States::RECOVERED;
        agentCounter[i] = 0;
        return;
    }

    agentCounter[i] = counter;
}
