#ifndef RNG_H
#define RNG_H
#include <cstdint>

// O mesmo fluxo roda no host (validacao, ferramentas) e no device (kernels).
// Fora do hipcc a macro some e o arquivo continua C++ puro.
//
// O hipcc NAO define __CUDACC__ - quem marca a compilacao de device aqui e
// __HIPCC__. O __CUDACC__ fica no teste so para o caso de alguem incluir este
// cabecalho a partir da versao CUDA: o fluxo de sorteios e o mesmo nas duas, e
// e isso que permite comparar os dois binarios saida a saida.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define RNG_HOST_DEVICE __host__ __device__
#else
#define RNG_HOST_DEVICE
#endif

// Mistura splitmix64: avalanche boa o suficiente para que sementes vizinhas
// (id e turno consecutivos) produzam fluxos sem correlacao.
RNG_HOST_DEVICE inline uint64_t splitMix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Fluxo de sorteios de um agente num turno.
//
// Substitui o mt19937, que custava caro pelo motivo errado: 624 words de
// seeding para fazer dois sorteios e ser jogado fora. Aqui cada sorteio e uma
// funcao pura de (semente do turno, indice do sorteio), ~5 ns contra ~1 us.
//
// Duas propriedades que importam mais que a velocidade:
//
// 1. A semente vem de (baseSeed, id do agente, turno), entao o sorteio de um
//    agente continua independente da ordem em que os agentes rodam. E o que
//    sustenta o laco paralelo de Game::step.
//
// 2. Nao ha estado a carregar: 12 bytes, nenhuma alocacao, nenhum ponteiro. Um
//    mt19937 por thread seria 2,5 KB, inviavel em GPU com milhoes de threads.
//    Numa CDNA/RDNA isso e ainda mais duro que em CUDA: o estado de um agente
//    precisa caber em VGPR, e cada VGPR gasto derruba o numero de wavefronts
//    residentes por CU. Doze bytes cabem; 2,5 KB derramariam para scratch.
//
// O indice existe para que um turno possa consumir varios sorteios sem que eles
// se repitam: hoje sao dois (o slot do passeio e a probabilidade da transicao),
// e sobra espaco caso a transmissao por contato passe a sortear tambem.
struct AgentRng
{
    uint64_t seed;
    uint32_t index;

    RNG_HOST_DEVICE uint32_t next()
    {
        uint64_t hash = splitMix64(seed ^ (uint64_t) index);
        index++;
        return (uint32_t) (hash ^ (hash >> 32));
    }

    // Inteiro uniforme em [0, bound). Multiply-shift no lugar de modulo: sem
    // divisao, sem laco de rejeicao, sem branch - e o vies e da ordem de
    // bound / 2^32, invisivel para os valores usados aqui (6 slots, alguns
    // milhares de nos). Tambem e o que a GPU quer.
    RNG_HOST_DEVICE uint32_t below(uint32_t bound)
    {
        return (uint32_t) (((uint64_t) next() * (uint64_t) bound) >> 32);
    }
};

#endif //RNG_H
