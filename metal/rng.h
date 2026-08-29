#ifndef RNG_H
#define RNG_H

// Este header compila dos DOIS lados: no host em C++ e dentro do kernel em MSL.
// Nao e conveniencia - e o que garante que CPU e GPU sorteiem exatamente os
// mesmos numeros. Duplicar o gerador nos dois lugares seria pedir para eles
// divergirem em silencio meses depois.
#ifdef __METAL_VERSION__
    #define RNG_U64 ulong
    #define RNG_U32 uint
#else
    #include <cstdint>
    #define RNG_U64 uint64_t
    #define RNG_U32 uint32_t
#endif

// Mistura splitmix64: avalanche boa o suficiente para que sementes vizinhas
// (id e turno consecutivos) produzam fluxos sem correlacao.
inline RNG_U64 splitMix64(RNG_U64 value)
{
    value += 0x9E3779B97F4A7C15UL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBUL;
    return value ^ (value >> 31);
}

// Semente do turno de um agente. Depende so de (semente base, id, turno), nunca
// de um fluxo compartilhado, entao a ordem em que os agentes rodam e irrelevante
// - seja o laco sequencial, o parallel for do OpenMP ou 1 milhao de threads de
// GPU disparadas sem ordem nenhuma.
inline RNG_U64 rngSeedFor(RNG_U32 baseSeed, int id, int turn)
{
    RNG_U64 hash = splitMix64((RNG_U64) baseSeed);
    hash = splitMix64(hash ^ (RNG_U64) (RNG_U32) id);
    hash = splitMix64(hash ^ (RNG_U64) (RNG_U32) turn);
    return hash;
}

// Fluxo de sorteios de um agente num turno.
//
// Substitui o mt19937, que custava caro pelo motivo errado: 624 words de
// seeding para fazer dois sorteios e ser jogado fora. Aqui cada sorteio e uma
// funcao pura de (semente do turno, indice do sorteio), ~5 ns contra ~1 us.
//
// Sao 12 bytes e nenhum estado a carregar. Um mt19937 por thread seria 2,5 KB,
// inviavel numa GPU com milhoes de threads em voo.
//
// O indice permite varios sorteios no mesmo turno sem repeticao: hoje sao dois
// (o slot do passeio e a probabilidade da transicao), e sobra espaco caso a
// transmissao por contato passe a sortear tambem.
struct AgentRng
{
    RNG_U64 seed;
    RNG_U32 index;

    RNG_U32 next()
    {
        RNG_U64 hash = splitMix64(seed ^ (RNG_U64) index);
        index++;
        return (RNG_U32) (hash ^ (hash >> 32));
    }

    // Inteiro uniforme em [0, bound). Multiply-shift no lugar de modulo: sem
    // divisao, sem laco de rejeicao, sem branch - e o vies e da ordem de
    // bound / 2^32, invisivel para os valores usados aqui (6 slots, alguns
    // milhares de nos). Tambem e o que a GPU quer.
    RNG_U32 below(RNG_U32 bound)
    {
        return (RNG_U32) (((RNG_U64) next() * (RNG_U64) bound) >> 32);
    }
};

#endif //RNG_H
