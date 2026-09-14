#ifndef DEPARTURE_H
#define DEPARTURE_H
#include <cstdint>

// Um agente que trocou de jogo.
//
// TODA viagem vira um destes, inclusive quando o jogo de destino pertence ao
// mesmo rank. Poderia ser escrita direta na lista do destino, como na versao
// sequencial - mas entao viagem local e viagem remota seguiriam caminhos
// diferentes, e a saida passaria a depender de quantos ranks existem. Um
// caminho so e o que faz `mpirun -n 1` e `mpirun -n 8` darem o mesmo resultado.
//
// De quebra some o turno duplo que a versao sequencial tem (ver
// docs/medicoes.md secao 7): as chegadas sao aplicadas na barreira entre
// turnos, entao nao existe "jogo que ainda nao andou" para receber alguem no
// meio do turno.
//
// O estado viaja junto porque o pool e replicado em ALOCACAO, nao em conteudo:
// cada rank so escreve nos slots dos agentes que estao nos seus jogos, entao o
// slot deste id no rank de destino esta velho.
//
// 16 bytes, sem ponteiro e sem padding implicito: vai como MPI_BYTE, entao o
// layout e o contrato.
struct Departure
{
    int32_t id;
    int32_t destinationGame;
    int32_t position;          // no de chegada, sorteado na ORIGEM
    uint8_t state;
    uint8_t counter;
    uint8_t travelLock;
    uint8_t padding;
};

static_assert(sizeof(Departure) == 16, "Departure vai por MPI_BYTE: o tamanho e contrato");

#endif //DEPARTURE_H
