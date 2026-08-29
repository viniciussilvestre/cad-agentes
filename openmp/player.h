#ifndef PLAYER_H
#define PLAYER_H
#include "enums.h"
#include "rng.h"
#include <vector>

using namespace std;

// Nao existe mais a classe Player: o agente e um indice.
//
// Antes cada agente era um objeto de 32 bytes com um ponteiro de volta para o
// Game, e todo estado vivia duas vezes - em agent.state e em
// game->agentsStatusVector[id], em agent.position e em
// game->agentsPositionVector[id]. Duas escritas e duas linhas de cache sujas por
// transicao, mais 3,2 GB de objetos com 100 milhoes de agentes.
//
// Agora o estado mora so nos vetores paralelos (SoA):
//
//   positions[id]   4 bytes
//   states[id]      1 byte
//   counters[id]    1 byte   (nunca passa de EXPOSED_TIME = 7)
//
// 6 bytes por agente contra 40, e o laco de agentes le e escreve exclusivamente
// o slot id - o que torna a paralelizacao obviamente segura em vez de
// argumentavel. E e exatamente a forma que o kernel CUDA vai querer: sem
// ponteiro para host, sem objeto, indice puro.
//
// O turno de um agente e uma funcao livre; as fases (mover, olhar em volta,
// avancar a doenca) sao estaticas dentro do .cpp porque ninguem mais precisa
// chama-las de fora.
void agentTurn(int id, const vector<int>& neighbors, int numberOfNodes,
               const vector<unsigned char>& infectedAtNode,
               vector<int>& positions, vector<States>& states,
               vector<unsigned char>& counters, AgentRng& rng);

#endif //PLAYER_H
