#ifndef GAME_H
#define GAME_H
#include "enums.h"
#include "rng.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>

using namespace std;

// Na versao Metal, Game e so o dono de um MAPA.
//
// Nas versoes de CPU ele tambem carregava a lista de quem esta presente
// (members), com capacidade fixa, reserva atomica de slot e compactacao entre
// turnos. Nada disso existe aqui, e o motivo e a forma do kernel: um unico laco
// cobre TODOS os agentes de TODOS os jogos, e cada agente carrega o indice do
// jogo em que esta. Nao ha lista para percorrer, entao nao ha lista para
// manter.
//
// Isso apaga tres coisas de uma vez:
//
//   - a capacidade fixa e a viagem recusada por lotacao;
//   - a reserva atomica de slot no jogo de destino;
//   - o turno duplo, porque nao existe "jogo que ainda nao andou" - o turno e
//     uma passada so sobre todos os agentes.
//
// Viajar passa a ser escrever dois valores nos proprios slots do agente: o
// indice do jogo e a posicao. Sem corrida nenhuma.
//
// NAO EXISTE COPIA DO GRAFO NO HOST.
//
// A versao CUDA monta a vizinhanca num vector proprio e depois a copia para o
// device, porque la o dado tem que atravessar o PCIe de qualquer forma. Aqui a
// memoria e unificada: initializeGame recebe o ponteiro para dentro do proprio
// MTLBuffer e o parser escreve LA. Some o vector intermediario (nos x 6 x 4
// bytes por jogo, vivo simultaneamente em cada thread da carga) e some a copia.
//
// nodeRegion e regionNodes continuam no host porque so servem para extrair os
// candidatos da semeadura por regiao; releaseHostMap os devolve logo em seguida.
class Game
{
    public:
        int numberOfNodes;
        int numberOfRegions;

        // So o que a semeadura por regiao precisa. A vizinhanca nao mora aqui.
        vector<int> nodeRegion;
        vector<vector<int>> regionNodes;

        Game();

        void setSeedMode(SeedMode mode, int regionOrNode = 0);

        // Escreve a vizinhanca DIRETO em neighbors, que aponta para dentro do
        // buffer da GPU e ja vem semeado com auto-lacos (ver prefillNeighbors).
        // expectedNodes vem do cabecalho, lido antes por declaredNodesOf: e o
        // que fixa o layout do bloco deste jogo.
        void initializeGame(const string& fileName, int* neighbors, int expectedNodes);

        // Le SO o cabecalho do arquivo e devolve o numero de nos declarado.
        //
        // Existe para que Simulation dimensione o vetor concatenado de mapas
        // antes de parsear qualquer um deles: sem isso o deslocamento de um
        // jogo so e conhecido depois que todos os anteriores terminaram, e a
        // subida nao tem como comecar cedo. Custa um getline por arquivo.
        [[nodiscard]] static int declaredNodesOf(const string& fileName);

        // Valida a semeadura antes de qualquer kernel: dentro do device nao ha
        // como reportar erro de configuracao com mensagem util.
        void validateSeeding() const;

        // Devolve a memoria do mapa. Chamado assim que o grafo esta no device.
        void releaseHostMap();

        [[nodiscard]] SeedMode seedModeOf() const { return seedMode; }
        [[nodiscard]] int seedTargetOf() const { return seedTarget; }

    private:
        // Slot livre e o que ainda aponta para o proprio no: o buffer chega
        // semeado com auto-lacos, e aresta com origem igual ao destino e
        // descartada na leitura, entao "aponta para si" nao tem outro
        // significado possivel.
        int claimSlot(int* neighbors, int node, int preferredSlot) const;

        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //GAME_H
