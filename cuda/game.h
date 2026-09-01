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

// Na versao CUDA, Game e so o dono de um MAPA.
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
// O mapa e efemero no host: existe entre initializeGame e a subida para o
// device, e releaseHostMap o devolve em seguida. Segurar os vetores ate o fim
// da simulacao custava centenas de MB por jogo para nada.
class Game
{
    public:
        int numberOfNodes;
        int numberOfRegions;

        // Grafo do jogo, no host. Sobe para o device concatenado com o dos
        // outros (ver Simulation) e morre logo depois.
        vector<int> neighbors;
        vector<int> nodeRegion;
        vector<vector<int>> regionNodes;

        Game();

        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        void initializeGame(const string& fileName);

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
        int claimSlot(int node, int preferredSlot);

        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //GAME_H
