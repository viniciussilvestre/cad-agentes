#ifndef GAME_H
#define GAME_H
#include "player.h"
#include "pool.h"
#include "departure.h"
#include "rng.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <cstdint>

using namespace std;
using Point = pair<int, int>;
using Edge = pair<Point, Point>;

class Game
{
    public:
        // Quantos agentes este jogo semeia. Depois do setup, o numero de
        // agentes presentes e members.size(), que muda com a viagem.
        int initialNumberHealthy;
        int initialNumberInfected;
        int firstAgent;
        int turns;
        int max_x_coordinate;
        int max_y_coordinate;
        int min_x_coordinate;
        int min_y_coordinate;
        int numberOfNodes;
        int numberOfRegions;
        bool isRunning;
        // Bandeira por no, e nao contagem: checkSurroundings so pergunta se ha
        // ALGUM infectado ali. Ver buildInfectedAtNode.
        vector<unsigned char> infectedAtNode;
        // O estado dos agentes NAO mora aqui: mora no pool global, indexado por
        // id (ver pool.h). O jogo e um ambiente - mapa, retrato de infectados e
        // a lista de quem esta aqui agora.
        AgentPool* pool = nullptr;

        // Ids globais presentes neste jogo neste momento. E o unico lugar que a
        // viagem entre jogos mexe.
        //
        // Tamanho FIXO, alocado uma vez em startAllDumbThreads:
        //
        //   capacidade = agentes + agentes / TRAVEL_CAPACITY_DIVISOR
        //
        // Nunca cresce. Um push_back de um agente chegando reallocaria o vetor
        // debaixo do laco que esta percorrendo esta lista, e esse laco passaria
        // a ler memoria liberada - o que trava ou corrompe, e nao e a corrida
        // inofensiva de "o agente entra no turno que vem". Com capacidade fixa
        // nada muda de lugar, e o pior caso e uma viagem recusada.
        //
        // memberCount marca o fim util. Os slots de quem viajou ficam com -1
        // ate a compactacao no fim do turno.
        vector<int> members;
        int memberCount = 0;

        // Todos os ambientes, para a viagem saber onde entregar o agente, e o
        // indice deste jogo dentro deles. Neste rank so os jogos proprios tem
        // mapa carregado; dos outros so se conhece o tamanho (ver nodeCountOf).
        vector<Game>* world = nullptr;
        int gameIndex = 0;

        // Preenchidos pelo Simulation antes do primeiro turno.
        //
        // gameOwner diz de qual rank e cada jogo, e nodeCountOf da o numero de
        // nos de TODOS eles - inclusive os que este rank nunca leu. A origem
        // precisa do tamanho do grafo de destino porque e ela quem sorteia o no
        // de chegada; fazer esse sorteio no destino mudaria o fluxo de numeros
        // do agente conforme onde ele estivesse.
        const int* gameOwner = nullptr;
        const int* nodeCountOf = nullptr;
        int myRank = 0;

        // Onde as viagens deste turno sao anotadas. Nada e aplicado durante o
        // turno: quem aplica e o Simulation, na barreira.
        vector<Departure>* outbox = nullptr;
        // O que sobra do grafo depois da leitura e so isto: a vizinhanca de
        // grau fixo e a regiao de cada no.
        //
        // As coordenadas dos nos NAO ficam. Elas servem para classificar a
        // direcao de cada aresta enquanto o arquivo e lido, e morrem ali. Antes
        // viviam num vector<vector<int>> membro - 56 bytes por no para guardar
        // dois inteiros - junto com `edges`, `graphCoords` e `graphEdges`, que
        // eram preenchidos e nunca lidos por ninguem. Com um mapa de 5500x5500
        // isso era mais de 7 GB por jogo de estrutura morta.
        vector<int> neighbors;
        vector<int> nodeRegion;
        vector<vector<int>> regionNodes;

        Game();

        // Liga o jogo ao pool e diz de qual faixa de ids ele nasce dono. Os
        // ids sao globais e contiguos por jogo no setup; depois da primeira
        // viagem a faixa deixa de descrever quem esta aqui - quem descreve e
        // members.
        void configure(AgentPool& agentPool, int firstAgent, int numberOfHealthy, int numberOfInfected);

        void setSeed(unsigned int seed);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        [[nodiscard]] int generateLocation(int id) const;
        [[nodiscard]] int generateInfectedLocation(int id) const;
        [[nodiscard]] AgentRng agentEngine(int id, int turn) const;
        // Contagens dos agentes PRESENTES neste jogo. Com viagem ligada elas
        // mudam de turno para turno mesmo sem ninguem adoecer; quem conserva o
        // total e a Simulation, que soma o pool inteiro.
        [[nodiscard]] vector<int> countStates() const;
        [[nodiscard]] vector<int> countStatesByRegion() const;
        static uint64_t hashSeed(unsigned int seed, int id, int turn);
        void initializeGame(const string& fileName = "A1100a.txt");
        int claimSlot(int node, int preferredSlot);
        void startAllDumbThreads();
        void buildInfectedAtNode();

        // Um turno deste ambiente. O turno vem de fora: o relogio e da
        // Simulation, porque os jogos precisam andar juntos - o RNG e indexado
        // por (id, turno), e um agente que viaja tem que continuar no mesmo
        // turno do outro lado.
        void step(int turn);

        // Anota a saida do agente no outbox e libera o slot de origem. Nao
        // escreve no jogo de destino: isso acontece na barreira, em
        // acceptArrival.
        void travel(int id, int memberIndex, AgentRng& rng);

        // Poe um agente que chegou na lista deste jogo.
        //
        // SO PODE SER CHAMADO NA BARREIRA entre turnos, com ninguem percorrendo
        // a lista - e por isso que aqui a lista pode crescer, coisa que a versao
        // sequencial nao pode fazer. La um jogo escreve na lista de outro
        // enquanto o dono a percorre, entao a capacidade e fixa e o pior caso e
        // uma viagem recusada por lotacao. Aqui nao ha corrida, entao nao ha
        // recusa: nenhum agente e perdido por falta de espaco.
        void acceptArrival(const Departure& arrival);

        // Tira os -1 deixados por quem viajou. Roda entre turnos, quando
        // ninguem esta percorrendo a lista.
        void compactMembers();

        void reset();

    private:
        unsigned int baseSeed;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //GAME_H
