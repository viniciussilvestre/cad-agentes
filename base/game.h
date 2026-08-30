#ifndef GAME_H
#define GAME_H
#include "player.h"
#include "pool.h"
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
        // viagem entre jogos precisa mexer.
        vector<int> members;
        vector<vector<int>> nodes;
        vector<vector<int>> edges;
        vector<vector<int>> graphCoords;
        vector<Edge> graphEdges;
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
        [[nodiscard]] pair<int, int> mapNodes(int x, int y, int x_min, int x_max, int y_min, int y_max) const;
        // Contagens dos agentes PRESENTES neste jogo. Com viagem ligada elas
        // mudam de turno para turno mesmo sem ninguem adoecer; quem conserva o
        // total e a Simulation, que soma o pool inteiro.
        [[nodiscard]] vector<int> countStates() const;
        [[nodiscard]] vector<int> countStatesByRegion() const;
        static uint64_t hashSeed(unsigned int seed, int id, int turn);
        void initializeGame(const string& fileName = "A1100a.txt");
        void remapNodes();
        int claimSlot(int node, int preferredSlot);
        void startAllDumbThreads();
        void buildInfectedAtNode();

        // Um turno deste ambiente. O turno vem de fora: o relogio e da
        // Simulation, porque os jogos precisam andar juntos - o RNG e indexado
        // por (id, turno), e um agente que viaja tem que continuar no mesmo
        // turno do outro lado.
        void step(int turn);
        void reset();

    private:
        unsigned int baseSeed;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //GAME_H
