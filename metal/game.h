#ifndef GAME_H
#define GAME_H
#include "enums.h"
#include "rng.h"
#include "gpu.h"
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
        int initialNumberHealthy;
        int initialNumberInfected;
        int turns;
        int max_x_coordinate;
        int max_y_coordinate;
        int min_x_coordinate;
        int min_y_coordinate;
        int numberOfNodes;
        int numberOfRegions;
        bool isRunning;
        // Bandeira por no, e nao contagem: o unico consumidor testa "> 0". E o
        // que o kernel markInfected ja fazia; o caminho de CPU agora faz igual.
        vector<unsigned char> infectedAtNode;

        // Estado dos agentes em SoA - um vetor por campo, e nao um
        // vector<Player>.
        //
        // Nao e so arrumacao: e o que a GPU exige. Um MTLBuffer e um bloco
        // plano de um tipo so, sem ponteiro e sem padding, e o Player carregava
        // um Game* de 8 bytes que nao faz sentido nenhum do outro lado.
        //
        // De quebra some a duplicacao que existia entre Player.position e
        // agentsPositionVector: agora ha uma fonte unica por campo, e a classe
        // de bug em que os dois dessincronizam deixa de ser possivel.
        // 6 bytes por agente: 4 de posicao, 1 de estado, 1 de contador. O
        // contador nunca passa de EXPOSED_TIME (7) e o estado vai de 0 a 4,
        // entao os 4 bytes que cada um usava eram 3 desperdicados - por agente,
        // por buffer, dos dois lados da fronteira.
        vector<int> agentsPositionVector;
        vector<States> agentsStatusVector;
        vector<unsigned char> agentCounter;
        vector<vector<int>> nodes;
        vector<vector<int>> edges;
        vector<vector<int>> graphCoords;
        vector<Edge> graphEdges;
        vector<int> neighbors;
        vector<int> nodeRegion;
        vector<vector<int>> regionNodes;

        Game(int initialNumberHealthy, int initialNumberInfected);

        void setSeed(unsigned int seed);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        void setCsvOutput(const string& fileName);
        [[nodiscard]] int generateLocation(int id) const;
        [[nodiscard]] int generateInfectedLocation(int id) const;
        [[nodiscard]] AgentRng agentEngine(int id, int turn) const;
        [[nodiscard]] pair<int, int> mapNodes(int x, int y, int x_min, int x_max, int y_min, int y_max) const;
        [[nodiscard]] vector<int> countStates() const;
        [[nodiscard]] vector<int> countStatesByRegion() const;
        static uint64_t hashSeed(unsigned int seed, int id, int turn);
        void checkConservation(const vector<int>& counts) const;
        void initializeGame(const string& fileName = "A1100a.txt");
        void initiateThreads(int numberOfThreads, States state);
        void remapNodes();
        int claimSlot(int node, int preferredSlot);
        void startAllDumbThreads();
        void buildInfectedAtNode();
        void stepAgent(int i, AgentRng& rng);
        void step();

        // Liga a GPU. Precisa vir depois de startAllDumbThreads(), porque e o
        // estado inicial dos agentes que sobe para os buffers. Se falhar, avisa
        // e o jogo segue na CPU - a saida e a mesma, so mais lenta.
        void enableGpu(const string& metallibPath);
        void run(int numberOfTurns);
        void reset();
        void quit();

    private:
        // Com a GPU ligada, os vetores de agente acima ficam so como area de
        // setup: a verdade passa a ser a memoria dos MTLBuffer. Como e memoria
        // unificada, ler de la e ler memoria comum - as contagens por turno nao
        // custam transferencia nenhuma.
        bool useGpu = false;
        GpuRunner gpu;

        unsigned int baseSeed;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
        string csvFileName;
};

#endif //GAME_H
