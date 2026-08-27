#ifndef GAMEC_PLAGUEGAME_H
#define GAMEC_PLAGUEGAME_H
#include <vector>
#include "Player.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <utility>
#include <random>

// Onde os infectados iniciais comecam. O passeio aleatorio leva centenas de
// turnos para misturar, entao a semeadura muda a dinamica por completo.
enum class SeedMode { Uniform, SingleRegion, SingleNode };

class PlagueGame
{
    public:
        // Estados SEIR: enum em vez de static int, para que instanciar um
        // segundo PlagueGame nao reescreva estado global.
        enum SEIRStates { SUSCEPTIBLE = 0, EXPOSED = 1, INFECTED = 2, RECOVERED = 3, DEAD = 4 };
        static const unsigned int DEFAULT_SEED = 42;
        // Grau fixo: 4 vizinhos de grade + 2 atalhos de longo alcance.
        static constexpr int SLOTS = 6;

        int lowerResolutionX = 62;
        int lowerResolutionY = 60;
        int resolutionX = 1920;
        int resolutionY = 1080;
        double widthPercentage;
        double heightPercentage;
        int initialNumberHealthy;
        int initialNumberInfected;
        static int totalNumberOfThreads;
        static std::vector<int> agentsPositionVector;
        static std::vector<int> agentsStatusVector;
        vector<Player> agentsVector;
        static int turns;
        vector<vector<int>> nodes;
        vector<vector<int>> edges;
        vector<vector<int>> graphCoords;
        int numberOfNodes;
        int numberOfRegions;
        // Vizinhanca de grau fixo, tamanho numberOfNodes * SLOTS.
        //
        // Layout SoA sobre os slots: indexado por neighbors[k * numberOfNodes + i],
        // e nao por [i * SLOTS + k]. Cada agente le um unico slot por turno, entao
        // agentes que sorteiem o mesmo k acabam lendo posicoes vizinhas - e o
        // layout que serve a uma versao GPU futura, e nao custa nada agora.
        //
        // Slots 0..3: vizinhos de grade (cima, baixo, direita, esquerda).
        // Slots 4..5: atalhos de longo alcance (e transbordo).
        // Slot sem aresta real: recebe o proprio i (auto-laco). Ver a nota longa
        // em initializeGame() - o auto-laco nao e padding, sustenta duas
        // propriedades do modelo.
        vector<int> neighbors;
        // Regiao de cada no e os nos de cada regiao.
        vector<int> nodeRegion;
        vector<vector<int>> regionNodes;
        // Numero de INFECTED em cada no, reconstruido no inicio de cada turno.
        // Para fins de contato basta saber se ha algum infectado no no, entao
        // checkSurroundings e O(1). touchedNodes guarda os nos escritos no turno
        // para que a limpeza custe o numero de agentes, e nao numberOfNodes.
        vector<int> infectedAtNode;
        vector<int> touchedNodes;
        using Point = pair<int, int>;
        using Edge = pair<Point, Point>;
        vector<Edge> graphEdges;
        int max_x_coordinate;
        int max_y_coordinate;
        int min_x_coordinate;
        int min_y_coordinate;
        bool isRunning;

        PlagueGame(int initialNumberHealthy, int initialNumberInfected);

        void setSeed(unsigned int seed);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        void setCsvOutput(const string& fileName);
        static unsigned int hashSeed(unsigned int seed, int id, int turn);
        [[nodiscard]] mt19937 agentEngine(int id, int turn) const;
        [[nodiscard]] int generateLocation(int id) const;
        [[nodiscard]] int generateInfectedLocation(int id) const;
        void initializeGame(const string& fileName = "A1100a.txt");
        void initiateThreads(int numberOfThreads, int SEIRState);
        void startAllDumbThreads();
        [[nodiscard]] pair<int, int> mapNodes(int x, int y, int x_min, int x_max, int y_min, int y_max) const;
        void buildInfectedAtNode();
        [[nodiscard]] vector<int> countStates() const;
        [[nodiscard]] vector<int> countStatesByRegion() const;
        void checkConservation(const vector<int>& counts) const;
        void reset();
        void quit();
        void remapNodes();
        void step();
        void run(int numberOfTurns);

    private:
        unsigned int baseSeed;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
        string csvFileName;

        int claimSlot(int node, int preferredSlot);
};

#endif //GAMEC_PLAGUEGAME_H
