#ifndef GAME_H
#define GAME_H
#include "player.h"
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
        // Bandeira por no, e nao contagem: checkSurroundings so pergunta se ha
        // ALGUM infectado ali. Ver buildInfectedAtNode.
        vector<unsigned char> infectedAtNode;
        vector<Player> agents;
        vector<int> agentsPositionVector;
        vector<States> agentsStatusVector;
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
        void step();
        void run(int numberOfTurns);
        void reset();
        void quit();

    private:
        unsigned int baseSeed;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
        string csvFileName;
};

#endif //GAME_H
