#ifndef GAME_H
#define GAME_H
#include "player.h"
#include <vector>
#include <random>
#include <iostream>

using namespace std;
using Point = pair<int, int>;
using Edge = pair<Point, Point>;

class Game
{
    public:
        static int newId;
        int id;
        int initialNumberHealthy;
        int initialNumberInfected;
        int turns;
        int max_x_coordinate;
        int max_y_coordinate;
        int min_x_coordinate;
        int min_y_coordinate;
        double widthPercentage;
        double heightPercentage;
        int numberOfNodes;
        int numberOfRegions;
        bool isRunning;
        vector<int> infectedAtNode;
        vector<int> touchedNodes;
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

        [[nodiscard]] int generateLocation(int id) const;
        [[nodiscard]] mt19937 agentEngine(int id, int turn) const;
        [[nodiscard]] pair<int, int> mapNodes(int x, int y, int x_min, int x_max, int y_min, int y_max) const;
        [[nodiscard]] vector<int> countStates() const;
        [[nodiscard]] vector<int> countStatesByRegion() const;
        static unsigned int hashSeed(unsigned int seed, int id, int turn);
        void checkConservation(const vector<int>& counts) const;
        void initiateThreads(int numberOfThreads, States state);
        void remapNodes();
        int claimSlot(int deltaX, int deltaY);
        int cardinalSlot(int node, int preferredSlot);
        void startAllDumbThreads();
        void buildInfectedAtNode();
        void step();
        void reset();
        void quit();

    private:
        unsigned int baseSeed;
        static uint64_t splitMix64(uint64_t value);
};

#endif //GAME_H