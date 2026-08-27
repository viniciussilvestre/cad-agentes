#ifndef GAMEC_PLAYER_H
#define GAMEC_PLAYER_H
#include <vector>
#include <random>

using namespace std;

class PlagueGame;

class Player {
    public:
        static int newId;
        int INFECTION_RATE = 20;
        int INCUBATION_TIME = 3;
        int RECOVERY_TIME = 10;
        double MORTALITY = 4.4;
        int id;
        int position;
        int SEIRState;
        int infectionCounter;
        int deathCounter;

        Player(int position, int SEIRState);

        [[nodiscard]] int getPosition() const;
        [[nodiscard]] int getSEIRState() const;
        void commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<int>& infectedAtNode, mt19937& rng);
        void deathOrRecovered(mt19937& rng);
        void infection(mt19937& rng);
        void checkSurroundings(const vector<int>& infectedAtNode);
        void updatePosition(const vector<int>& neighbors, int numberOfNodes, mt19937& rng);
};


#endif //GAMEC_PLAYER_H
