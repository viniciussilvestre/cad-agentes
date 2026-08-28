#ifndef PLAYER_H
#define PLAYER_H
#include "enums.h"
#include <vector>
#include <random>

using namespace std;

class Game;

class Player
{
    private:

    public:
        // O id e local ao jogo dono do agente: vale como indice direto em
        // agentsPositionVector e agentsStatusVector. Nao existe contador global,
        // senao o segundo Game criaria agentes com id fora dos limites dos
        // proprios vetores.
        int id;
        int position;
        int infectionCounter;
        Game* game;
        States state;

        Player(int id, int position, States state, Game* game);

        [[nodiscard]] int getPosition() const;
        [[nodiscard]] States getState() const;
        void commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<int>& infectedAtNode, mt19937& rng);
        void infection(mt19937& rng);
        void checkSurroundings(const vector<int>& infectedAtNode);
        void updatePosition(const vector<int>& neighbors, int numberOfNodes, mt19937& rng);
};

#endif //PLAYER_H
