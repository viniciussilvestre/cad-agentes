#ifndef PLAYER_H
#define PLAYER_H
#include "enums.h"
#include "rng.h"
#include <vector>

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
        void commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<unsigned char>& infectedAtNode, AgentRng& rng);
        void infection(AgentRng& rng);
        void checkSurroundings(const vector<unsigned char>& infectedAtNode);
        void updatePosition(const vector<int>& neighbors, int numberOfNodes, AgentRng& rng);
};

#endif //PLAYER_H
