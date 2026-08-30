#ifndef POOL_H
#define POOL_H
#include "enums.h"
#include <vector>

using namespace std;

// Estado de TODOS os agentes, indexado por id global, uma unica vez.
//
// O pool e global e os jogos sao ambientes que apontam para ele. A separacao
// existe por causa da viagem entre jogos: um agente que muda de ambiente nao
// pode ter o estado copiado de um lugar para outro, senao a viagem custaria
// proporcional ao tamanho do agente e abriria a janela em que ele existe duas
// vezes ou nenhuma. Com o pool global, mudar de jogo e mexer numa lista de
// inteiros - o estado nunca se move.
//
// Continua sendo SoA de 6 bytes por agente (ver player.h). O id e a posicao no
// pool e nao muda nunca, o que tambem mantem o RNG indexado por (id, turno)
// valendo: viajar nao altera o fluxo de sorteios de ninguem.
struct AgentPool
{
    vector<int> positions;
    vector<States> states;
    vector<unsigned char> counters;

    void resize(int numberOfAgents)
    {
        positions.assign((size_t) numberOfAgents, 0);
        states.assign((size_t) numberOfAgents, States::SUSCEPTIBLE);
        counters.assign((size_t) numberOfAgents, 0);
    }

    [[nodiscard]] int size() const
    {
        return (int) states.size();
    }
};

#endif //POOL_H
