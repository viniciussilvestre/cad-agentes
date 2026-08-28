#include "player.h"
#include "constants.h"
#include "game.h"

Player::Player(int id, int position, States state, Game* game)
{
    this->id = id;
    this->position = position;
    this->state = state;
    this->game = game;
    // O contador e a posicao do agente no cronograma de INFECTION_TIME turnos.
    // Quem ja nasce INFECTED nao passa pela incubacao: comeca no fim da janela
    // de exposicao, senao no turno EXPOSED_TIME cairia no sorteio de
    // INFECTION_RATE e 80% dos infectados iniciais virariam suscetiveis.
    infectionCounter = (state == States::INFECTED) ? Constants::EXPOSED_TIME : 0;
}

int Player::getPosition() const
{
    return position;
}

States Player::getState() const
{
    return state;
}

void Player::infection(mt19937& rng)
{
    // Cronograma de INFECTION_TIME turnos com um unico contador:
    //
    //   1 .. EXPOSED_TIME-1   incubando, nenhum sorteio
    //   == EXPOSED_TIME       sorteia UMA vez se a exposicao virou doenca
    //   .. INFECTION_TIME     os DEATH_TIME turnos finais: sorteia a morte a
    //                         cada um deles
    //   == INFECTION_TIME     sobreviveu aos tres sorteios -> RECOVERED
    //
    // Nao consumir o RNG nos turnos de incubacao e inofensivo: o engine e novo
    // a cada turno, indexado por (id, turno), entao nao existe fluxo continuo
    // que pudesse ficar defasado.
    infectionCounter++;

    if (infectionCounter < Constants::EXPOSED_TIME)
    {
        return;
    }

    double lower_bound = 0;
    double upper_bound = 100;
    uniform_real_distribution<double> uniform(lower_bound, upper_bound);
    double probability = uniform(rng);

    if (infectionCounter == Constants::EXPOSED_TIME)
    {
        if (probability <= Constants::INFECTION_RATE)
        {
            state = States::INFECTED;
        }
        else
        {
            // A exposicao nao virou doenca: o agente volta a ser suscetivel e
            // pode ser exposto de novo num turno seguinte.
            state = States::SUSCEPTIBLE;
            infectionCounter = 0;
        }
        game->agentsStatusVector[id] = state;
        return;
    }

    // DEAD e terminal, e o return aqui e o que impede a ressurreicao: sem ele,
    // quem morre no turno INFECTION_TIME cairia no bloco de RECOVERED logo
    // abaixo e voltaria a vida.
    if (probability <= Constants::MORTALITY)
    {
        state = States::DEAD;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
        return;
    }

    if (infectionCounter == Constants::INFECTION_TIME)
    {
        state = States::RECOVERED;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
    }
}

void Player::commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<int>& infectedAtNode, mt19937& rng)
{
    // DEAD e terminal: nao se move, nao transmite, nao transiciona.
    if (state == States::DEAD)
    {
        return;
    }

    updatePosition(neighbors, numberOfNodes, rng);

    // Cada agente executa no maximo uma transicao por turno: se checkSurroundings
    // acabou de expor o agente, ele nao avanca a incubacao no mesmo turno.
    States previousState = state;
    checkSurroundings(infectedAtNode);
    if (state != previousState)
    {
        return;
    }

    if (state == States::EXPOSED || state == States::INFECTED)
    {
        infection(rng);
    }
}

void Player::checkSurroundings(const vector<int>& infectedAtNode)
{
    if (state != States::SUSCEPTIBLE && state != States::RECOVERED)
    {
        return;
    }
    // O(1): para fins de contato basta saber se ha algum INFECTED no no. O
    // proprio agente e SUSCEPTIBLE ou RECOVERED aqui, entao nunca se conta a si
    // mesmo. infectedAtNode e reconstruido pelo Game no inicio do turno.
    if (infectedAtNode[position] > 0)
    {
        state = States::EXPOSED;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
    }
}

void Player::updatePosition(const vector<int>& neighbors, int numberOfNodes, mt19937& rng)
{
    // Grau fixo: todo no tem exatamente SLOTS slots e os vazios apontam para o
    // proprio no. Sorteia-se um slot e pronto - sem laco de rejeicao, sem -1,
    // sem checagem de limites, sem branch.
    //
    // Layout SoA sobre os slots: neighbors[k * numberOfNodes + i]. Agentes que
    // sorteiem o mesmo k leem posicoes proximas na memoria.
    uniform_int_distribution<int> uniform(0, Constants::SLOTS - 1);
    int slot = uniform(rng);
    position = neighbors[slot * numberOfNodes + position];
    game->agentsPositionVector[id] = position;
}
