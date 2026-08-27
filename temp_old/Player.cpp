#include "Player.h"
#include "PlagueGame.h"

int Player::newId = 0;

Player::Player(int position, int SEIRState)
{
    id = newId;
    newId++;
    this->position = position;
    this->SEIRState = SEIRState;
    infectionCounter = 0;
    deathCounter = 0;
}

int Player::getPosition() const
{
    return position;
}

int Player::getSEIRState() const
{
    return SEIRState;
}

void Player::commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<int>& infectedAtNode, mt19937& rng)
{
    // DEAD e terminal: nao se move, nao transmite, nao transiciona.
    if (SEIRState == PlagueGame::DEAD)
    {
        return;
    }

    updatePosition(neighbors, numberOfNodes, rng);

    // Cada agente executa no maximo uma transicao por turno: se checkSurroundings
    // acabou de expor o agente, ele nao avanca a incubacao no mesmo turno.
    int previousState = SEIRState;
    checkSurroundings(infectedAtNode);
    if (SEIRState != previousState)
    {
        return;
    }

    if (SEIRState == PlagueGame::EXPOSED)
    {
        infection(rng);
    }
    else if (SEIRState == PlagueGame::INFECTED)
    {
        deathOrRecovered(rng);
    }
}

void Player::deathOrRecovered(mt19937& rng)
{
    if (SEIRState != PlagueGame::INFECTED)
    {
        return;
    }
    deathCounter++;
    if (deathCounter == RECOVERY_TIME)
    {
        double lower_bound = 0;
        double upper_bound = 100;
        uniform_real_distribution<double> uniform(lower_bound, upper_bound);
        double probability = uniform(rng);
        if (probability <= MORTALITY)
        {
            SEIRState = PlagueGame::DEAD;
        }
        else
        {
            SEIRState = PlagueGame::RECOVERED;
        }
        deathCounter = 0;
        PlagueGame::agentsStatusVector[id] = SEIRState;
    }
}

void Player::infection(mt19937& rng)
{
    // So um agente EXPOSED incuba: sem este guarda um SUSCEPTIBLE poderia
    // virar INFECTED sem nunca ter sido exposto.
    if (SEIRState != PlagueGame::EXPOSED)
    {
        return;
    }
    infectionCounter++;
    if (infectionCounter == INCUBATION_TIME)
    {
        double lower_bound = 0;
        double upper_bound = 100;
        uniform_real_distribution<double> uniform(lower_bound, upper_bound);
        double probability = uniform(rng);
        if (probability <= INFECTION_RATE)
        {
            SEIRState = PlagueGame::INFECTED;
            deathCounter = 0;
        }
        else
        {
            SEIRState = PlagueGame::SUSCEPTIBLE;
        }
        infectionCounter = 0;
        PlagueGame::agentsStatusVector[id] = SEIRState;
    }
}

void Player::checkSurroundings(const vector<int>& infectedAtNode)
{
    if (SEIRState != PlagueGame::SUSCEPTIBLE && SEIRState != PlagueGame::RECOVERED)
    {
        return;
    }
    // O(1): para fins de contato basta saber se ha algum INFECTED no no. O
    // proprio agente e SUSCEPTIBLE ou RECOVERED aqui, entao nunca se conta a si
    // mesmo. infectedAtNode e reconstruido pelo PlagueGame no inicio do turno.
    if (infectedAtNode[position] > 0)
    {
        SEIRState = PlagueGame::EXPOSED;
        infectionCounter = 0;
        PlagueGame::agentsStatusVector[id] = SEIRState;
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
    uniform_int_distribution<int> uniform(0, PlagueGame::SLOTS - 1);
    int slot = uniform(rng);
    position = neighbors[slot * numberOfNodes + position];
    PlagueGame::agentsPositionVector[id] = position;
}
