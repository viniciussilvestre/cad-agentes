#include "player.h"
#include "constants.h"
#include "game.h"

int Player::newId = 0;

Player::Player(int position, States state, Game* game)
{
    id = newId;
    newId++;
    this->position = position;
    this->state = state;
    this->game = game;
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
    double lower_bound = 0;
    double upper_bound = 100;
    uniform_real_distribution<double> uniform(lower_bound, upper_bound);
    double probability = uniform(rng);

    infectionCounter++;
    if (infectionCounter <= Constants::EXPOSED_TIME)
    {
        if (probability <= Constants::INFECTION_RATE)
        {
            state = States::INFECTED;
        }
        else
        {
            state = States::SUSCEPTIBLE;
            infectionCounter = 0;
        }
        game->agentsStatusVector[id] = state;
    }
    if (infectionCounter > Constants::EXPOSED_TIME && infectionCounter <= Constants::INFECTION_TIME)
    {
        if (probability <= Constants::MORTALITY)
        {
            state = States::DEAD;
            game->agentsStatusVector[id] = state;
        }
    }
    if (infectionCounter == Constants::INFECTION_TIME)
    {
        state = States::RECOVERED;
        infectionCounter = 0;
    }
}

void Player::commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<int>& infectedAtNode, mt19937& rng)
{
    if (state == States::DEAD)
    {
        return;
    }

    updatePosition(neighbors, numberOfNodes, rng);

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
    if (infectedAtNode[position] > 0)
    {
        state = States::EXPOSED;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
    }
}

void Player::updatePosition(const vector<int>& neighbors, int numberOfNodes, mt19937& rng)
{
    uniform_int_distribution<int> uniform(0, Constants::SLOTS - 1);
    int slot = uniform(rng);
    position = neighbors[slot * numberOfNodes + position];
    game->agentsPositionVector[id] = position;
}