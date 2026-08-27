#include "game.h"
#include "constants.h"

int Game::newId = 0;

Game::Game(int initialNumberHealthy, int initialNumberInfected)
{
    id = newId;
    newId++;
    turns = 0;
    isRunning = false;
    min_x_coordinate = numeric_limits<int>::max();
    max_x_coordinate = numeric_limits<int>::min();
    min_y_coordinate = numeric_limits<int>::max();
    max_y_coordinate = numeric_limits<int>::min();
    numberOfNodes = 0;
    numberOfRegions = 0;
    agentsPositionVector.assign((initialNumberHealthy + initialNumberInfected), 0);
    agentsStatusVector.assign((initialNumberHealthy + initialNumberInfected), States::SUSCEPTIBLE);
    this->initialNumberHealthy = initialNumberHealthy;
    this->initialNumberInfected = initialNumberInfected;
}

void Game::initiateThreads(int numberOfThreads, States state)
{
    for (int i = 0; i < numberOfThreads; i++)
    {
        int playerId = Player::newId;
        int startNode = generateLocation(playerId);
        agents.emplace_back(startNode, state, this);
        Player& player = agents.back();
        agentsPositionVector[i] = player.position;
        agentsStatusVector[i] = player.state;
    }
}

void Game::startAllDumbThreads()
{
    initiateThreads(initialNumberHealthy, States::SUSCEPTIBLE);
    initiateThreads(initialNumberInfected, States::INFECTED);
}

int Game::generateLocation(int id) const
{
    mt19937 rng = agentEngine(id, -1);
    int lower_bound = 0;
    int upper_bound = numberOfNodes - 1;
    uniform_int_distribution<int> uniform(lower_bound, upper_bound);
    return uniform(rng);
}

mt19937 Game::agentEngine(int id, int turn) const
{
    return mt19937(hashSeed(baseSeed, id, turn));
}

unsigned int Game::hashSeed(unsigned int seed, int id, int turn)
{
    uint64_t hash = splitMix64((uint64_t) seed);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) id);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) turn);
    return (unsigned int) (hash ^ (hash >> 32));
}

uint64_t Game::splitMix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

pair<int, int> Game::mapNodes(int x, int y, int x_min, int x_max, int y_min, int y_max) const
{
    int range_x = x_max - x_min;
    int range_y = y_max - y_min;
    int new_x = Constants::lowerResolutionX + (((x - x_min) * ((Constants::resolutionX - 62) - Constants::lowerResolutionX)) / range_x);
    int new_y = Constants::lowerResolutionY + (((y - y_min) * ((Constants::resolutionY - 60) - Constants::lowerResolutionY)) / range_y);
    return make_pair(new_x, new_y);
}

void Game::remapNodes()
{
    for (vector<int> & node: nodes)
    {
        int x;
        int y;
        tie(x, y) = mapNodes(node[0], node[1], min_x_coordinate, max_x_coordinate, min_y_coordinate, max_y_coordinate);
        node[0] = x;
        node[1] = y;
        graphCoords.push_back({node[0], node[1]});
    }

    for (vector<int> & edge: edges)
    {
        int edgeSource = edge[0];
        int edgeDestination = edge[1];
        if (edgeSource < 0 || edgeSource >= (int) nodes.size())
            continue;
        if (edgeDestination < 0 || edgeDestination >= (int) nodes.size())
            continue;
        int xSource = nodes[edgeSource][0];
        int ySource = nodes[edgeSource][1];
        int xDestination = nodes[edgeDestination][0];
        int yDestination = nodes[edgeDestination][1];
        Point source = {xSource, ySource};
        Point destination = {xDestination, yDestination};
        Edge connection = {source, destination};
        graphEdges.emplace_back(connection);
    }
}

int cardinalSlot(int deltaX, int deltaY)
{
    if (abs(deltaY) >= abs(deltaX))
    {
        return (deltaY < 0) ? 0 : 1;
    }
    return (deltaX > 0) ? 2 : 3;
}

int Game::claimSlot(int node, int preferredSlot)
{
    if (neighbors[preferredSlot * numberOfNodes + node] == -1)
    {
        return preferredSlot;
    }
    for (int slot = 4; slot < Constants::SLOTS; slot++)
    {
        if (neighbors[slot * numberOfNodes + node] == -1)
        {
            return slot;
        }
    }
    for (int slot = 0; slot < 4; slot++)
    {
        if (neighbors[slot * numberOfNodes + node] == -1)
        {
            return slot;
        }
    }
    return -1;
}

void Game::buildInfectedAtNode()
{
    for (int node : touchedNodes)
    {
        infectedAtNode[node] = 0;
    }
    touchedNodes.clear();

    for (const Player& agent : agents)
    {
        if (agent.state == States::INFECTED)
        {
            if (infectedAtNode[agent.position] == 0)
            {
                touchedNodes.emplace_back(agent.position);
            }
            infectedAtNode[agent.position]++;
        }
    }
}

vector<int> Game::countStates() const
{
    vector<int> counts(5, 0);
    for (States state : agentsStatusVector)
    {
        counts[static_cast<int>(state)]++;
    }
    return counts;
}

vector<int> Game::countStatesByRegion() const
{
    vector<int> counts((size_t) numberOfRegions * 5, 0);
    for (int i = 0; i < (initialNumberHealthy + initialNumberInfected); i++)
    {
        int region = nodeRegion[agentsPositionVector[i]];
        counts[region * 5 + static_cast<int>(agentsStatusVector[i])]++;
    }
    return counts;
}

void Game::checkConservation(const vector<int>& counts) const
{
    int totalNumberOfThreads = initialNumberHealthy + initialNumberInfected;
    int total = 0;
    for (int count : counts)
    {
        total += count;
    }
    if (total != totalNumberOfThreads)
    {
        cerr << "Invariante de conservacao violada no turno " << turns << ": S+E+I+R+D = "
             << total << " != " << totalNumberOfThreads << endl;
        exit(EXIT_FAILURE);
    }
}

void Game::step()
{
    buildInfectedAtNode();
    for (Player& agent : agents)
    {
        if (agent.state == States::DEAD)
        {
            continue;
        }
        mt19937 rng = agentEngine(agent.id, turns);
        agent.commonRoutine(neighbors, numberOfNodes, infectedAtNode, rng);
    }
    turns++;
}

void Game::reset()
{
    Player::newId = 0;
    agents.clear();
    fill(infectedAtNode.begin(), infectedAtNode.end(), 0);
    touchedNodes.clear();
    fill(agentsPositionVector.begin(), agentsPositionVector.end(), 0);
    fill(agentsStatusVector.begin(), agentsStatusVector.end(), States::SUSCEPTIBLE);
    turns = 0;
    isRunning = false;
}

void Game::quit()
{
    reset();
}