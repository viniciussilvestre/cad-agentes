#include "game.h"
#include "constants.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

Game::Game(int initialNumberHealthy, int initialNumberInfected)
{
    turns = 0;
    isRunning = false;
    min_x_coordinate = numeric_limits<int>::max();
    max_x_coordinate = numeric_limits<int>::min();
    min_y_coordinate = numeric_limits<int>::max();
    max_y_coordinate = numeric_limits<int>::min();
    numberOfNodes = 0;
    numberOfRegions = 0;
    baseSeed = Constants::DEFAULT_SEED;
    agentsPositionVector.assign((initialNumberHealthy + initialNumberInfected), 0);
    agentsStatusVector.assign((initialNumberHealthy + initialNumberInfected), States::SUSCEPTIBLE);
    this->initialNumberHealthy = initialNumberHealthy;
    this->initialNumberInfected = initialNumberInfected;
}

void Game::setSeed(unsigned int seed)
{
    baseSeed = seed;
}

void Game::setSeedMode(SeedMode mode, int regionOrNode)
{
    seedMode = mode;
    seedTarget = regionOrNode;
}

void Game::setCsvOutput(const string& fileName)
{
    csvFileName = fileName;
}

void Game::initiateThreads(int numberOfThreads, States state)
{
    for (int i = 0; i < numberOfThreads; i++)
    {
        // agents e a unica dona dos agentes: o id e a posicao na qual o agente
        // entra no vetor, entao serve de indice direto nos vetores consolidados.
        int playerId = (int) agents.size();
        int startNode = (state == States::INFECTED) ? generateInfectedLocation(playerId) : generateLocation(playerId);
        agents.emplace_back(playerId, startNode, state, this);
        Player& player = agents.back();
        agentsPositionVector[player.id] = player.position;
        agentsStatusVector[player.id] = player.state;
    }
}

void Game::startAllDumbThreads()
{
    initiateThreads(initialNumberHealthy, States::SUSCEPTIBLE);
    initiateThreads(initialNumberInfected, States::INFECTED);
}

int Game::generateLocation(int id) const
{
    // Turno -1 marca o sorteio de setup, fora do laco de turnos.
    mt19937 rng = agentEngine(id, -1);
    int lower_bound = 0;
    int upper_bound = numberOfNodes - 1;
    uniform_int_distribution<int> uniform(lower_bound, upper_bound);
    return uniform(rng);
}

int Game::generateInfectedLocation(int id) const
{
    // Os suscetiveis continuam sempre uniformes; so os infectados iniciais
    // seguem o modo de semeadura.
    if (seedMode == SeedMode::SingleNode)
    {
        if (seedTarget < 0 || seedTarget >= numberOfNodes)
        {
            cerr << "Erro: semeadura SingleNode aponta para o no " << seedTarget
                 << ", fora de [0, " << numberOfNodes << ")." << endl;
            exit(EXIT_FAILURE);
        }
        return seedTarget;
    }
    if (seedMode == SeedMode::SingleRegion)
    {
        if (seedTarget < 0 || seedTarget >= numberOfRegions)
        {
            cerr << "Erro: semeadura SingleRegion aponta para a regiao " << seedTarget
                 << ", fora de [0, " << numberOfRegions << ")." << endl;
            exit(EXIT_FAILURE);
        }
        const vector<int>& candidates = regionNodes[seedTarget];
        if (candidates.empty())
        {
            cerr << "Erro: a regiao " << seedTarget << " nao tem nenhum no." << endl;
            exit(EXIT_FAILURE);
        }
        // Turno -2: sorteio de setup separado do uniforme, para que trocar o
        // modo de semeadura nao desloque as posicoes dos suscetiveis.
        mt19937 rng = agentEngine(id, -2);
        uniform_int_distribution<int> uniform(0, (int) candidates.size() - 1);
        return candidates[uniform(rng)];
    }
    return generateLocation(id);
}

mt19937 Game::agentEngine(int id, int turn) const
{
    // O RNG e indexado por (id do agente, turno), nunca por um fluxo sequencial
    // compartilhado: os sorteios de um agente num turno dependem so desses tres
    // valores, e nao da ordem em que os agentes rodam. Assim a saida continua
    // reprodutivel se o laco de agentes for paralelizado depois.
    return mt19937(hashSeed(baseSeed, id, turn));
}

unsigned int Game::hashSeed(unsigned int seed, int id, int turn)
{
    uint64_t hash = splitMix64((uint64_t) seed);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) id);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) turn);
    return (unsigned int) (hash ^ (hash >> 32));
}

// Mistura splitmix64: avalanche boa o suficiente para que sementes vizinhas
// (id e turno consecutivos) produzam fluxos sem correlacao.
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

// Escolhe o slot cardinal de uma aresta a partir do deslocamento entre as
// coordenadas dos dois nos. A regra e simetrica: se v esta a direita de u,
// entao u esta a esquerda de v, logo os dois lados concordam sobre o eixo e
// recebem slots opostos.
static int cardinalSlot(int deltaX, int deltaY)
{
    if (abs(deltaY) >= abs(deltaX))
    {
        return (deltaY < 0) ? 0 : 1;   // cima, baixo
    }
    return (deltaX > 0) ? 2 : 3;       // direita, esquerda
}

int Game::claimSlot(int node, int preferredSlot)
{
    if (neighbors[preferredSlot * numberOfNodes + node] == -1)
    {
        return preferredSlot;
    }
    // Slot cardinal ocupado (um atalho apontando na mesma direcao do vizinho
    // de grade): transborda para os slots de atalho e, no limite, para os
    // cardinais livres.
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

void Game::initializeGame(const string& fileName)
{
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Erro: nao foi possivel abrir o arquivo do grafo '" << fileName << "'." << endl;
        exit(EXIT_FAILURE);
    }

    vector<string> lines;
    string line;
    while (getline(file, line)) {
        lines.push_back(line);
    }
    file.close();

    if (lines.empty()) {
        cerr << "Erro: o arquivo do grafo '" << fileName << "' esta vazio." << endl;
        exit(EXIT_FAILURE);
    }

    vector<string> items;
    stringstream iss(lines[0]);
    copy(istream_iterator<string>(iss), istream_iterator<string>(), back_inserter(items));
    if (items.size() < 3) {
        cerr << "Erro: cabecalho invalido em '" << fileName
             << "': esperado 'numero_de_nos numero_de_arestas numero_de_regioes'." << endl;
        exit(EXIT_FAILURE);
    }
    numberOfRegions = stoi(items[2]);
    if (numberOfRegions < 1) {
        cerr << "Erro: numero de regioes invalido em '" << fileName << "': " << numberOfRegions << "." << endl;
        exit(EXIT_FAILURE);
    }

    // Primeira passada: os nos. As coordenadas cruas sao guardadas antes de
    // remapNodes() porque a classificacao cardinal das arestas depende delas -
    // o remapeamento escala X e Y por fatores diferentes e distorceria o eixo
    // dominante.
    int counter = 0;
    size_t edgeStart = lines.size();
    for (size_t i = 1; i < lines.size(); ++i) {
        line = lines[i];
        if (line.empty()) {
            counter++;
            if (counter == 2) {
                edgeStart = i + 1;
                break;
            }
            continue;
        }
        if (counter == 1) {
            vector<string> values;
            stringstream iss2(line);
            copy(istream_iterator<string>(iss2), istream_iterator<string>(), back_inserter(values));
            if (values.size() < 3) {
                cerr << "Erro: linha de no invalida em '" << fileName << "' (linha " << (i + 1)
                     << "): esperado 'x y regiao'." << endl;
                exit(EXIT_FAILURE);
            }
            int x_coordinate = stoi(values[0]);
            int y_coordinate = stoi(values[1]);
            int region = stoi(values[2]);
            if (region < 0 || region >= numberOfRegions) {
                cerr << "Erro: no " << nodes.size() << " tem regiao " << region
                     << ", fora de [0, " << numberOfRegions << ")." << endl;
                exit(EXIT_FAILURE);
            }
            nodes.push_back({x_coordinate, y_coordinate});
            nodeRegion.push_back(region);
            if (x_coordinate > max_x_coordinate)
                max_x_coordinate = x_coordinate;
            if (y_coordinate > max_y_coordinate)
                max_y_coordinate = y_coordinate;
            if (x_coordinate < min_x_coordinate)
                min_x_coordinate = x_coordinate;
            if (y_coordinate < min_y_coordinate)
                min_y_coordinate = y_coordinate;
        }
    }

    if (nodes.empty()) {
        cerr << "Erro: nenhum no lido de '" << fileName << "'." << endl;
        exit(EXIT_FAILURE);
    }
    numberOfNodes = (int) nodes.size();

    // Construcao direta da vizinhanca de grau fixo. Sem matriz densa: com
    // 40.000 nos uma matriz nos x nos passaria de 6 GB e nem alocaria.
    neighbors.assign((size_t) numberOfNodes * Constants::SLOTS, -1);
    vector<int> degree(numberOfNodes, 0);

    for (size_t i = edgeStart; i < lines.size(); ++i) {
        line = lines[i];
        if (line.empty()) {
            continue;
        }
        vector<string> values;
        stringstream iss3(line);
        copy(istream_iterator<string>(iss3), istream_iterator<string>(), back_inserter(values));
        if (values.size() < 2) {
            cerr << "Erro: linha de aresta invalida em '" << fileName << "' (linha " << (i + 1)
                 << "): esperado 'origem destino peso'." << endl;
            exit(EXIT_FAILURE);
        }
        int source_node = stoi(values[0]);
        int destination_node = stoi(values[1]);
        if (source_node < 0 || source_node >= numberOfNodes ||
            destination_node < 0 || destination_node >= numberOfNodes) {
            cerr << "Erro: aresta (" << source_node << ", " << destination_node
                 << ") em '" << fileName << "' referencia no fora de [0, " << numberOfNodes << ")." << endl;
            exit(EXIT_FAILURE);
        }
        if (source_node == destination_node) {
            continue;
        }
        edges.push_back({source_node, destination_node});

        degree[source_node]++;
        degree[destination_node]++;
        if (degree[source_node] > Constants::SLOTS) {
            cerr << "Erro: no " << source_node << " tem grau " << degree[source_node]
                 << ", acima do teto de " << Constants::SLOTS << " arestas." << endl;
            exit(EXIT_FAILURE);
        }
        if (degree[destination_node] > Constants::SLOTS) {
            cerr << "Erro: no " << destination_node << " tem grau " << degree[destination_node]
                 << ", acima do teto de " << Constants::SLOTS << " arestas." << endl;
            exit(EXIT_FAILURE);
        }

        int deltaX = nodes[destination_node][0] - nodes[source_node][0];
        int deltaY = nodes[destination_node][1] - nodes[source_node][1];
        int forwardSlot = claimSlot(source_node, cardinalSlot(deltaX, deltaY));
        int backwardSlot = claimSlot(destination_node, cardinalSlot(-deltaX, -deltaY));
        if (forwardSlot == -1 || backwardSlot == -1) {
            cerr << "Erro: sem slot livre para a aresta (" << source_node << ", "
                 << destination_node << "); teto de " << Constants::SLOTS << " arestas por no." << endl;
            exit(EXIT_FAILURE);
        }
        neighbors[forwardSlot * numberOfNodes + source_node] = destination_node;
        neighbors[backwardSlot * numberOfNodes + destination_node] = source_node;
    }

    // ------------------------------------------------------------------
    // AUTO-LACO NOS SLOTS VAZIOS - duas razoes, nenhuma delas padding.
    //
    // 1. Distribuicao estacionaria uniforme. Com todo no tendo exatamente
    //    SLOTS slots e a relacao sendo simetrica (se u tem v num slot, v tem u
    //    num slot), a matriz de transicao e simetrica, logo duplamente
    //    estocastica, logo a densidade de agentes converge para exatamente
    //    uniforme. Sem o auto-laco a estacionaria seria proporcional ao grau e
    //    as bordas do mapa ficariam permanentemente subpovoadas - no esquema
    //    antigo, no de canto com 0,62x a densidade do interior, razao
    //    maximo/minimo de 1,67. Ver testUniformStationary().
    //
    // 2. Quebra da bipartição. Uma grade e bipartida (coloracao de xadrez). Se
    //    o agente fosse obrigado a se mover todo turno, sua paridade
    //    (linha + coluna) % 2 alternaria de forma deterministica, e dois
    //    agentes de paridade inicial oposta nunca ocupariam o mesmo no -
    //    metade de todos os pares seria incapaz de transmitir a doenca, para
    //    sempre. O auto-laco torna o passeio lazy e destroi a periodicidade,
    //    inclusive com --shortcuts 0. Ver testLaziness().
    // ------------------------------------------------------------------
    for (int slot = 0; slot < Constants::SLOTS; slot++)
    {
        for (int node = 0; node < numberOfNodes; node++)
        {
            if (neighbors[slot * numberOfNodes + node] == -1)
            {
                neighbors[slot * numberOfNodes + node] = node;
            }
        }
    }

    regionNodes.assign(numberOfRegions, vector<int>());
    for (int node = 0; node < numberOfNodes; node++)
    {
        regionNodes[nodeRegion[node]].emplace_back(node);
    }

    infectedAtNode.assign(numberOfNodes, 0);
    touchedNodes.clear();

    remapNodes();
}

void Game::buildInfectedAtNode()
{
    // Zera so os nos escritos no turno anterior. Com numberOfNodes muito maior
    // que o numero de agentes, limpar o vetor inteiro custaria mais que o
    // trabalho util (40.000 nos contra alguns milhares de agentes).
    for (int node : touchedNodes)
    {
        infectedAtNode[node] = 0;
    }
    touchedNodes.clear();

    // Uma unica passada sobre agents. DEAD nunca e INFECTED, entao nao
    // transmite nem conta como contato.
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
    // Contagens tiradas da visao consolidada, nao dos agentes.
    vector<int> counts(5, 0);
    for (States state : agentsStatusVector)
    {
        counts[static_cast<int>(state)]++;
    }
    return counts;
}

vector<int> Game::countStatesByRegion() const
{
    // Layout: counts[regiao * 5 + estado]. A regiao de um agente e a do no em
    // que ele esta; um DEAD conta na regiao onde morreu.
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

void Game::run(int numberOfTurns)
{
    isRunning = true;

    ofstream csvFile;
    if (!csvFileName.empty())
    {
        csvFile.open(csvFileName);
        if (!csvFile.is_open())
        {
            cerr << "Erro: nao foi possivel escrever o CSV em '" << csvFileName << "'." << endl;
            exit(EXIT_FAILURE);
        }
        csvFile << "turno,regiao,S,E,I,R,D" << endl;
    }

    cout << "turno,S,E,I,R,D" << endl;
    // Cronometra so o step(): o custo do CSV e do cout nao diz nada sobre o
    // ganho de paralelizar o laco de agentes.
    double stepSeconds = 0.0;
    for (int turn = 0; turn < numberOfTurns; turn++)
    {
        chrono::steady_clock::time_point before = chrono::steady_clock::now();
        step();
        chrono::steady_clock::time_point after = chrono::steady_clock::now();
        stepSeconds += chrono::duration<double>(after - before).count();

        vector<int> counts = countStates();
        checkConservation(counts);
        cout << turns << "," << counts[static_cast<int>(States::SUSCEPTIBLE)]
             << "," << counts[static_cast<int>(States::EXPOSED)]
             << "," << counts[static_cast<int>(States::INFECTED)]
             << "," << counts[static_cast<int>(States::RECOVERED)]
             << "," << counts[static_cast<int>(States::DEAD)] << endl;

        if (csvFile.is_open())
        {
            vector<int> regionCounts = countStatesByRegion();
            // A invariante tambem precisa valer somada sobre as regioes.
            checkConservation(regionCounts);
            for (int region = 0; region < numberOfRegions; region++)
            {
                csvFile << turns << "," << region
                        << "," << regionCounts[region * 5 + static_cast<int>(States::SUSCEPTIBLE)]
                        << "," << regionCounts[region * 5 + static_cast<int>(States::EXPOSED)]
                        << "," << regionCounts[region * 5 + static_cast<int>(States::INFECTED)]
                        << "," << regionCounts[region * 5 + static_cast<int>(States::RECOVERED)]
                        << "," << regionCounts[region * 5 + static_cast<int>(States::DEAD)] << "\n";
            }
        }
    }
    isRunning = false;

    // Metrica unica de desempenho, comparavel entre versoes e entre escalas.
    // Sai em cerr, e nao em cout, porque o stdout e comparado byte a byte
    // contra o golden.
    double elapsed = (stepSeconds > 1e-9) ? stepSeconds : 1e-9;
    double agentTurns = (double) agents.size() * (double) numberOfTurns;
    cerr << "passo: " << fixed << setprecision(3) << elapsed << " s   "
         << setprecision(0) << (agentTurns / elapsed) << " agentes*turnos/s" << endl;
}

void Game::reset()
{
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
