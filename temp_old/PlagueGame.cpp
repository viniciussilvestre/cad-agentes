#include "PlagueGame.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <cstdlib>
#include <cstdint>

int PlagueGame::totalNumberOfThreads = 0;
std::vector<int> PlagueGame::agentsPositionVector;
std::vector<int> PlagueGame::agentsStatusVector;
int PlagueGame::turns = 0;

PlagueGame::PlagueGame(int initialNumberHealthy, int initialNumberInfected)
{
    lowerResolutionX = 62;
    lowerResolutionY = 60;
    widthPercentage = 0.976;
    heightPercentage = 0.902;
    turns = 0;
    isRunning = false;
    min_x_coordinate = numeric_limits<int>::max();
    max_x_coordinate = numeric_limits<int>::min();
    min_y_coordinate = numeric_limits<int>::max();
    max_y_coordinate = numeric_limits<int>::min();
    this->initialNumberHealthy = initialNumberHealthy;
    this->initialNumberInfected = initialNumberInfected;
    totalNumberOfThreads = initialNumberHealthy + initialNumberInfected;
    numberOfNodes = 0;
    numberOfRegions = 0;
    agentsPositionVector.assign(totalNumberOfThreads, 0);
    agentsStatusVector.assign(totalNumberOfThreads, SUSCEPTIBLE);
    baseSeed = DEFAULT_SEED;
}

void PlagueGame::setSeed(unsigned int seed)
{
    baseSeed = seed;
}

void PlagueGame::setSeedMode(SeedMode mode, int regionOrNode)
{
    seedMode = mode;
    seedTarget = regionOrNode;
}

void PlagueGame::setCsvOutput(const string& fileName)
{
    csvFileName = fileName;
}

// Mistura splitmix64: avalanche boa o suficiente para que sementes vizinhas
// (id e turno consecutivos) produzam fluxos sem correlacao.
static uint64_t splitMix64(uint64_t value)
{
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

unsigned int PlagueGame::hashSeed(unsigned int seed, int id, int turn)
{
    uint64_t hash = splitMix64((uint64_t) seed);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) id);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) turn);
    return (unsigned int) (hash ^ (hash >> 32));
}

mt19937 PlagueGame::agentEngine(int id, int turn) const
{
    // O RNG e indexado por (id do agente, turno), nunca por um fluxo sequencial
    // compartilhado: os sorteios de um agente num turno dependem so desses tres
    // valores, e nao da ordem em que os agentes rodam. Assim a saida continua
    // reprodutivel se o laco de agentes for paralelizado depois.
    return mt19937(hashSeed(baseSeed, id, turn));
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

int PlagueGame::claimSlot(int node, int preferredSlot)
{
    if (neighbors[preferredSlot * numberOfNodes + node] == -1)
    {
        return preferredSlot;
    }
    // Slot cardinal ocupado (um atalho apontando na mesma direcao do vizinho
    // de grade): transborda para os slots de atalho e, no limite, para os
    // cardinais livres.
    for (int slot = 4; slot < SLOTS; slot++)
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

void PlagueGame::initializeGame(const string& fileName)
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
    neighbors.assign((size_t) numberOfNodes * SLOTS, -1);
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
        if (degree[source_node] > SLOTS) {
            cerr << "Erro: no " << source_node << " tem grau " << degree[source_node]
                 << ", acima do teto de " << SLOTS << " arestas." << endl;
            exit(EXIT_FAILURE);
        }
        if (degree[destination_node] > SLOTS) {
            cerr << "Erro: no " << destination_node << " tem grau " << degree[destination_node]
                 << ", acima do teto de " << SLOTS << " arestas." << endl;
            exit(EXIT_FAILURE);
        }

        int deltaX = nodes[destination_node][0] - nodes[source_node][0];
        int deltaY = nodes[destination_node][1] - nodes[source_node][1];
        int forwardSlot = claimSlot(source_node, cardinalSlot(deltaX, deltaY));
        int backwardSlot = claimSlot(destination_node, cardinalSlot(-deltaX, -deltaY));
        if (forwardSlot == -1 || backwardSlot == -1) {
            cerr << "Erro: sem slot livre para a aresta (" << source_node << ", "
                 << destination_node << "); teto de " << SLOTS << " arestas por no." << endl;
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
    for (int slot = 0; slot < SLOTS; slot++)
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

void PlagueGame::initiateThreads(int numberOfThreads, int SEIRState)
{
    for (int i = 0; i < numberOfThreads; i++)
    {
        // agentsVector e a unica dona dos agentes: nada mais guarda copias.
        int id = Player::newId;
        int startNode = (SEIRState == INFECTED) ? generateInfectedLocation(id) : generateLocation(id);
        agentsVector.emplace_back(startNode, SEIRState);
        Player& player = agentsVector.back();
        agentsPositionVector[player.id] = player.position;
        agentsStatusVector[player.id] = player.SEIRState;
    }
}

void PlagueGame::startAllDumbThreads()
{
    initiateThreads(initialNumberHealthy, SUSCEPTIBLE);
    initiateThreads(initialNumberInfected, INFECTED);
}

pair<int, int> PlagueGame::mapNodes(int x, int y, int x_min, int x_max, int y_min, int y_max) const
{
    int range_x = x_max - x_min;
    int range_y = y_max - y_min;
    int new_x = lowerResolutionX + (((x - x_min) * ((resolutionX - 62) - lowerResolutionX)) / range_x);
    int new_y = lowerResolutionY + (((y - y_min) * ((resolutionY - 60) - lowerResolutionY)) / range_y);
    return make_pair(new_x, new_y);
}

int PlagueGame::generateLocation(int id) const
{
    // Turno -1 marca o sorteio de setup, fora do laco de turnos.
    mt19937 rng = agentEngine(id, -1);
    int lower_bound = 0;
    int upper_bound = numberOfNodes - 1;
    uniform_int_distribution<int> uniform(lower_bound, upper_bound);
    return uniform(rng);
}

int PlagueGame::generateInfectedLocation(int id) const
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

void PlagueGame::buildInfectedAtNode()
{
    // Zera so os nos escritos no turno anterior. Com numberOfNodes muito maior
    // que o numero de agentes, limpar o vetor inteiro custaria mais que o
    // trabalho util (40.000 nos contra alguns milhares de agentes).
    for (int node : touchedNodes)
    {
        infectedAtNode[node] = 0;
    }
    touchedNodes.clear();

    // Uma unica passada sobre agentsVector. DEAD nunca e INFECTED, entao nao
    // transmite nem conta como contato.
    for (const Player& agent : agentsVector)
    {
        if (agent.SEIRState == INFECTED)
        {
            if (infectedAtNode[agent.position] == 0)
            {
                touchedNodes.emplace_back(agent.position);
            }
            infectedAtNode[agent.position]++;
        }
    }
}

vector<int> PlagueGame::countStates() const
{
    // Contagens tiradas da visao consolidada, nao dos agentes.
    vector<int> counts(5, 0);
    for (int state : agentsStatusVector)
    {
        counts[state]++;
    }
    return counts;
}

vector<int> PlagueGame::countStatesByRegion() const
{
    // Layout: counts[regiao * 5 + estado]. A regiao de um agente e a do no em
    // que ele esta; um DEAD conta na regiao onde morreu.
    vector<int> counts((size_t) numberOfRegions * 5, 0);
    for (int i = 0; i < totalNumberOfThreads; i++)
    {
        int region = nodeRegion[agentsPositionVector[i]];
        counts[region * 5 + agentsStatusVector[i]]++;
    }
    return counts;
}

void PlagueGame::checkConservation(const vector<int>& counts) const
{
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

void PlagueGame::reset()
{
    Player::newId = 0;
    agentsVector.clear();
    fill(infectedAtNode.begin(), infectedAtNode.end(), 0);
    touchedNodes.clear();
    fill(agentsPositionVector.begin(), agentsPositionVector.end(), 0);
    fill(agentsStatusVector.begin(), agentsStatusVector.end(), (int) SUSCEPTIBLE);
    turns = 0;
    isRunning = false;
}

void PlagueGame::quit()
{
    reset();
}

void PlagueGame::remapNodes()
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

void PlagueGame::step()
{
    buildInfectedAtNode();
    for (Player& agent : agentsVector)
    {
        if (agent.SEIRState == DEAD)
        {
            continue;
        }
        mt19937 rng = agentEngine(agent.id, turns);
        agent.commonRoutine(neighbors, numberOfNodes, infectedAtNode, rng);
    }
    turns++;
}

void PlagueGame::run(int numberOfTurns)
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
    for (int turn = 0; turn < numberOfTurns; turn++)
    {
        step();
        vector<int> counts = countStates();
        checkConservation(counts);
        cout << turns << "," << counts[SUSCEPTIBLE] << "," << counts[EXPOSED] << ","
             << counts[INFECTED] << "," << counts[RECOVERED] << "," << counts[DEAD] << endl;

        if (csvFile.is_open())
        {
            vector<int> regionCounts = countStatesByRegion();
            // A invariante tambem precisa valer somada sobre as regioes.
            checkConservation(regionCounts);
            for (int region = 0; region < numberOfRegions; region++)
            {
                csvFile << turns << "," << region
                        << "," << regionCounts[region * 5 + SUSCEPTIBLE]
                        << "," << regionCounts[region * 5 + EXPOSED]
                        << "," << regionCounts[region * 5 + INFECTED]
                        << "," << regionCounts[region * 5 + RECOVERED]
                        << "," << regionCounts[region * 5 + DEAD] << "\n";
            }
        }
    }
    isRunning = false;
}
