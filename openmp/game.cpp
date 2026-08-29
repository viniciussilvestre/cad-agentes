#include "game.h"
#include "constants.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <omp.h>

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
    agentsInfectionCounter.assign((initialNumberHealthy + initialNumberInfected), 0);
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

void Game::startAllDumbThreads()
{
    // Os ids sao 0..N-1 e a posicao inicial e funcao pura do id, entao o setup
    // tambem e paralelo. Antes era um emplace_back serial de 100 milhoes de
    // objetos - alocacao, crescimento do vetor e copia, tudo em uma thread.
    //
    // Os suscetiveis vem primeiro e os infectados depois, como antes: o RNG e
    // indexado pelo id, entao trocar a ordem daria outro fluxo a cada agente.
    //
    // As checagens de limite da semeadura sao feitas aqui fora porque
    // generateInfectedLocation aborta com exit() em caso de erro, e sair de
    // dentro de uma regiao paralela nao e forma de reportar problema de
    // configuracao.
    if (initialNumberInfected > 0)
    {
        (void) generateInfectedLocation(0);
    }

    int numberOfAgents = initialNumberHealthy + initialNumberInfected;
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < numberOfAgents; id++)
    {
        bool infected = (id >= initialNumberHealthy);
        agentsPositionVector[id] = infected ? generateInfectedLocation(id) : generateLocation(id);
        agentsStatusVector[id] = infected ? States::INFECTED : States::SUSCEPTIBLE;
        // O contador mede turnos no estado atual e zera em toda transicao, entao
        // vale zero tanto para quem nasce SUSCEPTIBLE quanto para quem nasce
        // INFECTED - este ultimo comeca a fase infecciosa do zero, como qualquer
        // outro que acabou de adoecer.
        agentsInfectionCounter[id] = 0;
    }
}

int Game::generateLocation(int id) const
{
    // Turno -1 marca o sorteio de setup, fora do laco de turnos.
    AgentRng rng = agentEngine(id, -1);
    return (int) rng.below((uint32_t) numberOfNodes);
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
        AgentRng rng = agentEngine(id, -2);
        return candidates[rng.below((uint32_t) candidates.size())];
    }
    return generateLocation(id);
}

AgentRng Game::agentEngine(int id, int turn) const
{
    // O RNG e indexado por (id do agente, turno), nunca por um fluxo sequencial
    // compartilhado: os sorteios de um agente num turno dependem so desses tres
    // valores, e nao da ordem em que os agentes rodam. E o que permite
    // paralelizar o laco de agentes sem mudar a saida.
    AgentRng rng;
    rng.seed = hashSeed(baseSeed, id, turn);
    rng.index = 0;
    return rng;
}

uint64_t Game::hashSeed(unsigned int seed, int id, int turn)
{
    uint64_t hash = splitMix64((uint64_t) seed);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) id);
    hash = splitMix64(hash ^ (uint64_t) (uint32_t) turn);
    return hash;
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

    remapNodes();
}

void Game::buildInfectedAtNode()
{
    // Bandeira, e nao contagem. checkSurroundings so pergunta se ha ALGUM
    // infectado no no (player.cpp), nunca quantos - entao guardar a contagem
    // era trabalho jogado fora, e era ele que obrigava esta passada a ser
    // serial: `infectedAtNode[pos]++` de varias threads exige atomico.
    //
    // Com bandeira, duas threads que escrevem o mesmo no escrevem o mesmo 1. A
    // corrida existe e e inofensiva: nao ha valor intermediario possivel, e a
    // saida continua bit-a-bit identica a da versao sequencial. Sem atomico,
    // sem reducao, sem merge de listas por thread.
    //
    // O `if` antes da escrita nao e microtuning, e o que evita a tempestade de
    // coerencia: com 100 milhoes de agentes em 40.000 nos sao ~2.500 agentes
    // por no, logo ~2.500 escritas redundantes na mesma linha de cache, cada
    // uma invalidando a copia de todos os outros nucleos. Lendo antes, a linha
    // fica em estado compartilhado e so a primeira escrita de cada no paga o
    // custo; as demais viram leitura local.
    //
    // touchedNodes morreu junto: manter a lista dos nos sujos exigiria um merge
    // por thread, e zerar o vetor inteiro e uma passada sequencial de
    // numberOfNodes bytes - irrelevante perto de numberOfAgents.
    fill(infectedAtNode.begin(), infectedAtNode.end(), 0);

    // DEAD nunca e INFECTED, entao nao transmite nem conta como contato.
    int numberOfAgents = (int) agentsStatusVector.size();
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < numberOfAgents; id++)
    {
        if (agentsStatusVector[id] == States::INFECTED && infectedAtNode[agentsPositionVector[id]] == 0)
        {
            infectedAtNode[agentsPositionVector[id]] = 1;
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
    // buildInfectedAtNode e O(agentes) e roda em paralelo (ver la o porque de
    // nao precisar de atomico). Serve tambem de barreira: o retrato dos
    // infectados precisa estar pronto antes de qualquer agente comecar a ler,
    // e o fim da regiao paralela de la garante isso.
    buildInfectedAtNode();

    // O laco de agentes e paralelo por construcao, nao por acaso:
    //
    // - o RNG e indexado por (id do agente, turno), nunca por um fluxo
    //   sequencial compartilhado, entao os sorteios de um agente nao dependem
    //   da ordem em que os agentes rodam;
    // - cada agente so escreve nos proprios campos e em
    //   agentsPositionVector[id] / agentsStatusVector[id]. Os ids sao locais ao
    //   jogo e disjuntos, entao nenhuma thread toca o indice de outra;
    // - neighbors e infectedAtNode sao somente leitura durante o laco.
    //   checkSurroundings apenas le o retrato tirado acima.
    //
    // Por isso a saida e bit-a-bit identica a da versao sequencial com qualquer
    // numero de threads. Se deixar de ser, ha race. Ver scripts/check_golden.sh.
    //
    // O laco precisa ser indexado: range-for so vira forma canonica de OpenMP
    // a partir da versao 5.0.
    //
    // schedule(guided), e nao static, por dois motivos que se somam:
    //
    // - o trabalho por agente nao e uniforme: DEAD sai na hora, EXPOSED e
    //   INFECTED sorteiam mais que um SUSCEPTIBLE que so anda;
    // - em CPU com nucleos heterogeneos (os 8 P-cores e 2 E-cores do Apple
    //   Silicon aqui) fatias iguais fazem todo mundo esperar pelo nucleo lento.
    //
    // Medido com 1.000.000 de agentes e 100 turnos, em segundos:
    //
    //                 8 threads   10 threads
    //   static          0,272       0,351
    //   dynamic,4096    0,282       0,256
    //   guided          0,261       0,271
    //
    // static e a pior com 10 threads justamente porque os E-cores seguram a
    // barreira. guided fica perto do melhor nas duas colunas, entao e o padrao.
    int numberOfAgents = (int) agentsStatusVector.size();
    #pragma omp parallel for schedule(guided)
    for (int id = 0; id < numberOfAgents; id++)
    {
        if (agentsStatusVector[id] == States::DEAD)
        {
            continue;
        }
        AgentRng rng = agentEngine(id, turns);
        agentTurn(id, neighbors, numberOfNodes, infectedAtNode,
                  agentsPositionVector, agentsStatusVector, agentsInfectionCounter, rng);
    }
    turns++;
}

void Game::run(int numberOfTurns)
{
    isRunning = true;
    chrono::steady_clock::time_point before = chrono::steady_clock::now();
    double time = 0.0;
    for (int turn = 0; turn < numberOfTurns; turn++)
    {
        step();
    }
    cout << "turno,S,E,I,R,D" << endl;
    vector<int> counts = countStates();
    checkConservation(counts);

    cout << turns << "," << counts[static_cast<int>(States::SUSCEPTIBLE)]
        << "," << counts[static_cast<int>(States::EXPOSED)]
        << "," << counts[static_cast<int>(States::INFECTED)]
        << "," << counts[static_cast<int>(States::RECOVERED)]
        << "," << counts[static_cast<int>(States::DEAD)] << endl;
    isRunning = false;

    chrono::steady_clock::time_point after = chrono::steady_clock::now();
    time = chrono::duration<double>(after - before).count();
    double elapsed = (time > 1e-9) ? time : 1e-9;
    cerr << "tempo: " << fixed << setprecision(3) << elapsed << " s   "<< endl;
}

void Game::reset()
{
    fill(infectedAtNode.begin(), infectedAtNode.end(), 0);
    fill(agentsPositionVector.begin(), agentsPositionVector.end(), 0);
    fill(agentsStatusVector.begin(), agentsStatusVector.end(), States::SUSCEPTIBLE);
    fill(agentsInfectionCounter.begin(), agentsInfectionCounter.end(), 0);
    turns = 0;
    isRunning = false;
}

void Game::quit()
{
    reset();
}
