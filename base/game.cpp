#include "game.h"
#include "constants.h"
#include <algorithm>
#include <iterator>
#include <limits>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

Game::Game()
{
    initialNumberHealthy = 0;
    initialNumberInfected = 0;
    firstAgent = 0;
    turns = 0;
    isRunning = false;
    min_x_coordinate = numeric_limits<int>::max();
    max_x_coordinate = numeric_limits<int>::min();
    min_y_coordinate = numeric_limits<int>::max();
    max_y_coordinate = numeric_limits<int>::min();
    numberOfNodes = 0;
    numberOfRegions = 0;
    baseSeed = Constants::DEFAULT_SEED;
}

void Game::configure(AgentPool& agentPool, int firstAgentId, int numberOfHealthy, int numberOfInfected)
{
    pool = &agentPool;
    firstAgent = firstAgentId;
    initialNumberHealthy = numberOfHealthy;
    initialNumberInfected = numberOfInfected;
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

void Game::startAllDumbThreads()
{
    // Os ids sao globais e a posicao inicial e funcao pura do id, entao o setup
    // e um laco direto sobre o pool. Antes era um emplace_back de 100 milhoes
    // de objetos - alocacao, crescimento do vetor e copia.
    //
    // Cada jogo semeia a propria faixa [firstAgent, firstAgent + n). Como o RNG
    // e indexado pelo id GLOBAL, dois jogos com o mesmo mapa e o mesmo numero de
    // agentes ainda assim evoluem diferente - o que se quer de ambientes
    // independentes rodando lado a lado.
    //
    // Os suscetiveis vem primeiro e os infectados depois, como antes: o RNG e
    // indexado pelo id, entao trocar a ordem daria outro fluxo a cada agente.
    //
    // As checagens de limite da semeadura sao feitas de uma vez aqui, e nao a
    // cada agente: a versao OpenMP precisa disso porque generateInfectedLocation
    // aborta com exit() e sair de dentro de uma regiao paralela nao e forma de
    // reportar problema de configuracao. As duas versoes seguem iguais.
    if (initialNumberInfected > 0)
    {
        (void) generateInfectedLocation(firstAgent);
    }

    int numberOfAgents = initialNumberHealthy + initialNumberInfected;

    // Capacidade fixa, com folga para receber viajantes. Alocada aqui e nunca
    // mais mexida: e o que permite que um jogo escreva na lista de outro
    // enquanto o dono dela a percorre.
    members.assign((size_t) (numberOfAgents + numberOfAgents / Constants::TRAVEL_CAPACITY_DIVISOR), -1);
    memberCount = numberOfAgents;

    for (int local = 0; local < numberOfAgents; local++)
    {
        int id = firstAgent + local;
        bool infected = (local >= initialNumberHealthy);
        pool->positions[id] = infected ? generateInfectedLocation(id) : generateLocation(id);
        pool->states[id] = infected ? States::INFECTED : States::SUSCEPTIBLE;
        // O contador mede turnos no estado atual e zera em toda transicao, entao
        // vale zero tanto para quem nasce SUSCEPTIBLE quanto para quem nasce
        // INFECTED - este ultimo comeca a fase infecciosa do zero, como qualquer
        // outro que acabou de adoecer.
        pool->counters[id] = 0;
        pool->travelLock[id] = 0;
        members[local] = id;
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
                 << "): esperado 'origem destino'." << endl;
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
    // era trabalho jogado fora.
    //
    // Aqui, sozinho, isso valeria pouco. A forma existe para que esta versao e
    // a OpenMP facam exatamente o mesmo trabalho por agente: la a contagem era
    // o que obrigava a passada a ser serial (`infectedAtNode[pos]++` de varias
    // threads exige atomico), e a bandeira e o que a liberou. Comparar as duas
    // so diz alguma coisa se o unico ponto diferente for o numero de threads.
    // Pelo mesmo motivo o `if` antes da escrita fica: la ele evita a tempestade
    // de coerencia, aqui e uma leitura de uma linha que a escrita ja traria.
    //
    // touchedNodes morreu junto: zerar o vetor inteiro e uma passada de
    // numberOfNodes bytes, irrelevante perto de numberOfAgents.
    fill(infectedAtNode.begin(), infectedAtNode.end(), 0);

    // Varre so quem esta NESTE jogo. DEAD nunca e INFECTED, entao nao transmite
    // nem conta como contato.
    int numberOfMembers = memberCount;
    for (int index = 0; index < numberOfMembers; index++)
    {
        int id = members[index];
        if (id < 0) { continue; }
        if (pool->states[id] == States::INFECTED && infectedAtNode[pool->positions[id]] == 0)
        {
            infectedAtNode[pool->positions[id]] = 1;
        }
    }
}

vector<int> Game::countStates() const
{
    // So os presentes: quem viajou ja conta no jogo de destino.
    vector<int> counts(5, 0);
    for (int index = 0; index < memberCount; index++)
    {
        int id = members[index];
        if (id < 0) { continue; }
        counts[static_cast<int>(pool->states[id])]++;
    }
    return counts;
}

vector<int> Game::countStatesByRegion() const
{
    // Layout: counts[regiao * 5 + estado]. A regiao de um agente e a do no em
    // que ele esta; um DEAD conta na regiao onde morreu.
    vector<int> counts((size_t) numberOfRegions * 5, 0);
    for (int index = 0; index < memberCount; index++)
    {
        int id = members[index];
        if (id < 0) { continue; }
        int region = nodeRegion[pool->positions[id]];
        counts[region * 5 + static_cast<int>(pool->states[id])]++;
    }
    return counts;
}

void Game::step(int turn)
{
    // O retrato dos infectados precisa estar pronto antes de qualquer agente
    // comecar a ler: quem se move neste turno nao muda o que os outros
    // enxergam.
    buildInfectedAtNode();

    // Este e o laco que a versao OpenMP paraleliza. Ele e paralelizavel por
    // construcao, nao por acaso:
    //
    // - o RNG e indexado por (id do agente, turno), nunca por um fluxo
    //   sequencial compartilhado, entao os sorteios de um agente nao dependem
    //   da ordem em que os agentes rodam;
    // - cada agente so escreve no slot id do pool, e os ids sao disjuntos;
    // - neighbors e infectedAtNode sao somente leitura durante o laco.
    //
    // Por isso as duas versoes dao a mesma saida com qualquer numero de
    // threads. Se deixarem de dar, ha race na outra.
    //
    // O laco anda sobre members, e nao sobre a faixa de ids: com viagem entre
    // jogos os dois deixam de coincidir, e quem manda e members.
    // numberOfMembers e lido UMA vez, antes do laco. Quem chegar durante o
    // turno entra depois deste limite e so e processado no turno seguinte -
    // que e o comportamento desejado: o agente ja andou uma vez neste turno,
    // no jogo de origem, e processa-lo de novo aqui seria um turno duplo.
    int numberOfMembers = memberCount;
    for (int index = 0; index < numberOfMembers; index++)
    {
        int id = members[index];
        // Slot de quem viajou para ca e ja saiu de novo, ou lixo de uma
        // compactacao ainda nao feita.
        if (id < 0)
        {
            continue;
        }
        if (pool->states[id] == States::DEAD)
        {
            continue;
        }
        AgentRng rng = agentEngine(id, turn);
        agentTurn(id, neighbors, numberOfNodes, infectedAtNode,
                  pool->positions, pool->states, pool->counters, rng);

        // A viagem vem depois do turno normal e usa sorteios a partir do
        // indice 2: agentTurn consome 0 (o slot do passeio) e as vezes 1 (a
        // probabilidade da transicao). Fixar o indice aqui faz os sorteios de
        // viagem nao dependerem de qual caminho o agente tomou la dentro.
        if (world != nullptr && world->size() > 1)
        {
            if (pool->travelLock[id] > 0)
            {
                pool->travelLock[id]--;
            }
            else
            {
                rng.index = 2;
                if (rng.next() < Constants::TRAVEL_THRESHOLD)
                {
                    travel(id, index, rng);
                }
            }
        }
    }
    turns = turn + 1;
}

bool Game::travel(int id, int memberIndex, AgentRng& rng)
{
    // Sorteia entre os OUTROS jogos: tira este do baralho e desloca, em vez de
    // sortear e repetir se cair em si mesmo - laco de rejeicao aqui seria
    // divergencia de warp na versao de GPU, pelo mesmo motivo do passeio.
    int numberOfGames = (int) world->size();
    int destinationIndex = (int) rng.below((uint32_t) (numberOfGames - 1));
    if (destinationIndex >= gameIndex)
    {
        destinationIndex++;
    }
    Game& destination = (*world)[destinationIndex];

    // O destino tem o proprio mapa, entao a posicao antiga nao vale la. O
    // agente chega num no sorteado do grafo de destino.
    int landing = (int) rng.below((uint32_t) destination.numberOfNodes);

    // Reserva o slot no destino ANTES de sair daqui. Se a lista de la estiver
    // cheia, a viagem e recusada e o agente fica onde esta - nunca some.
    int slot = destination.claimMemberSlot();
    if (slot < 0)
    {
        return false;
    }

    destination.members[slot] = id;
    pool->positions[id] = landing;
    pool->travelLock[id] = (unsigned char) Constants::TRAVEL_LOCK_TURNS;

    // So agora o slot de origem e liberado. Nesta ordem o agente existe nos
    // dois lugares por um instante, e nunca em nenhum: uma contagem tirada no
    // meio pode contar a mais, jamais a menos.
    members[memberIndex] = -1;
    return true;
}

int Game::claimMemberSlot()
{
    if (memberCount >= (int) members.size())
    {
        return -1;
    }
    int slot = memberCount;
    memberCount++;
    return slot;
}

void Game::compactMembers()
{
    // Roda entre turnos, com ninguem percorrendo a lista. Empurra os ids vivos
    // para o inicio e encolhe memberCount; a capacidade continua a mesma.
    //
    // O teto e reposto aqui porque na versao OpenMP a reserva de slot
    // incrementa antes de conferir: com a lista cheia, memberCount pode passar
    // da capacidade por um instante. Os slots acima do teto nunca sao escritos,
    // mas percorrer ate la seria leitura fora do vetor.
    if (memberCount > (int) members.size())
    {
        memberCount = (int) members.size();
    }

    int write = 0;
    for (int read = 0; read < memberCount; read++)
    {
        if (members[read] >= 0)
        {
            members[write] = members[read];
            write++;
        }
    }
    for (int index = write; index < memberCount; index++)
    {
        members[index] = -1;
    }
    memberCount = write;
}

void Game::reset()
{
    // So o que e do ambiente. O pool e da Simulation, que o zera de uma vez -
    // zerar aqui apagaria os agentes dos outros jogos.
    fill(infectedAtNode.begin(), infectedAtNode.end(), 0);
    members.clear();
    turns = 0;
    isRunning = false;
}
