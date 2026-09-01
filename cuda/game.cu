#include "game.h"
#include "constants.h"
#include <cstdlib>

Game::Game()
{
    numberOfNodes = 0;
    numberOfRegions = 0;
}

void Game::setSeedMode(SeedMode mode, int regionOrNode)
{
    seedMode = mode;
    seedTarget = regionOrNode;
}

// A posicao inicial de cada agente e sorteada no device, entao o host so
// precisa reprovar uma semeadura impossivel antes de lancar o kernel: la
// dentro nao ha como imprimir erro nem abortar com mensagem util.
void Game::validateSeeding() const
{
    if (seedMode == SeedMode::SingleNode)
    {
        if (seedTarget < 0 || seedTarget >= numberOfNodes)
        {
            cerr << "Erro: semeadura SingleNode aponta para o no " << seedTarget
                 << ", fora de [0, " << numberOfNodes << ")." << endl;
            exit(EXIT_FAILURE);
        }
        return;
    }
    if (seedMode == SeedMode::SingleRegion)
    {
        if (seedTarget < 0 || seedTarget >= numberOfRegions)
        {
            cerr << "Erro: semeadura SingleRegion aponta para a regiao " << seedTarget
                 << ", fora de [0, " << numberOfRegions << ")." << endl;
            exit(EXIT_FAILURE);
        }
        if (regionNodes[seedTarget].empty())
        {
            cerr << "Erro: a regiao " << seedTarget << " nao tem nenhum no." << endl;
            exit(EXIT_FAILURE);
        }
    }
}

void Game::releaseHostMap()
{
    // swap com temporario vazio, e nao clear(): clear zera o tamanho e mantem a
    // capacidade, que aqui e justamente o que se quer devolver.
    vector<int>().swap(neighbors);
    vector<int>().swap(nodeRegion);
    vector<vector<int>>().swap(regionNodes);
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

// Le um inteiro com sinal a partir de cursor, pulando brancos antes dele.
// Devolve false quando nao ha numero nenhum dali em diante.
//
// Substitui stringstream + istream_iterator, que montava um vector<string> por
// LINHA. Num mapa de 5500x5500 sao 91 milhoes de linhas, e era isso, e nao o
// grafo, que dominava o tempo e a memoria da carga.
static bool readInt(const char*& cursor, int& value)
{
    while (*cursor == ' ' || *cursor == '\t' || *cursor == '\r')
    {
        cursor++;
    }
    bool negative = false;
    if (*cursor == '-') { negative = true; cursor++; }
    else if (*cursor == '+') { cursor++; }
    if (*cursor < '0' || *cursor > '9')
    {
        return false;
    }
    long long parsed = 0;
    while (*cursor >= '0' && *cursor <= '9')
    {
        parsed = parsed * 10 + (*cursor - '0');
        cursor++;
    }
    value = (int) (negative ? -parsed : parsed);
    return true;
}

int Game::declaredNodesOf(const string& fileName)
{
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Erro: nao foi possivel abrir o arquivo do grafo '" << fileName << "'." << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    if (!getline(file, line))
    {
        cerr << "Erro: o arquivo do grafo '" << fileName << "' esta vazio." << endl;
        exit(EXIT_FAILURE);
    }

    int declaredNodes = 0;
    const char* cursor = line.c_str();
    if (!readInt(cursor, declaredNodes) || declaredNodes < 1)
    {
        cerr << "Erro: cabecalho invalido em '" << fileName
             << "': esperado 'numero_de_nos numero_de_arestas numero_de_regioes'." << endl;
        exit(EXIT_FAILURE);
    }
    return declaredNodes;
}

void Game::initializeGame(const string& fileName)
{
    // O arquivo e lido em fluxo, uma linha por vez, e nao carregado inteiro
    // para um vector<string>. Guardar as 91 milhoes de linhas de um mapa de
    // 5500x5500 custava mais memoria que o grafo que sai delas.
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Erro: nao foi possivel abrir o arquivo do grafo '" << fileName << "'." << endl;
        exit(EXIT_FAILURE);
    }

    string line;
    if (!getline(file, line))
    {
        cerr << "Erro: o arquivo do grafo '" << fileName << "' esta vazio." << endl;
        exit(EXIT_FAILURE);
    }

    int declaredNodes = 0;
    int declaredEdges = 0;
    const char* cursor = line.c_str();
    if (!readInt(cursor, declaredNodes) || !readInt(cursor, declaredEdges) ||
        !readInt(cursor, numberOfRegions))
    {
        cerr << "Erro: cabecalho invalido em '" << fileName
             << "': esperado 'numero_de_nos numero_de_arestas numero_de_regioes'." << endl;
        exit(EXIT_FAILURE);
    }
    if (numberOfRegions < 1) {
        cerr << "Erro: numero de regioes invalido em '" << fileName << "': " << numberOfRegions << "." << endl;
        exit(EXIT_FAILURE);
    }

    // As coordenadas sao LOCAIS: servem so para classificar a direcao das
    // arestas enquanto o arquivo e lido, e morrem com a funcao.
    vector<int> coordinateX;
    vector<int> coordinateY;
    vector<int> degree;
    if (declaredNodes > 0)
    {
        coordinateX.reserve((size_t) declaredNodes);
        coordinateY.reserve((size_t) declaredNodes);
        nodeRegion.reserve((size_t) declaredNodes);
    }

    int blankLines = 0;
    size_t lineNumber = 1;
    bool nodesClosed = false;

    while (getline(file, line))
    {
        lineNumber++;
        if (line.empty())
        {
            blankLines++;
            continue;
        }

        cursor = line.c_str();

        if (blankLines == 1)
        {
            int x_coordinate = 0;
            int y_coordinate = 0;
            int region = 0;
            if (!readInt(cursor, x_coordinate) || !readInt(cursor, y_coordinate) ||
                !readInt(cursor, region))
            {
                cerr << "Erro: linha de no invalida em '" << fileName << "' (linha " << lineNumber
                     << "): esperado 'x y regiao'." << endl;
                exit(EXIT_FAILURE);
            }
            if (region < 0 || region >= numberOfRegions) {
                cerr << "Erro: no " << coordinateX.size() << " tem regiao " << region
                     << ", fora de [0, " << numberOfRegions << ")." << endl;
                exit(EXIT_FAILURE);
            }
            coordinateX.push_back(x_coordinate);
            coordinateY.push_back(y_coordinate);
            nodeRegion.push_back(region);
            continue;
        }

        if (blankLines < 2)
        {
            continue;
        }

        // Primeira linha de aresta: o numero de nos ja esta fechado, entao a
        // vizinhanca pode ser dimensionada.
        if (!nodesClosed)
        {
            if (coordinateX.empty()) {
                cerr << "Erro: nenhum no lido de '" << fileName << "'." << endl;
                exit(EXIT_FAILURE);
            }
            numberOfNodes = (int) coordinateX.size();
            neighbors.assign((size_t) numberOfNodes * Constants::SLOTS, -1);
            degree.assign((size_t) numberOfNodes, 0);
            nodesClosed = true;
        }

        int source_node = 0;
        int destination_node = 0;
        if (!readInt(cursor, source_node) || !readInt(cursor, destination_node))
        {
            cerr << "Erro: linha de aresta invalida em '" << fileName << "' (linha " << lineNumber
                 << "): esperado 'origem destino'." << endl;
            exit(EXIT_FAILURE);
        }
        if (source_node < 0 || source_node >= numberOfNodes ||
            destination_node < 0 || destination_node >= numberOfNodes) {
            cerr << "Erro: aresta (" << source_node << ", " << destination_node
                 << ") em '" << fileName << "' referencia no fora de [0, " << numberOfNodes << ")." << endl;
            exit(EXIT_FAILURE);
        }
        if (source_node == destination_node) {
            continue;
        }

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

        int deltaX = coordinateX[destination_node] - coordinateX[source_node];
        int deltaY = coordinateY[destination_node] - coordinateY[source_node];
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

    if (!nodesClosed)
    {
        if (coordinateX.empty()) {
            cerr << "Erro: nenhum no lido de '" << fileName << "'." << endl;
            exit(EXIT_FAILURE);
        }
        numberOfNodes = (int) coordinateX.size();
        neighbors.assign((size_t) numberOfNodes * Constants::SLOTS, -1);
    }

    // ------------------------------------------------------------------
    // AUTO-LACO NOS SLOTS VAZIOS - tres razoes, nenhuma delas padding.
    //
    // 1. Distribuicao estacionaria uniforme. Com todo no tendo exatamente
    //    SLOTS slots e a relacao sendo simetrica, a matriz de transicao e
    //    duplamente estocastica e a densidade converge para uniforme. Sem o
    //    auto-laco a estacionaria seria proporcional ao grau e as bordas
    //    ficariam permanentemente subpovoadas.
    //
    // 2. Quebra da biparticao. Numa grade, um agente obrigado a se mover
    //    alternaria de paridade todo turno, e dois agentes de paridade oposta
    //    nunca ocupariam o mesmo no - metade dos pares seria incapaz de
    //    transmitir a doenca, para sempre.
    //
    // 3. Na GPU, grau fixo tambem significa nenhum branch no passeio: as 32
    //    threads de um warp fazem a mesma sequencia de instrucoes, sorteiam
    //    slots diferentes e nada diverge. Um laco de rejeicao sobre grau
    //    variavel serializaria o warp pelo pior caso.
    //
    // O PREENCHIMENTO ACONTECE NO DEVICE (ver fillSelfLoopsKernel). O vetor
    // sobe com os -1 e e corrigido la, porque a passada e uma leitura e uma
    // escrita sobre os SLOTS * numberOfNodes inteiros do mapa - centenas de MB
    // que a CPU nao tem motivo para tocar de novo: depois da subida ninguem
    // mais le este vetor no host.
    // ------------------------------------------------------------------

    // O cabecalho e usado por declaredNodesOf para dimensionar o vetor
    // concatenado ANTES do parse. Se ele mentir, o deslocamento de todos os
    // jogos seguintes sai errado em silencio - dai a checagem ser fatal.
    if (numberOfNodes != declaredNodes)
    {
        cerr << "Erro: '" << fileName << "' declara " << declaredNodes
             << " nos no cabecalho mas lista " << numberOfNodes << "." << endl;
        exit(EXIT_FAILURE);
    }

    regionNodes.assign(numberOfRegions, vector<int>());
    for (int node = 0; node < numberOfNodes; node++)
    {
        regionNodes[nodeRegion[node]].emplace_back(node);
    }
}
