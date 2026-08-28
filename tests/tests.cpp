// Testes das duas propriedades que o auto-laco nos slots vazios sustenta.
//
// Os dois usam uma grade pequena sem atalhos, escrita pelo proprio teste para
// que ele conheca a geometria (largura) e nao dependa do genGraph.

// Sem caminho de pasta: quem escolhe a versao testada e o -I do build, entao o
// mesmo arquivo roda contra base/ e contra openmp/.
#include "game.h"
#include "constants.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

using namespace std;

static const int TEST_SPACING = 100;

static void writeGridMap(const string& fileName, int width, int height)
{
    int numberOfNodes = width * height;
    vector<pair<int, int>> edgeList;
    for (int i = 0; i < numberOfNodes; i++)
    {
        if (i % width != width - 1)
        {
            edgeList.emplace_back(i, i + 1);
        }
        if (i + width < numberOfNodes)
        {
            edgeList.emplace_back(i, i + width);
        }
    }
    sort(edgeList.begin(), edgeList.end());

    ofstream output(fileName);
    if (!output.is_open())
    {
        cerr << "Erro: nao foi possivel escrever o mapa de teste '" << fileName << "'." << endl;
        exit(EXIT_FAILURE);
    }
    output << numberOfNodes << "\t" << edgeList.size() << "\t1\n\n";
    for (int i = 0; i < numberOfNodes; i++)
    {
        output << (i % width) * TEST_SPACING << "\t" << (i / width) * TEST_SPACING << "\t0\n";
    }
    output << "\n";
    output << fixed << setprecision(2);
    for (const pair<int, int>& edge : edgeList)
    {
        output << edge.first << "\t" << edge.second << "\t" << (double) TEST_SPACING << "\n";
    }
    output.close();
}

static int gridParity(int position, int width)
{
    return ((position / width) + (position % width)) % 2;
}

// Razao entre o no mais e o menos visitado por um unico agente. Com o auto-laco
// a matriz de transicao e simetrica, logo duplamente estocastica, logo a
// estacionaria e exatamente uniforme e a razao tende a 1. Sem o auto-laco a
// estacionaria seria proporcional ao grau e a razao ficaria perto de 1,67.
static bool testUniformStationary()
{
    // 4x4 com 500.000 turnos, e nao 6x6 com 200.000: o histograma de um unico
    // passeio correlacionado converge devagar. Em 6x6/200.000 o desvio amostral
    // sozinho ja da razao ~1,19, acima da tolerancia, mesmo com a estacionaria
    // exatamente uniforme. Nesta configuracao a pior razao em 10 sementes e
    // 1,067. A grade menor nao enfraquece o teste: ela ainda tem cantos de grau
    // 2 e interior de grau 4, entao sem o auto-laco a estacionaria seria
    // proporcional ao grau e a razao iria a 4/2 = 2,0.
    const int WIDTH = 4;
    const int HEIGHT = 4;
    const int TURNS = 500000;
    const double TOLERANCE = 1.15;
    string mapFile = "test_grid_stationary.txt";
    writeGridMap(mapFile, WIDTH, HEIGHT);

    Game game(1, 0);
    game.setSeed(12345);
    game.initializeGame(mapFile);
    game.startAllDumbThreads();

    // Checagem exata, sem ruido amostral: a matriz de transicao P tem
    // P[i][j] = (numero de slots de i apontando para j) / SLOTS. Se todas as
    // somas de coluna derem 1, P e duplamente estocastica e a estacionaria e
    // uniforme por construcao. E o mesmo que o histograma mede, so que sem
    // depender de quantos turnos foram rodados.
    int numberOfNodes = WIDTH * HEIGHT;
    vector<double> columnSum(numberOfNodes, 0.0);
    for (int node = 0; node < numberOfNodes; node++)
    {
        for (int slot = 0; slot < Constants::SLOTS; slot++)
        {
            columnSum[game.neighbors[slot * numberOfNodes + node]] += 1.0 / Constants::SLOTS;
        }
    }
    double minimumColumn = columnSum[0];
    double maximumColumn = columnSum[0];
    for (double value : columnSum)
    {
        minimumColumn = min(minimumColumn, value);
        maximumColumn = max(maximumColumn, value);
    }
    bool doublyStochastic = (fabs(minimumColumn - 1.0) < 1e-12) && (fabs(maximumColumn - 1.0) < 1e-12);

    vector<long long> visits(numberOfNodes, 0);
    for (int turn = 0; turn < TURNS; turn++)
    {
        game.step();
        visits[game.agents[0].position]++;
    }
    remove(mapFile.c_str());

    long long minimum = visits[0];
    long long maximum = visits[0];
    for (long long value : visits)
    {
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }
    double ratio = (minimum > 0) ? (double) maximum / (double) minimum : 1e9;

    bool withinTolerance = (ratio < TOLERANCE);
    bool passed = doublyStochastic && withinTolerance;
    cout << "testUniformStationary: grade " << WIDTH << "x" << HEIGHT << ", " << TURNS
         << " turnos, 1 agente" << endl;
    cout << "  somas de coluna de P: min=" << fixed << setprecision(12) << minimumColumn
         << " max=" << maximumColumn << "  ->  " << (doublyStochastic ? "ok" : "FALHOU") << endl;
    cout << "  visitas: minimo=" << minimum << " maximo=" << maximum
         << " esperado=" << (TURNS / numberOfNodes) << endl;
    cout << "  razao maximo/minimo = " << fixed << setprecision(4) << ratio
         << " (tolerancia < " << TOLERANCE << ")  ->  " << (withinTolerance ? "ok" : "FALHOU") << endl;
    cout << "  " << (passed ? "PASSOU" : "FALHOU") << endl;
    return passed;
}

// Numa grade bipartida, um agente obrigado a se mover todo turno alterna de
// paridade de forma deterministica, e dois agentes de paridade inicial oposta
// nunca se encontrariam. O auto-laco torna o passeio lazy e quebra isso.
static bool testLaziness()
{
    const int WIDTH = 8;
    const int HEIGHT = 8;
    const int AGENTS = 200;
    const int TURNS = 200;
    string mapFile = "test_grid_laziness.txt";
    writeGridMap(mapFile, WIDTH, HEIGHT);

    Game game(AGENTS, 0);
    game.setSeed(999);
    game.initializeGame(mapFile);
    game.startAllDumbThreads();

    vector<int> initialParity(AGENTS);
    for (int i = 0; i < AGENTS; i++)
    {
        initialParity[i] = gridParity(game.agents[i].position, WIDTH);
    }

    int turnsWithBothParities = 0;
    long long oppositeContacts = 0;
    vector<int> atNodeByInitialParity((size_t) WIDTH * HEIGHT * 2, 0);
    for (int turn = 0; turn < TURNS; turn++)
    {
        game.step();
        fill(atNodeByInitialParity.begin(), atNodeByInitialParity.end(), 0);
        int currentByParity[2] = {0, 0};
        for (int i = 0; i < AGENTS; i++)
        {
            int position = game.agents[i].position;
            currentByParity[gridParity(position, WIDTH)]++;
            atNodeByInitialParity[position * 2 + initialParity[i]]++;
        }
        if (currentByParity[0] > 0 && currentByParity[1] > 0)
        {
            turnsWithBothParities++;
        }
        for (int node = 0; node < WIDTH * HEIGHT; node++)
        {
            long long even = atNodeByInitialParity[node * 2];
            long long odd = atNodeByInitialParity[node * 2 + 1];
            oppositeContacts += even * odd;
        }
    }
    remove(mapFile.c_str());

    bool bothParities = (turnsWithBothParities == TURNS);
    bool crossContacts = (oppositeContacts > 0);
    bool passed = bothParities && crossContacts;
    cout << "testLaziness: grade " << WIDTH << "x" << HEIGHT << ", " << AGENTS << " agentes, "
         << TURNS << " turnos" << endl;
    cout << "  turnos com as duas paridades presentes: " << turnsWithBothParities << "/" << TURNS
         << "  ->  " << (bothParities ? "ok" : "FALHOU") << endl;
    cout << "  contatos entre paridade inicial oposta: " << oppositeContacts
         << "  ->  " << (crossContacts ? "ok" : "FALHOU") << endl;
    cout << "  " << (passed ? "PASSOU" : "FALHOU") << endl;
    return passed;
}

// Dois jogos no mesmo processo precisam ser independentes. Com o contador de id
// global que existia antes, os agentes do segundo jogo nasciam com id >= N e
// escreviam fora dos limites dos proprios vetores consolidados: o segundo jogo
// divergia do primeiro quando nao corrompia o heap.
static bool testTwoGamesAreIndependent()
{
    const int WIDTH = 8;
    const int HEIGHT = 8;
    const int HEALTHY = 150;
    const int INFECTED = 10;
    const int TURNS = 120;
    string mapFile = "test_grid_twogames.txt";
    writeGridMap(mapFile, WIDTH, HEIGHT);

    Game first(HEALTHY, INFECTED);
    first.setSeed(2024);
    first.initializeGame(mapFile);
    first.startAllDumbThreads();

    Game second(HEALTHY, INFECTED);
    second.setSeed(2024);
    second.initializeGame(mapFile);
    second.startAllDumbThreads();

    bool idsInRange = true;
    for (const Player& agent : second.agents)
    {
        if (agent.id < 0 || agent.id >= (int) second.agentsStatusVector.size())
        {
            idsInRange = false;
        }
    }

    bool identical = true;
    for (int turn = 0; turn < TURNS; turn++)
    {
        first.step();
        second.step();
        if (first.countStates() != second.countStates())
        {
            identical = false;
            break;
        }
    }
    remove(mapFile.c_str());

    vector<int> counts = first.countStates();
    int total = 0;
    for (int count : counts)
    {
        total += count;
    }
    bool conserved = (total == HEALTHY + INFECTED);

    bool passed = idsInRange && identical && conserved;
    cout << "testTwoGamesAreIndependent: 2 jogos, " << (HEALTHY + INFECTED) << " agentes cada, "
         << TURNS << " turnos" << endl;
    cout << "  ids do segundo jogo dentro de [0, " << second.agentsStatusVector.size() << ")"
         << "  ->  " << (idsInRange ? "ok" : "FALHOU") << endl;
    cout << "  mesma semente da as mesmas contagens em todo turno  ->  "
         << (identical ? "ok" : "FALHOU") << endl;
    cout << "  conservacao S+E+I+R+D = " << total << " (esperado " << (HEALTHY + INFECTED) << ")"
         << "  ->  " << (conserved ? "ok" : "FALHOU") << endl;
    cout << "  " << (passed ? "PASSOU" : "FALHOU") << endl;
    return passed;
}

int main()
{
    bool allPassed = true;
    allPassed = testUniformStationary() && allPassed;
    cout << endl;
    allPassed = testLaziness() && allPassed;
    cout << endl;
    allPassed = testTwoGamesAreIndependent() && allPassed;
    cout << endl << (allPassed ? "TODOS OS TESTES PASSARAM" : "HOUVE FALHA") << endl;
    return allPassed ? EXIT_SUCCESS : EXIT_FAILURE;
}
