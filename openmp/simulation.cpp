#include "simulation.h"
#include "constants.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <omp.h>

Simulation::Simulation(int numberOfGames, int healthy, int infected)
{
    healthyPerGame = healthy;
    infectedPerGame = infected;
    baseSeed = Constants::DEFAULT_SEED;

    if (numberOfGames < 1)
    {
        cerr << "Erro: numero de jogos invalido: " << numberOfGames << "." << endl;
        exit(EXIT_FAILURE);
    }
    games.resize((size_t) numberOfGames);
    chooseNestedParallelism();
}

void Simulation::setNestedParallelism(bool enabled)
{
    // Dois niveis ativos, ou so um. Com um, a regiao paralela de dentro de
    // Game::step vira codigo serial dentro de cada thread do laco de jogos, e o
    // que roda em paralelo sao os ambientes.
    //
    // Qual e melhor nao e questao de gosto, e de quantos jogos existem para
    // dividir entre as threads. Medido com 28 threads:
    //
    //                       so jogos   aninhado
    //    2 jogos, 8,4M ag.    2,738 s    0,603 s
    //    8 jogos, 8,4M ag.    1,245 s    0,750 s
    //   28 jogos, 8,4M ag.    0,272 s    1,085 s
    //   56 jogos, 8,4M ag.    0,494 s    2,059 s
    //
    // Com poucos jogos, o laco de fora deixa a maioria das threads paradas - 2
    // jogos usam 2 threads de 28. Com jogos demais, o aninhamento cria 28x28 =
    // 784 threads disputando 28 nucleos e a sobrecarga come tudo.
    //
    // A virada e exatamente em jogos == threads, que e o que
    // chooseNestedParallelism usa.
    omp_set_max_active_levels(enabled ? 2 : 1);
}

void Simulation::chooseNestedParallelism()
{
    setNestedParallelism((int) games.size() < omp_get_max_threads());
}

void Simulation::setSeed(unsigned int seed)
{
    baseSeed = seed;
    for (Game& game : games)
    {
        game.setSeed(seed);
    }
}

void Simulation::setSeedMode(SeedMode mode, int regionOrNode)
{
    seedMode = mode;
    seedTarget = regionOrNode;
    for (Game& game : games)
    {
        game.setSeedMode(mode, regionOrNode);
    }
}

void Simulation::initialize(const string& fileName)
{
    int agentsPerGame = healthyPerGame + infectedPerGame;
    pool.resize(agentsPerGame * (int) games.size());

    for (size_t index = 0; index < games.size(); index++)
    {
        // Faixas de ids contiguas e disjuntas. Contiguas porque o laco de um
        // jogo varre members em ordem, e no setup members e exatamente a faixa
        // - entao a leitura do pool sai sequencial. Disjuntas porque o id e a
        // identidade do agente no pool: dois jogos com a mesma faixa seriam dois
        // jogos escrevendo no mesmo agente.
        games[index].configure(pool, (int) index * agentsPerGame, healthyPerGame, infectedPerGame);
        games[index].initializeGame(fileName);
    }
}

void Simulation::start()
{
    for (Game& game : games)
    {
        game.startAllDumbThreads();
    }
}

void Simulation::step()
{
    // Dois niveis de paralelismo: os ambientes aqui, os agentes de cada um
    // dentro de Game::step. Sao independentes - um jogo nao le nem escreve o
    // infectedAtNode nem o members de outro, e o pool e escrito por id, que e
    // disjunto entre jogos enquanto nao ha viagem.
    //
    // O aninhamento so vale a pena quando um nivel sozinho nao enche a maquina:
    // com 1 jogo o laco de fora nao tem o que dividir, e com jogos demais o de
    // dentro vira fatia pequena. Por isso ele e opcional (ver
    // Simulation::setNestedParallelism): quem decide e a medicao, nao o padrao.
    //
    // Sem aninhamento ligado, o OpenMP serializa a regiao interna e o efeito e
    // paralelizar so os jogos - que ja e o comportamento certo quando ha jogos
    // suficientes.
    //
    // O caso de um jogo so desvia ANTES do pragma, e nao com uma clausula
    // if(): uma regiao paralela com if falso ainda e uma regiao paralela, e as
    // regioes de dentro de Game::step passam a ser aninhadas. O libgomp so
    // reaproveita o pool de threads quente no nivel de cima, entao aninhadas
    // elas voltam a criar e destruir threads de verdade a cada turno. Medido:
    // o conjunto de testes (500.000 turnos com 1 agente) saiu de segundos para
    // mais de 90 s com 28 threads. Com o desvio em C++, o caminho de um jogo
    // fica exatamente como era antes de existir Simulation.
    int numberOfGames = (int) games.size();
    if (numberOfGames == 1)
    {
        games[0].step(turns);
    }
    else
    {
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < numberOfGames; index++)
        {
            games[index].step(turns);
        }
    }
    turns++;
}

vector<int> Simulation::countStates() const
{
    // Sobre o pool, e nao somando os jogos. Todo agente existe em exatamente um
    // slot do pool, esteja onde estiver - entao esta contagem continua valendo
    // com viagem ligada, quando somar os jogos passaria a depender de nenhum
    // agente estar em transito.
    vector<int> counts(5, 0);
    for (States state : pool.states)
    {
        counts[static_cast<int>(state)]++;
    }
    return counts;
}

void Simulation::checkConservation(const vector<int>& counts) const
{
    int expected = pool.size();
    int total = 0;
    for (int count : counts)
    {
        total += count;
    }
    if (total != expected)
    {
        cerr << "Invariante de conservacao violada no turno " << turns << ": S+E+I+R+D = "
             << total << " != " << expected << endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::run(int numberOfTurns)
{
    chrono::steady_clock::time_point before = chrono::steady_clock::now();
    for (int turn = 0; turn < numberOfTurns; turn++)
    {
        step();
    }
    chrono::steady_clock::time_point after = chrono::steady_clock::now();

    cout << "turno,S,E,I,R,D" << endl;
    vector<int> counts = countStates();
    checkConservation(counts);

    cout << turns << "," << counts[static_cast<int>(States::SUSCEPTIBLE)]
        << "," << counts[static_cast<int>(States::EXPOSED)]
        << "," << counts[static_cast<int>(States::INFECTED)]
        << "," << counts[static_cast<int>(States::RECOVERED)]
        << "," << counts[static_cast<int>(States::DEAD)] << endl;

    double time = chrono::duration<double>(after - before).count();
    double elapsed = (time > 1e-9) ? time : 1e-9;
    cerr << "tempo: " << fixed << setprecision(3) << elapsed << " s   " << endl;
}

void Simulation::reset()
{
    pool.resize(pool.size());
    for (Game& game : games)
    {
        game.reset();
    }
    turns = 0;
}
