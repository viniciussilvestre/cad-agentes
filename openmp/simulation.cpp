#include "simulation.h"
#include "constants.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <omp.h>

Simulation::Simulation(int numberOfGames, int healthy, int infected)
{
    totalHealthy = healthy;
    totalInfected = infected;

    if (numberOfGames < 1)
    {
        cerr << "Erro: numero de jogos invalido: " << numberOfGames << "." << endl;
        exit(EXIT_FAILURE);
    }
    games.resize((size_t) numberOfGames);
    mapFiles.assign((size_t) numberOfGames, "A1100a.txt");
    setSeed(Constants::DEFAULT_SEED);
}

void Simulation::setNestedParallelism(bool enabled)
{
    // Dois niveis ativos, ou so um. Com um, a regiao paralela de dentro de
    // Game::step vira codigo serial dentro de cada thread do laco de jogos, e o
    // que roda em paralelo sao os ambientes.
    //
    // Qual e melhor depende de quantos jogos ha para dividir entre as threads,
    // e a diferenca e grande nos dois sentidos - com poucos jogos o laco de
    // fora deixa a maioria das threads parada, com jogos demais o aninhamento
    // cria threads^2 disputando os mesmos nucleos. As medicoes estao em
    // docs/medicoes.md.
    omp_set_max_active_levels(enabled ? 2 : 1);
}

void Simulation::setSeed(unsigned int seed)
{
    for (Game& game : games)
    {
        game.setSeed(seed);
    }
}

void Simulation::setSeeds(const vector<unsigned int>& seeds)
{
    for (size_t index = 0; index < games.size(); index++)
    {
        games[index].setSeed(seeds[index]);
    }
}

void Simulation::setMaps(const vector<string>& files)
{
    mapFiles = files;
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
    setMaps(vector<string>(games.size(), fileName));
    initialize();
}

void Simulation::initialize()
{
    pool.resize(totalHealthy + totalInfected);

    int numberOfGames = (int) games.size();
    int firstAgent = 0;

    for (int index = 0; index < numberOfGames; index++)
    {
        // Divisao igual, com a sobra distribuida um agente por jogo entre os
        // primeiros. Sem isso, --agents 10 --games 4 perderia dois agentes na
        // divisao inteira e a conservacao acusaria.
        int healthy = totalHealthy / numberOfGames + (index < totalHealthy % numberOfGames ? 1 : 0);
        int infected = totalInfected / numberOfGames + (index < totalInfected % numberOfGames ? 1 : 0);

        // Faixas de ids contiguas e disjuntas. Contiguas porque o laco de um
        // jogo varre members em ordem, e no setup members e exatamente a faixa
        // - entao a leitura do pool sai sequencial. Disjuntas porque o id e a
        // identidade do agente no pool: dois jogos com a mesma faixa seriam dois
        // jogos escrevendo no mesmo agente.
        games[index].configure(pool, firstAgent, healthy, infected);
        games[index].initializeGame(mapFiles[index]);
        firstAgent += healthy + infected;
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
