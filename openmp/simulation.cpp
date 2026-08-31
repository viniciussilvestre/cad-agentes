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

    // As faixas de id sao calculadas em serie porque cada uma depende da
    // anterior. Ler o arquivo, nao: e a parte cara, e um jogo nao toca no
    // mapa de outro.
    vector<int> firstAgentOf((size_t) numberOfGames, 0);
    for (int index = 0; index < numberOfGames; index++)
    {
        int healthy = totalHealthy / numberOfGames + (index < totalHealthy % numberOfGames ? 1 : 0);
        int infected = totalInfected / numberOfGames + (index < totalInfected % numberOfGames ? 1 : 0);
        firstAgentOf[(size_t) index] = firstAgent;
        firstAgent += healthy + infected;
    }

    // Um mapa por jogo, lidos em paralelo. Com 21 mapas de 1,5 GB, em serie
    // isso e um core sozinho por varios minutos enquanto os outros 20 esperam.
    #pragma omp parallel for schedule(dynamic)
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
        games[index].configure(pool, firstAgentOf[(size_t) index], healthy, infected);
        // A viagem precisa alcancar os outros ambientes. O ponteiro e para o
        // vetor, e nao para um elemento dele: games ja tem o tamanho final
        // desde o construtor, entao nada realoca depois disto.
        games[index].world = &games;
        games[index].gameIndex = index;
        games[index].initializeGame(mapFiles[index]);
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
    // dentro de Game::step.
    //
    // O caso de um jogo so desvia ANTES do pragma, e nao com uma clausula
    // if(): uma regiao paralela com if falso ainda e uma regiao paralela, e as
    // regioes de dentro de Game::step passam a ser aninhadas. O libgomp so
    // reaproveita o pool de threads quente no nivel de cima, entao aninhadas
    // elas voltam a criar e destruir threads de verdade a cada turno. Medido:
    // o conjunto de testes (500.000 turnos com 1 agente) saiu de segundos para
    // mais de 90 s com 28 threads.
    int numberOfGames = (int) games.size();
    if (numberOfGames == 1)
    {
        games[0].step(turns);
    }
    else
    {
        #pragma omp parallel for
        for (int index = 0; index < numberOfGames; index++)
        {
            games[index].step(turns);
        }
    }

    // Compactacao depois que TODOS os jogos andaram: durante o turno as listas
    // estao sendo lidas e recebendo chegadas, e so aqui e seguro mexer nelas.
    // Esta barreira e o que torna a compactacao segura - por isso ela nao pode
    // subir para dentro de Game::step.
    #pragma omp parallel for
    for (int index = 0; index < numberOfGames; index++)
    {
        games[index].compactMembers();
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

    // Uma linha por ambiente. Com viagem ligada, o total de um jogo muda de
    // turno para turno mesmo sem ninguem adoecer - quem conserva e a soma, e
    // por isso a conferencia e feita sobre o pool inteiro logo abaixo.
    //
    // A contagem e paralela e a impressao e serial: sao O(agentes) por jogo,
    // mas a saida precisa sair em ordem de jogo para ser comparavel.
    int numberOfGames = (int) games.size();
    vector<vector<int>> countsPerGame((size_t) numberOfGames);

    #pragma omp parallel for schedule(dynamic)
    for (int index = 0; index < numberOfGames; index++)
    {
        countsPerGame[(size_t) index] = games[index].countStates();
    }

    cout << "jogo,turno,S,E,I,R,D" << endl;
    for (int index = 0; index < numberOfGames; index++)
    {
        const vector<int>& counts = countsPerGame[(size_t) index];
        cout << index << "," << turns
             << "," << counts[static_cast<int>(States::SUSCEPTIBLE)]
             << "," << counts[static_cast<int>(States::EXPOSED)]
             << "," << counts[static_cast<int>(States::INFECTED)]
             << "," << counts[static_cast<int>(States::RECOVERED)]
             << "," << counts[static_cast<int>(States::DEAD)] << "\n";
    }
    cout.flush();

    checkConservation(countStates());

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
