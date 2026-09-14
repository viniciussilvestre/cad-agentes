#include "simulation.h"
#include "constants.h"
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

Simulation::Simulation(int numberOfGames, int healthy, int infected)
{
    totalHealthy = healthy;
    totalInfected = infected;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    if (numberOfGames < 1)
    {
        if (rank == 0)
        {
            cerr << "Erro: numero de jogos invalido: " << numberOfGames << "." << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    // Mais ranks que jogos nao e erro, mas os ranks sobrando ficariam sem nada
    // para fazer e ainda assim entrariam em todo coletivo. Melhor dizer.
    if (ranks > numberOfGames && rank == 0)
    {
        cerr << "Aviso: " << ranks << " ranks para " << numberOfGames
             << " jogos; " << (ranks - numberOfGames) << " ficam ociosos." << endl;
    }

    games.resize((size_t) numberOfGames);
    mapFiles.assign((size_t) numberOfGames, "A1100a.txt");
    setSeed(Constants::DEFAULT_SEED);

    // Rodizio, e nao blocos: com jogos que nao dividem igualmente entre os
    // ranks, o rodizio deixa a diferenca em um jogo, e o bloco pode deixar um
    // rank com o dobro.
    gameOwner.assign((size_t) numberOfGames, 0);
    for (int index = 0; index < numberOfGames; index++)
    {
        gameOwner[(size_t) index] = index % ranks;
        if (gameOwner[(size_t) index] == rank)
        {
            localGames.push_back(index);
        }
    }
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

    // As faixas de id sao calculadas para TODOS os jogos, em todo rank: elas
    // dependem so da configuracao, e o id precisa significar a mesma coisa em
    // todo lugar para o viajante achar o proprio slot do outro lado.
    for (int index = 0; index < numberOfGames; index++)
    {
        int healthy = totalHealthy / numberOfGames + (index < totalHealthy % numberOfGames ? 1 : 0);
        int infected = totalInfected / numberOfGames + (index < totalInfected % numberOfGames ? 1 : 0);

        games[index].configure(pool, firstAgent, healthy, infected);
        games[index].world = &games;
        games[index].gameIndex = index;
        games[index].myRank = rank;
        games[index].outbox = &outbox;
        firstAgent += healthy + infected;
    }

    // O mapa, esse sim, so dos jogos deste rank. E onde a decomposicao paga.
    for (int index : localGames)
    {
        games[(size_t) index].initializeGame(mapFiles[(size_t) index]);
    }

    // Cada rank conhece o tamanho so dos grafos que leu; a viagem precisa de
    // todos, porque o no de chegada e sorteado na origem. Um int por jogo.
    nodeCountOf.assign((size_t) numberOfGames, 0);
    for (int index : localGames)
    {
        nodeCountOf[(size_t) index] = games[(size_t) index].numberOfNodes;
    }
    MPI_Allreduce(MPI_IN_PLACE, nodeCountOf.data(), numberOfGames, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    // Ponteiros so agora: gameOwner e nodeCountOf ja nao mudam mais de lugar.
    for (Game& game : games)
    {
        game.gameOwner = gameOwner.data();
        game.nodeCountOf = nodeCountOf.data();
    }
}

void Simulation::start()
{
    for (int index : localGames)
    {
        games[(size_t) index].startAllDumbThreads();
    }
}

// Troca os viajantes e aplica as chegadas.
//
// Roda na barreira entre turnos, quando ninguem esta percorrendo lista nenhuma.
// E o unico ponto de comunicacao do laco.
void Simulation::exchangeTravellers()
{
    int numberOfGames = (int) games.size();

    // Separa por rank de destino. O bucket do proprio rank nao vai para a rede:
    // e aplicado junto com o que chegar, para que local e remoto sigam o mesmo
    // caminho.
    vector<vector<Departure>> byRank((size_t) ranks);
    for (const Departure& departure : outbox)
    {
        byRank[(size_t) gameOwner[(size_t) departure.destinationGame]].push_back(departure);
    }
    outbox.clear();

    vector<int> sendCounts((size_t) ranks, 0);
    vector<int> recvCounts((size_t) ranks, 0);
    for (int target = 0; target < ranks; target++)
    {
        // Contagens em BYTES, porque o payload vai como MPI_BYTE: assim nao ha
        // tipo derivado a registrar e o layout do struct e o unico contrato.
        sendCounts[(size_t) target] = (int) (byRank[(size_t) target].size() * sizeof(Departure));
    }
    // O bucket local nao viaja.
    sendCounts[(size_t) rank] = 0;

    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    vector<int> sendDispls((size_t) ranks, 0);
    vector<int> recvDispls((size_t) ranks, 0);
    int sendTotal = 0;
    int recvTotal = 0;
    for (int target = 0; target < ranks; target++)
    {
        sendDispls[(size_t) target] = sendTotal;
        recvDispls[(size_t) target] = recvTotal;
        sendTotal += sendCounts[(size_t) target];
        recvTotal += recvCounts[(size_t) target];
    }

    vector<char> sendBuffer((size_t) sendTotal);
    for (int target = 0; target < ranks; target++)
    {
        if (sendCounts[(size_t) target] > 0)
        {
            copy((const char*) byRank[(size_t) target].data(),
                 (const char*) byRank[(size_t) target].data() + sendCounts[(size_t) target],
                 sendBuffer.begin() + sendDispls[(size_t) target]);
        }
    }

    vector<char> recvBuffer((size_t) recvTotal);
    MPI_Alltoallv(sendBuffer.data(), sendCounts.data(), sendDispls.data(), MPI_BYTE,
                  recvBuffer.data(), recvCounts.data(), recvDispls.data(), MPI_BYTE,
                  MPI_COMM_WORLD);

    // Junta o que ficou em casa com o que chegou.
    vector<Departure> arrivals = byRank[(size_t) rank];
    const Departure* received = (const Departure*) recvBuffer.data();
    size_t receivedCount = (size_t) recvTotal / sizeof(Departure);
    arrivals.insert(arrivals.end(), received, received + receivedCount);

    // Ordem canonica antes de aplicar. A posicao de um agente dentro de members
    // nao muda resultado nenhum, mas a ordem de chegada pela rede nao e
    // reproduzivel - e sem esta ordenacao duas execucoes iguais dariam listas
    // em ordens diferentes, o que atrapalha qualquer comparacao byte a byte de
    // despejo de estado.
    sort(arrivals.begin(), arrivals.end(),
         [](const Departure& left, const Departure& right) { return left.id < right.id; });

    for (const Departure& arrival : arrivals)
    {
        if (arrival.destinationGame < 0 || arrival.destinationGame >= numberOfGames ||
            gameOwner[(size_t) arrival.destinationGame] != rank)
        {
            cerr << "Erro: rank " << rank << " recebeu o agente " << arrival.id
                 << " para o jogo " << arrival.destinationGame << ", que nao e seu." << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        games[(size_t) arrival.destinationGame].acceptArrival(arrival);
    }
}

void Simulation::step()
{
    for (int index : localGames)
    {
        games[(size_t) index].step(turns);
    }

    exchangeTravellers();

    // Compactacao depois das chegadas: e quando as listas param de mudar.
    for (int index : localGames)
    {
        games[(size_t) index].compactMembers();
    }
    turns++;
}

vector<int> Simulation::countStates() const
{
    vector<int> counts(5, 0);
    for (int index : localGames)
    {
        vector<int> local = games[(size_t) index].countStates();
        for (int state = 0; state < 5; state++)
        {
            counts[(size_t) state] += local[(size_t) state];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, counts.data(), 5, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
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
        if (rank == 0)
        {
            cerr << "Invariante de conservacao violada no turno " << turns << ": S+E+I+R+D = "
                 << total << " != " << expected << endl;
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

void Simulation::run(int numberOfTurns)
{
    // Todos comecam juntos, senao o tempo medido inclui a diferenca de quem
    // terminou a carga de mapa primeiro.
    MPI_Barrier(MPI_COMM_WORLD);
    chrono::steady_clock::time_point before = chrono::steady_clock::now();
    for (int turn = 0; turn < numberOfTurns; turn++)
    {
        step();
    }
    chrono::steady_clock::time_point after = chrono::steady_clock::now();

    // Contagens por jogo, de todos os jogos. Cada rank preenche so as faixas
    // dos seus e a soma completa o quadro - com jogos disjuntos entre ranks, a
    // reducao por soma equivale a um Gatherv, e cabe em numberOfGames * 5 ints.
    int numberOfGames = (int) games.size();
    vector<int> perGame((size_t) numberOfGames * 5, 0);
    for (int index : localGames)
    {
        vector<int> counts = games[(size_t) index].countStates();
        for (int state = 0; state < 5; state++)
        {
            perGame[(size_t) index * 5 + (size_t) state] = counts[(size_t) state];
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, perGame.data(), numberOfGames * 5, MPI_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    checkConservation(countStates());

    // O tempo do rank mais lento e o tempo da simulacao: os outros ficaram
    // esperando por ele no coletivo.
    double time = chrono::duration<double>(after - before).count();
    double slowest = time;
    MPI_Reduce(&time, &slowest, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank != 0)
    {
        return;
    }

    cout << "jogo,turno,S,E,I,R,D" << endl;
    for (int index = 0; index < numberOfGames; index++)
    {
        cout << index << "," << turns
             << "," << perGame[(size_t) index * 5 + (size_t) States::SUSCEPTIBLE]
             << "," << perGame[(size_t) index * 5 + (size_t) States::EXPOSED]
             << "," << perGame[(size_t) index * 5 + (size_t) States::INFECTED]
             << "," << perGame[(size_t) index * 5 + (size_t) States::RECOVERED]
             << "," << perGame[(size_t) index * 5 + (size_t) States::DEAD] << "\n";
    }
    cout.flush();

    double elapsed = (slowest > 1e-9) ? slowest : 1e-9;
    cerr << "tempo: " << fixed << setprecision(3) << elapsed << " s   " << endl;
}
