#include "simulation.h"
#include "constants.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

Simulation::Simulation(int numberOfGames, int healthy, int infected)
{
    totalHealthy = healthy;
    totalInfected = infected;
    numberOfAgents = healthy + infected;

    if (numberOfGames < 1)
    {
        cerr << "Erro: numero de jogos invalido: " << numberOfGames << "." << endl;
        exit(EXIT_FAILURE);
    }
    // gameOf tem um byte por agente, o que economiza centenas de MB na escala
    // de 600 milhoes - e limita o numero de jogos a 256.
    if (numberOfGames > 256)
    {
        cerr << "Erro: no maximo 256 jogos (gameOf tem um byte por agente)." << endl;
        exit(EXIT_FAILURE);
    }

    games.resize((size_t) numberOfGames);
    mapFiles.assign((size_t) numberOfGames, "A1100a.txt");
    seeds.assign((size_t) numberOfGames, Constants::DEFAULT_SEED);
}

// initialize ja escreve mapas e configuracao nos buffers e libera o grafo do
// host logo depois, entao mexer em qualquer um dos tres dali em diante nao teria
// efeito nenhum. Falhar alto e melhor que rodar com uma configuracao que o
// usuario acha que aplicou.
void Simulation::refuseAfterInitialize(const char* what) const
{
    if (initialized)
    {
        cerr << "Erro: " << what << " precisa ser chamado antes de initialize()." << endl;
        exit(EXIT_FAILURE);
    }
}

void Simulation::setSeed(unsigned int seed)
{
    refuseAfterInitialize("setSeed");
    seeds.assign(games.size(), seed);
}

void Simulation::setSeeds(const vector<unsigned int>& newSeeds)
{
    refuseAfterInitialize("setSeeds");
    if (newSeeds.size() < games.size())
    {
        cerr << "Erro: " << newSeeds.size() << " sementes para " << games.size()
             << " jogos." << endl;
        exit(EXIT_FAILURE);
    }
    for (size_t index = 0; index < games.size(); index++)
    {
        seeds[index] = newSeeds[index];
    }
}

void Simulation::setSeedMode(SeedMode mode, int regionOrNode)
{
    refuseAfterInitialize("setSeedMode");
    for (Game& game : games)
    {
        game.setSeedMode(mode, regionOrNode);
    }
}

void Simulation::setMaps(const vector<string>& files)
{
    refuseAfterInitialize("setMaps");
    if (files.size() < games.size())
    {
        cerr << "Erro: " << files.size() << " mapas para " << games.size()
             << " jogos." << endl;
        exit(EXIT_FAILURE);
    }
    mapFiles = files;
}

void Simulation::setMetallib(const string& path)
{
    refuseAfterInitialize("setMetallib");
    metallibPath = path;
}

void Simulation::initialize(const string& fileName)
{
    setMaps(vector<string>(games.size(), fileName));
    initialize();
}

// Le so os cabecalhos, para saber onde cada mapa comeca no buffer concatenado
// antes de parsear qualquer um deles. Um getline por arquivo.
void Simulation::prescanMaps()
{
    int numberOfGames = (int) games.size();
    nodeOffset.assign((size_t) numberOfGames, 0);
    nodeCount.assign((size_t) numberOfGames, 0);
    totalNodes = 0;

    for (int index = 0; index < numberOfGames; index++)
    {
        int nodes = Game::declaredNodesOf(mapFiles[(size_t) index]);
        nodeOffset[(size_t) index] = (int) totalNodes;
        nodeCount[(size_t) index] = nodes;
        totalNodes += nodes;
    }

    // nodeOffset e int porque vai para o kernel como int. Estourar isso daria
    // deslocamento negativo em silencio.
    if (totalNodes > 2147483647LL)
    {
        cerr << "Erro: " << totalNodes << " nos no total, acima do teto de "
             << "2^31-1 imposto por nodeOffset." << endl;
        exit(EXIT_FAILURE);
    }
}

// Carga em duas fases, e a ordem entre elas e o ponto todo desta versao.
//
// A CUDA faz parse -> copia para buffer fixado -> DMA para o device -> kernel
// de correcao, com um pipeline de streams para esconder o PCIe atras do parse.
// Sao tres passadas completas sobre o maior array do programa, e existem porque
// o dado precisa mudar de memoria.
//
// Aqui o dado nao muda de memoria. Entao:
//
//   1. a GPU semeia o buffer INTEIRO com auto-lacos, de uma vez;
//   2. o parser da CPU sobrescreve, direto no mesmo buffer, so os slots que
//      viraram aresta.
//
// Uma escrita sequencial na GPU mais escritas espalhadas da CPU. Sem vetor
// intermediario, sem memcpy, sem passada de correcao. E o que a memoria
// unificada compra neste programa - o laco de turnos nao muda nada, porque la
// os dois modelos ja mantem tudo residente na GPU.
//
// O laco de parse e paralelo porque a leitura de arquivo domina: com dezenas de
// mapas grandes, em serie e um core sozinho enquanto os outros esperam.
void Simulation::loadMaps()
{
    int numberOfGames = (int) games.size();
    candidatesOf.assign((size_t) numberOfGames, vector<int>());

    // Fase 1. Precisa terminar antes de qualquer parse: a CPU vai sobrescrever
    // exatamente estas posicoes, e sem a barreira o kernel poderia apagar uma
    // aresta ja escrita.
    for (int index = 0; index < numberOfGames; index++)
    {
        gpu.prefillNeighbors(nodeOffset[(size_t) index], nodeCount[(size_t) index]);
    }
    gpu.finish();

    // Fase 2. Buffers StorageModeShared no Apple Silicon sao coerentes, entao
    // estas escritas ficam visiveis para a GPU sem sincronizacao explicita -
    // nao ha didModifyRange a chamar, que e coisa do modo Managed.
    #pragma omp parallel for schedule(dynamic)
    for (int index = 0; index < numberOfGames; index++)
    {
        Game& game = games[(size_t) index];
        game.initializeGame(mapFiles[(size_t) index],
                            gpu.neighborsAt(nodeOffset[(size_t) index]),
                            nodeCount[(size_t) index]);
        game.validateSeeding();

        // Candidatos da semeadura por regiao, copiados AQUI porque regionNodes
        // morre no fim desta iteracao.
        if (game.seedModeOf() == SeedMode::SingleRegion)
        {
            candidatesOf[(size_t) index] = game.regionNodes[(size_t) game.seedTargetOf()];
        }

        game.releaseHostMap();
    }
}

void Simulation::uploadMetadata()
{
    int numberOfGames = (int) games.size();

    // As faixas de id sao calculadas em serie porque cada uma depende da
    // anterior.
    firstAgentOf.assign((size_t) numberOfGames, 0);
    vector<int> healthyOf((size_t) numberOfGames, 0);
    int firstAgent = 0;
    for (int index = 0; index < numberOfGames; index++)
    {
        // Divisao igual, com a sobra distribuida um agente por jogo entre os
        // primeiros, para a soma bater exatamente com o que foi pedido.
        int healthy = totalHealthy / numberOfGames + (index < totalHealthy % numberOfGames ? 1 : 0);
        int infected = totalInfected / numberOfGames + (index < totalInfected % numberOfGames ? 1 : 0);
        firstAgentOf[(size_t) index] = firstAgent;
        healthyOf[(size_t) index] = healthy;
        firstAgent += healthy + infected;
    }

    // Candidatos concatenados. Ficam vazios para os jogos que nao usam
    // SingleRegion.
    vector<int> candidates;
    vector<int> candidateOffset((size_t) numberOfGames, 0);
    vector<int> candidateCount((size_t) numberOfGames, 0);
    vector<int> seedModeOf((size_t) numberOfGames, 0);
    vector<int> seedTargetOf((size_t) numberOfGames, 0);

    for (int index = 0; index < numberOfGames; index++)
    {
        seedModeOf[(size_t) index] = (int) games[(size_t) index].seedModeOf();
        seedTargetOf[(size_t) index] = games[(size_t) index].seedTargetOf();
        candidateOffset[(size_t) index] = (int) candidates.size();
        const vector<int>& list = candidatesOf[(size_t) index];
        candidates.insert(candidates.end(), list.begin(), list.end());
        candidateCount[(size_t) index] = (int) list.size();
    }
    // Um MTLBuffer de tamanho zero e invalido, e o kernel nunca le daqui quando
    // nao ha semeadura por regiao.
    if (candidates.empty())
    {
        candidates.push_back(0);
    }
    vector<vector<int>>().swap(candidatesOf);

    gpu.uploadMetadata(nodeOffset, nodeCount, firstAgentOf, healthyOf, seeds,
                       seedModeOf, seedTargetOf,
                       candidates, candidateOffset, candidateCount);
}

void Simulation::initialize()
{
    prescanMaps();

    if (!gpu.initialize(metallibPath, totalNodes, numberOfAgents, (int) games.size()))
    {
        exit(EXIT_FAILURE);
    }

    loadMaps();
    uploadMetadata();
    initialized = true;
}

void Simulation::start()
{
    if (!initialized)
    {
        cerr << "Erro: start() antes de initialize()." << endl;
        exit(EXIT_FAILURE);
    }
    gpu.seedAgents();
    gpu.finish();
}

void Simulation::step()
{
    gpu.runTurn(turns);
    turns++;
}

vector<vector<int>> Simulation::countStatesPerGame() const
{
    int numberOfGames = (int) games.size();
    vector<vector<int>> counts((size_t) numberOfGames, vector<int>(5, 0));

    // Uma passada na CPU sobre os buffers que a GPU escreveu. Com memoria
    // unificada isso e leitura de RAM comum, sem transferencia - por isso nao ha
    // kernel de reducao aqui, ao contrario da versao CUDA, onde o mesmo trabalho
    // atravessaria o PCIe.
    const unsigned char* states = gpu.states();
    const unsigned char* gameOf = gpu.gameOf();
    for (int id = 0; id < numberOfAgents; id++)
    {
        counts[gameOf[id]][states[id]]++;
    }
    return counts;
}

void Simulation::checkConservation(const vector<vector<int>>& counts) const
{
    long long total = 0;
    for (const vector<int>& perGame : counts)
    {
        for (int value : perGame)
        {
            total += value;
        }
    }
    if (total != (long long) numberOfAgents)
    {
        cerr << "Invariante de conservacao violada no turno " << turns << ": S+E+I+R+D = "
             << total << " != " << numberOfAgents << endl;
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
    // Os turnos sao enfileirados sem espera: sem o finish aqui, o cronometro
    // mediria o tempo de montar os command buffers, e nao o de executa-los.
    gpu.finish();
    chrono::steady_clock::time_point after = chrono::steady_clock::now();

    vector<vector<int>> counts = countStatesPerGame();
    checkConservation(counts);

    cout << "jogo,turno,S,E,I,R,D" << endl;
    for (size_t game = 0; game < counts.size(); game++)
    {
        cout << game << "," << turns
             << "," << counts[game][static_cast<int>(States::SUSCEPTIBLE)]
             << "," << counts[game][static_cast<int>(States::EXPOSED)]
             << "," << counts[game][static_cast<int>(States::INFECTED)]
             << "," << counts[game][static_cast<int>(States::RECOVERED)]
             << "," << counts[game][static_cast<int>(States::DEAD)] << "\n";
    }
    cout.flush();

    double time = chrono::duration<double>(after - before).count();
    double elapsed = (time > 1e-9) ? time : 1e-9;
    cerr << "tempo: " << fixed << setprecision(3) << elapsed << " s   " << endl;
}
