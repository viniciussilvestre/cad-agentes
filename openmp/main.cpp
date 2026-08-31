#include "simulation.h"
#include "constants.h"
#include <cstdlib>
#include <omp.h>

// ---------------------------------------------------------------------------
// Configuracao por jogo, escrita aqui na mao.
//
// Sao coisas que nao cabem bem numa flag - um mapa e uma semente por ambiente,
// em ordem. Deixe vazio para que todos os jogos usem --map e --seed.
//
// Sem validacao de proposito: quem edita estas listas e quem compila.
// ---------------------------------------------------------------------------
static const vector<string> gameMaps = {
    "map_0.txt", "map_1.txt", "map_2.txt", "map_3.txt", "map_4.txt", "map_5.txt", "map_6.txt", 
    "map_7.txt", "map_8.txt", "map_9.txt", "map_10.txt", "map_11.txt", "map_12.txt", "map_13.txt", 
    "map_14.txt", "map_15.txt", "map_16.txt", "map_17.txt", "map_18.txt", "map_19.txt", "map_20.txt"
};

static const vector<unsigned int> gameSeeds = {
    4, 87, 12, 93, 41, 65, 18, 30, 99, 52, 7, 76, 23, 88, 34, 
    11, 62, 45, 90, 19, 81
};

int main(int argc, char** argv)
{
    // --agents e --infected sao TOTAIS, divididos igualmente entre os jogos:
    // com --games 4 --agents 1000, sao 250 saudaveis por jogo.
    int initialNumberHealthy = 29250000;
    int initialNumberInfected = 1000000;
    int numberOfTurns = 1000;
    int numberOfGames = 21;
    // Zero significa "deixa o OpenMP decidir", que e o padrao dele: o numero de
    // nucleos logicos, ou o que estiver em OMP_NUM_THREADS.
    int numberOfThreads = 0;
    // Paralelismo nos dois lacos (jogos e agentes) ou so no de jogos. Escolha
    // explicita, como o numero de threads: ver docs/medicoes.md.
    bool nested = false;
    unsigned int seed = Constants::DEFAULT_SEED;
    string mapFile = "map.txt";
    SeedMode seedMode = SeedMode::Uniform;
    int seedTarget = 0;

    for (int i = 1; i < argc; i++)
    {
        string flag = argv[i];
        bool hasValue = (i + 1 < argc);
        if (flag == "--map" && hasValue) { mapFile = argv[++i]; }
        else if (flag == "--agents" && hasValue) { initialNumberHealthy = atoi(argv[++i]); }
        else if (flag == "--infected" && hasValue) { initialNumberInfected = atoi(argv[++i]); }
        else if (flag == "--turns" && hasValue) { numberOfTurns = atoi(argv[++i]); }
        else if (flag == "--seed" && hasValue) { seed = (unsigned int) strtoul(argv[++i], nullptr, 10); }
        else if (flag == "--games" && hasValue) { numberOfGames = atoi(argv[++i]); }
        else if (flag == "--threads" && hasValue) { numberOfThreads = atoi(argv[++i]); }
        else if (flag == "--nested") { nested = true; }
        else if (flag == "--seed-target" && hasValue) { seedTarget = atoi(argv[++i]); }
        else if (flag == "--seed-mode" && hasValue)
        {
            string mode = argv[++i];
            if (mode == "uniform") { seedMode = SeedMode::Uniform; }
            else if (mode == "region") { seedMode = SeedMode::SingleRegion; }
            else if (mode == "node") { seedMode = SeedMode::SingleNode; }
            else
            {
                cerr << "Erro: --seed-mode aceita 'uniform', 'region' ou 'node'." << endl;
                return EXIT_FAILURE;
            }
        }
        else
        {
            cerr << "Erro: flag desconhecida ou sem valor: '" << flag << "'." << endl;
            return EXIT_FAILURE;
        }
    }

    if (numberOfThreads > 0)
    {
        omp_set_num_threads(numberOfThreads);
    }

    Simulation simulation(numberOfGames, initialNumberHealthy, initialNumberInfected);
    simulation.setNestedParallelism(nested);

    if (gameSeeds.empty()) { simulation.setSeed(seed); }
    else { simulation.setSeeds(gameSeeds); }

    simulation.setSeedMode(seedMode, seedTarget);

    if (gameMaps.empty()) { simulation.initialize(mapFile); }
    else { simulation.setMaps(gameMaps); simulation.initialize(); }

    simulation.start();
    simulation.run(numberOfTurns);
    return 0;
}
