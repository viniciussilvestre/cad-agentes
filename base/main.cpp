#include "simulation.h"
#include "constants.h"
#include <cstdlib>

int main(int argc, char** argv)
{
    // --agents e --infected sao POR JOGO: com --games 4 --agents 1000, sao
    // 4.000 agentes no total. Assim aumentar o numero de ambientes nao encolhe
    // cada um deles.
    int initialNumberHealthy = 90000000;
    int initialNumberInfected = 10000000;
    int numberOfTurns = 1200;
    int numberOfGames = 1;
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

    Simulation simulation(numberOfGames, initialNumberHealthy, initialNumberInfected);
    simulation.setSeed(seed);
    simulation.setSeedMode(seedMode, seedTarget);
    simulation.initialize(mapFile);
    simulation.start();
    simulation.run(numberOfTurns);
    return 0;
}
