#include "game.h"
#include "constants.h"
#include <cstdlib>

int main(int argc, char** argv)
{
    int initialNumberHealthy = 790;
    int initialNumberInfected = 10;
    int numberOfTurns = 250;
    unsigned int seed = Constants::DEFAULT_SEED;
    string mapFile = "A1100a.txt";
    string csvFile;
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
        else if (flag == "--out-csv" && hasValue) { csvFile = argv[++i]; }
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

    Game game(initialNumberHealthy, initialNumberInfected);
    game.setSeed(seed);
    game.setSeedMode(seedMode, seedTarget);
    if (!csvFile.empty())
    {
        game.setCsvOutput(csvFile);
    }
    game.initializeGame(mapFile);
    game.startAllDumbThreads();
    game.run(numberOfTurns);
    return 0;
}
