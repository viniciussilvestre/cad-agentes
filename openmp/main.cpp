#include "simulation.h"
#include "constants.h"
#include <cstdlib>
#include <omp.h>

int main(int argc, char** argv)
{
    // --agents e --infected sao POR JOGO: com --games 4 --agents 1000, sao
    // 4.000 agentes no total. Assim aumentar o numero de ambientes nao encolhe
    // cada um deles.
    int initialNumberHealthy = 90000000;
    int initialNumberInfected = 10000000;
    int numberOfTurns = 1200;
    int numberOfGames = 1;
    // Zero significa "deixa o OpenMP decidir", que e o padrao dele: o numero de
    // nucleos logicos, ou o que estiver em OMP_NUM_THREADS.
    int numberOfThreads = 0;
    // -1 deixa a Simulation escolher pela relacao entre jogos e threads (ver
    // chooseNestedParallelism). 0 e 1 forcam, para medir.
    int nested = -1;
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
        else if (flag == "--nested") { nested = 1; }
        else if (flag == "--no-nested") { nested = 0; }
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
    if (nested >= 0)
    {
        simulation.setNestedParallelism(nested == 1);
    }
    simulation.setSeed(seed);
    simulation.setSeedMode(seedMode, seedTarget);
    simulation.initialize(mapFile);
    simulation.start();
    simulation.run(numberOfTurns);
    return 0;
}
