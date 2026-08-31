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
    "map_0", "map_1", "map_2", "map_3", "map_4", "map_5", "map_6", 
    "map_7", "map_8", "map_9", "map_10", "map_11", "map_12", "map_13", 
    "map_14", "map_15", "map_16", "map_17", "map_18", "map_19", "map_20", 
    "map_21", "map_22", "map_23", "map_24", "map_25", "map_26", "map_27", 
    "map_28", "map_29", "map_30", "map_31", "map_32", "map_33", "map_34", 
    "map_35", "map_36", "map_37", "map_38", "map_39", "map_40", "map_41", 
    "map_42", "map_43", "map_44", "map_45", "map_46", "map_47", "map_48", 
    "map_49", "map_50", "map_51", "map_52", "map_53", "map_54", "map_55", 
    "map_56", "map_57", "map_58", "map_59", "map_60", "map_61", "map_62", 
    "map_63", "map_64", "map_65", "map_66", "map_67", "map_68", "map_69", 
    "map_70", "map_71", "map_72", "map_73", "map_74", "map_75", "map_76", 
    "map_77", "map_78", "map_79", "map_80", "map_81", "map_82", "map_83", 
    "map_84", "map_85", "map_86", "map_87", "map_88", "map_89", "map_90", 
    "map_91", "map_92", "map_93", "map_94", "map_95", "map_96", "map_97", 
    "map_98", "map_99", "map_100", "map_101", "map_102", "map_103", "map_104", 
    "map_105", "map_106", "map_107", "map_108", "map_109", "map_110", "map_111", 
    "map_112", "map_113", "map_114", "map_115", "map_116", "map_117", "map_118", 
    "map_119", "map_120", "map_121", "map_122", "map_123", "map_124", "map_125", 
    "map_126", "map_127", "map_128"
};

static const vector<unsigned int> gameSeeds = {
    4, 87, 12, 93, 41, 65, 18, 30, 99, 52, 7, 76, 23, 88, 34, 
    11, 62, 45, 90, 19, 81, 3, 58, 27, 95, 14, 70, 39, 83, 6, 
    50, 72, 31, 96, 17, 64, 2, 85, 43, 78, 22, 59, 10, 91, 36, 
    68, 5, 48, 80, 15, 73, 29, 94, 53, 21, 66, 8, 89, 38, 75, 
    13, 60, 42, 97, 25, 84, 1, 49, 71, 33, 92, 56, 16, 79, 28, 
    63, 9, 86, 37, 74, 20, 54, 82, 47, 98, 24, 69, 32, 57, 100, 
    44, 77, 26, 61, 35, 85, 51, 67, 93, 19, 40, 73, 12, 88, 55, 
    30, 96, 22, 64, 46, 81, 15, 70, 27, 90, 58, 3, 72, 39, 84, 
    17, 63, 49, 95, 31, 68, 8, 53
};

int main(int argc, char** argv)
{
    // --agents e --infected sao TOTAIS, divididos igualmente entre os jogos:
    // com --games 4 --agents 1000, sao 250 saudaveis por jogo.
    int initialNumberHealthy = 90000000;
    int initialNumberInfected = 10000000;
    int numberOfTurns = 1000;
    int numberOfGames = 128;
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
