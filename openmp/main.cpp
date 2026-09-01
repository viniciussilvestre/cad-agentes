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
    "map_14.txt", "map_15.txt", "map_16.txt", "map_17.txt", "map_18.txt", "map_19.txt", "map_20.txt", 
    "map_21.txt", "map_22.txt", "map_23.txt", "map_24.txt", "map_25.txt", "map_26.txt", "map_27.txt", 
    "map_28.txt", "map_29.txt", "map_30.txt", "map_31.txt", "map_32.txt", "map_33.txt", "map_34.txt", 
    "map_35.txt", "map_36.txt", "map_37.txt", "map_38.txt", "map_39.txt", "map_40.txt", "map_41.txt", 
    "map_42.txt", "map_43.txt", "map_44.txt", "map_45.txt", "map_46.txt", "map_47.txt", "map_48.txt", 
    "map_49.txt", "map_50.txt", "map_51.txt", "map_52.txt", "map_53.txt", "map_54.txt", "map_55.txt", 
    "map_56.txt", "map_57.txt", "map_58.txt", "map_59.txt", "map_60.txt", "map_61.txt", "map_62.txt", 
    "map_63.txt", "map_64.txt", "map_65.txt", "map_66.txt", "map_67.txt", "map_68.txt", "map_69.txt", 
    "map_70.txt", "map_71.txt", "map_72.txt", "map_73.txt", "map_74.txt", "map_75.txt", "map_76.txt", 
    "map_77.txt", "map_78.txt", "map_79.txt", "map_80.txt", "map_81.txt", "map_82.txt", "map_83.txt", 
    "map_84.txt", "map_85.txt", "map_86.txt", "map_87.txt", "map_88.txt", "map_89.txt", "map_90.txt", 
    "map_91.txt", "map_92.txt", "map_93.txt", "map_94.txt", "map_95.txt", "map_96.txt", "map_97.txt", 
    "map_98.txt", "map_99.txt", "map_100.txt", "map_101.txt", "map_102.txt", "map_103.txt", "map_104.txt", 
    "map_105.txt", "map_106.txt", "map_107.txt", "map_108.txt", "map_109.txt", "map_110.txt", "map_111.txt", 
    "map_112.txt", "map_113.txt", "map_114.txt", "map_115.txt", "map_116.txt", "map_117.txt", "map_118.txt", 
    "map_119.txt", "map_120.txt", "map_121.txt", "map_122.txt", "map_123.txt", "map_124.txt", "map_125.txt", 
    "map_126.txt", "map_127.txt", "map_128.txt"
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
    int initialNumberHealthy = 1201884940;
    int initialNumberInfected = 20000000;
    int numberOfTurns = 600;
    int numberOfGames = 20;
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
