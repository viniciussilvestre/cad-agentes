#include "simulation.h"
#include "constants.h"
#include <cstdlib>

// ---------------------------------------------------------------------------
// Configuracao por jogo, escrita aqui na mao.
//
// Sao coisas que nao cabem bem numa flag - um mapa e uma semente por ambiente,
// em ordem. Deixe vazio para que todos os jogos usem --map e --seed.
//
// Sem validacao de proposito: quem edita estas listas e quem compila.
// ---------------------------------------------------------------------------
static const vector<string> gameMaps = {
    // "A1100a.txt",
    // "map.txt",
};

static const vector<unsigned int> gameSeeds = {
    // 42,
    // 2024,
};

int main(int argc, char** argv)
{
    // --agents e --infected sao TOTAIS, divididos igualmente entre os jogos:
    // com --games 4 --agents 1000, sao 250 saudaveis por jogo.
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

    if (gameSeeds.empty()) { simulation.setSeed(seed); }
    else { simulation.setSeeds(gameSeeds); }

    simulation.setSeedMode(seedMode, seedTarget);

    if (gameMaps.empty()) { simulation.initialize(mapFile); }
    else { simulation.setMaps(gameMaps); simulation.initialize(); }

    simulation.start();
    simulation.run(numberOfTurns);
    return 0;
}
