#ifndef SIMULATION_H
#define SIMULATION_H
#include "game.h"
#include <string>
#include <vector>

using namespace std;

// N ambientes rodando lado a lado, com o laco de agentes na GPU.
//
// A diferenca de forma para as versoes de CPU e que aqui NAO existe lista de
// presentes por jogo. Cada agente carrega o indice do jogo em que esta, e um
// unico kernel cobre todos os agentes de todos os jogos. O jogo deixa de ser
// uma colecao e vira so um mapa mais um deslocamento.
//
// Layout no device - tudo indexado por id global de agente:
//
//   positions[id]   4 bytes
//   states[id]      1
//   counters[id]    1
//   travelLock[id]  1
//   gameOf[id]      1     (teto de 256 jogos)
//                   -----
//                   8 bytes por agente
//
// Os mapas dos jogos vao concatenados num vetor so, com deslocamento e
// contagem por jogo. Assim o kernel acha a vizinhanca de qualquer agente com
// duas leituras, sem indireccao por ponteiro - que na GPU custaria caro.
//
// ORDEM DE CHAMADA. setSeeds, setSeedMode e setMaps precisam vir ANTES de
// initialize(), porque initialize ja sobe os mapas e a configuracao junto com
// eles, e libera o grafo do host em seguida. Chamar depois e erro fatal, nao
// silencioso.
class Simulation
{
    public:
        vector<Game> games;
        int turns = 0;

        // healthy e infected sao TOTAIS, divididos igualmente entre os jogos.
        Simulation(int numberOfGames, int healthy, int infected);
        ~Simulation();

        Simulation(const Simulation&) = delete;
        Simulation& operator=(const Simulation&) = delete;

        void setSeed(unsigned int seed);
        void setSeeds(const vector<unsigned int>& seeds);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        void setMaps(const vector<string>& mapFiles);

        // Le os mapas E os sobe para o device, em pipeline: cada jogo que
        // termina de ser parseado sobe na hora, na sua propria stream, enquanto
        // os outros ainda estao sendo lidos.
        void initialize();
        void initialize(const string& fileName);

        // Semeia os agentes no device.
        void start();

        // Um turno de todos os agentes de todos os jogos.
        void step();

        void run(int numberOfTurns);

        // counts[jogo][estado], trazido do device.
        [[nodiscard]] vector<vector<int>> countStatesPerGame();

    private:
        void prescanMaps();
        void loadAndUploadMaps();
        void uploadMetadata();
        void releaseDevice();
        void checkConservation(const vector<vector<int>>& counts) const;
        void refuseAfterInitialize(const char* what) const;

        int totalHealthy;
        int totalInfected;
        int numberOfAgents;
        vector<string> mapFiles;
        vector<unsigned int> seeds;
        vector<int> firstAgentOf;
        vector<int> nodeOffset;
        vector<int> nodeCount;

        // Nos candidatos da semeadura por regiao, extraidos por jogo durante a
        // carga: regionNodes morre junto com o resto do mapa do host.
        vector<vector<int>> candidatesOf;

        // Estado dos agentes, no device.
        int* devicePositions = nullptr;
        unsigned char* deviceStates = nullptr;
        unsigned char* deviceCounters = nullptr;
        unsigned char* deviceTravelLock = nullptr;
        unsigned char* deviceGameOf = nullptr;

        // Mapas concatenados.
        int* deviceNeighbors = nullptr;
        int* deviceNodeOffset = nullptr;
        int* deviceNodeCount = nullptr;

        // Um BIT por no, e nao um byte: o consumidor e uma leitura aleatoria
        // por agente por turno, entao o que decide o custo e quanto do vetor
        // cabe na L2. Dividir por oito e a diferenca entre caber e nao caber.
        unsigned int* deviceInfectedMask = nullptr;

        // Semeadura.
        unsigned int* deviceSeeds = nullptr;
        int* deviceFirstAgent = nullptr;
        int* deviceHealthyOf = nullptr;
        int* deviceSeedMode = nullptr;
        int* deviceSeedTarget = nullptr;
        int* deviceCandidates = nullptr;
        int* deviceCandidateOffset = nullptr;
        int* deviceCandidateCount = nullptr;

        int* deviceCounts = nullptr;
        long long totalNodes = 0;
        long long infectedMaskWords = 0;
        bool ready = false;
        bool initialized = false;
};

#endif //SIMULATION_H
