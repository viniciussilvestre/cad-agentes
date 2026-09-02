#ifndef SIMULATION_H
#define SIMULATION_H
#include "game.h"
#include "gpu.h"
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
// Layout dos buffers - tudo indexado por id global de agente:
//
//   positions[id]   4 bytes
//   states[id]      1
//   counters[id]    1
//   travelLock[id]  1
//   gameOf[id]      1     (teto de 256 jogos)
//                   -----
//                   8 bytes por agente
//
// Os mapas dos jogos vao concatenados num buffer so, com deslocamento e
// contagem por jogo. Assim o kernel acha a vizinhanca de qualquer agente com
// duas leituras, sem indireccao por ponteiro - que na GPU custaria caro.
//
// ORDEM DE CHAMADA. setSeeds, setSeedMode e setMaps precisam vir ANTES de
// initialize(), porque initialize ja escreve os mapas nos buffers junto com a
// configuracao e libera o grafo do host em seguida. Chamar depois e erro fatal,
// nao silencioso.
class Simulation
{
    public:
        vector<Game> games;
        int turns = 0;

        // healthy e infected sao TOTAIS, divididos igualmente entre os jogos.
        Simulation(int numberOfGames, int healthy, int infected);

        Simulation(const Simulation&) = delete;
        Simulation& operator=(const Simulation&) = delete;

        void setSeed(unsigned int seed);
        void setSeeds(const vector<unsigned int>& seeds);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        void setMaps(const vector<string>& mapFiles);

        // Onde esta o metallib com os kernels. Precisa vir antes de
        // initialize(), como o resto da configuracao.
        void setMetallib(const string& path);

        // Le os mapas e os escreve DIRETO no buffer final. Memoria unificada:
        // nao ha subida separada nem copia intermediaria - o parser escreve no
        // mesmo endereco que o kernel vai ler.
        void initialize();
        void initialize(const string& fileName);

        // Semeia os agentes na GPU.
        void start();

        // Um turno de todos os agentes de todos os jogos. Enfileira e volta.
        void step();

        void run(int numberOfTurns);

        // counts[jogo][estado]. Lido na CPU direto da memoria dos buffers, sem
        // kernel de reducao: com memoria unificada nao ha nada a trazer de
        // volta. Exige que a GPU ja tenha terminado (ver GpuRunner::finish).
        [[nodiscard]] vector<vector<int>> countStatesPerGame() const;

    private:
        void prescanMaps();
        void loadMaps();
        void uploadMetadata();
        void checkConservation(const vector<vector<int>>& counts) const;
        void refuseAfterInitialize(const char* what) const;

        GpuRunner gpu;

        int totalHealthy;
        int totalInfected;
        int numberOfAgents;
        string metallibPath = "kernels.metallib";
        vector<string> mapFiles;
        vector<unsigned int> seeds;
        vector<int> firstAgentOf;
        vector<int> nodeOffset;
        vector<int> nodeCount;

        // Nos candidatos da semeadura por regiao, extraidos por jogo durante a
        // carga: regionNodes morre junto com o resto do mapa do host.
        vector<vector<int>> candidatesOf;

        long long totalNodes = 0;
        bool initialized = false;
};

#endif //SIMULATION_H
