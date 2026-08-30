#ifndef SIMULATION_H
#define SIMULATION_H
#include "game.h"
#include "pool.h"
#include <string>
#include <vector>

using namespace std;

// N ambientes rodando lado a lado sobre um unico pool de agentes.
//
// Um Game deixou de ser "a simulacao" e virou "um ambiente": tem o proprio
// mapa, o proprio retrato de infectados e a propria lista de presentes. O que e
// global - o estado dos agentes e o relogio - mora aqui.
//
// O relogio precisa ser unico. O RNG e indexado por (id, turno), entao dois
// jogos em turnos diferentes dariam ao mesmo agente dois fluxos de sorteio
// dependendo de onde ele estivesse, e a viagem entre jogos passaria a mudar o
// resultado de quem viajou. Com um turno so, viajar nao altera fluxo nenhum.
//
// Cada jogo carrega o proprio mapa, mesmo quando o arquivo e o mesmo: sao
// ambientes distintos e o grafo pode divergir depois. O custo e uma copia do
// grafo por jogo (alguns MB), nao proporcional ao numero de agentes.
class Simulation
{
    public:
        AgentPool pool;
        vector<Game> games;
        int turns = 0;

        Simulation(int numberOfGames, int healthyPerGame, int infectedPerGame);

        // Liga ou desliga os dois niveis de paralelismo (jogos por fora,
        // agentes por dentro). O construtor ja escolhe sozinho pela regra de
        // chooseNestedParallelism; isto e para forcar na mao e medir.
        void setNestedParallelism(bool enabled);

        // Aninha quando ha menos jogos do que threads, e nao aninha quando ha
        // mais. Ver a tabela de medicoes na implementacao.
        void chooseNestedParallelism();

        void setSeed(unsigned int seed);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);

        // Carrega o mapa em todos os jogos e liga cada um ao pool, com faixas
        // de ids globais contiguas e disjuntas.
        void initialize(const string& fileName);

        // Semeia os agentes de todos os jogos.
        void start();

        // Um turno em todos os ambientes.
        void step();

        // Somadas sobre o pool inteiro, e nao sobre os jogos: e a unica visao
        // que conserva o total quando os agentes viajam.
        [[nodiscard]] vector<int> countStates() const;
        void checkConservation(const vector<int>& counts) const;

        void run(int numberOfTurns);
        void reset();

    private:
        int healthyPerGame;
        int infectedPerGame;
        unsigned int baseSeed;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //SIMULATION_H
