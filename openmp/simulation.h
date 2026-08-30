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

        // healthy e infected sao TOTAIS, divididos igualmente entre os jogos.
        // A sobra da divisao vai para os primeiros jogos, um agente cada, para
        // que a soma bata exatamente com o que foi pedido.
        Simulation(int numberOfGames, int healthy, int infected);

        // Liga os dois niveis de paralelismo: os ambientes por fora, os agentes
        // de cada um por dentro. Desligado por padrao, e uma escolha explicita
        // como o numero de threads - o certo depende de quantos jogos ha para
        // dividir entre as threads. Tabela em docs/medicoes.md.
        void setNestedParallelism(bool enabled);

        // Mesma semente para todos os jogos.
        void setSeed(unsigned int seed);

        // Uma semente por jogo. Jogos com o mesmo mapa e a mesma semente ainda
        // assim divergem, porque o RNG e indexado pelo id global e as faixas de
        // id sao disjuntas; a semente por jogo e para quando se quer separar os
        // ambientes de proposito.
        void setSeeds(const vector<unsigned int>& seeds);

        void setSeedMode(SeedMode mode, int regionOrNode = 0);

        // Um arquivo de mapa por jogo.
        void setMaps(const vector<string>& mapFiles);

        // Liga cada jogo ao pool, com faixas de ids globais contiguas e
        // disjuntas, e carrega o mapa de cada um.
        void initialize();

        // Atalho: todos os jogos com o mesmo mapa.
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
        int totalHealthy;
        int totalInfected;
        vector<string> mapFiles;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //SIMULATION_H
