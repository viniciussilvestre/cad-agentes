#ifndef SIMULATION_H
#define SIMULATION_H
#include "game.h"
#include "pool.h"
#include "departure.h"
#include <string>
#include <vector>

using namespace std;

// N ambientes divididos entre os ranks de MPI.
//
// DECOMPOSICAO. Cada rank e dono de um subconjunto dos jogos (`g % ranks`) e so
// carrega os mapas desses. E o ganho principal aqui: com 28 jogos e 4 ranks,
// cada maquina segura um quarto dos grafos em vez dos 28 inteiros. O teto de
// paralelismo, em compensacao, e o numero de jogos - nao adianta pedir mais
// ranks que isso, porque o laco de agentes de um jogo nao e dividido entre
// ranks. Dividi-lo exigiria replicar infectedAtNode e reduzi-lo todo turno, o
// que com mapas grandes custa mais que o trabalho que economiza.
//
// POOL REPLICADO EM ALOCACAO, NAO EM CONTEUDO. Todo rank aloca o pool inteiro,
// indexado por id global, mas so escreve nos slots dos agentes que estao nos
// seus jogos. Custa 7 bytes por agente por rank - 7,7 MB no maior ponto da
// varredura - e em troca o id continua sendo o indice direto, sem tabela de
// traducao no laco quente. Por isso o estado viaja junto na mensagem: o slot do
// viajante, no rank de destino, esta velho.
//
// VIAGEM. Toda troca de jogo vira uma Departure, mesmo quando o destino e do
// proprio rank, e todas sao aplicadas na barreira entre turnos. Ver
// departure.h: e o que faz a saida nao depender do numero de ranks.
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

        void setSeed(unsigned int seed);
        void setSeeds(const vector<unsigned int>& seeds);
        void setSeedMode(SeedMode mode, int regionOrNode = 0);
        void setMaps(const vector<string>& mapFiles);

        // Carrega SO os mapas dos jogos deste rank, e combina os tamanhos de
        // todos por Allgather - a viagem precisa deles para sortear o no de
        // chegada num grafo que este rank talvez nunca tenha lido.
        void initialize();
        void initialize(const string& fileName);

        // Semeia os agentes dos jogos deste rank.
        void start();

        // Um turno: os jogos locais andam, os viajantes sao trocados, as listas
        // sao compactadas.
        void step();

        // Somadas sobre os jogos LOCAIS e reduzidas entre os ranks. Contar o
        // pool nao serviria: ele e replicado, entao cada agente apareceria uma
        // vez por rank.
        [[nodiscard]] vector<int> countStates() const;
        void checkConservation(const vector<int>& counts) const;

        void run(int numberOfTurns);

        [[nodiscard]] int rankOf() const { return rank; }
        [[nodiscard]] bool isRoot() const { return rank == 0; }

    private:
        void exchangeTravellers();

        int rank = 0;
        int ranks = 1;
        // Dono de cada jogo, e os indices dos que sao deste rank.
        vector<int> gameOwner;
        vector<int> localGames;
        // Numero de nos de cada jogo, inclusive os de outros ranks.
        vector<int> nodeCountOf;
        // Viagens do turno corrente, de todos os jogos locais.
        vector<Departure> outbox;

        int totalHealthy;
        int totalInfected;
        vector<string> mapFiles;
        SeedMode seedMode = SeedMode::Uniform;
        int seedTarget = 0;
};

#endif //SIMULATION_H
