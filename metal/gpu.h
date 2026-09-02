#ifndef GPU_H
#define GPU_H
#include <string>
#include <vector>

using namespace std;

// Fachada da GPU. A interface e C++ puro de proposito: simulation.cpp e
// game.cpp nao incluem header nenhum do Metal e nao precisam ser compilados
// como Objective-C++. Todo tipo do Metal fica escondido atras do pimpl, dentro
// de gpu.mm.
//
// MEMORIA UNIFICADA. No Apple Silicon a GPU le a mesma memoria fisica que a
// CPU, entao "subir para o device" nao existe: alocar um MTLBuffer com
// StorageModeShared e escrever nele JA e subir. Duas consequencias que separam
// esta versao da CUDA:
//
//   - o mapa de cada jogo e parseado direto para dentro do buffer final
//     (neighborsAt), sem copia intermediaria nem pipeline de streams;
//   - as contagens do fim saem lendo states()/gameOf() na CPU, sem kernel de
//     reducao e sem trazer nada de volta. A CUDA precisa de um countKernel com
//     privatizacao em memoria compartilhada porque la a leitura atravessa PCIe.
class GpuRunner
{
    public:
        GpuRunner();
        ~GpuRunner();

        GpuRunner(const GpuRunner&) = delete;
        GpuRunner& operator=(const GpuRunner&) = delete;

        // Device, pipelines e os dois buffers grandes: os mapas concatenados e
        // a mascara de infectados. Devolve false com o motivo em cerr.
        bool initialize(const string& metallibPath,
                        long long totalNodes,
                        int numberOfAgents,
                        int numberOfGames);

        // Onde o mapa de um jogo deve ser escrito dentro do vetor concatenado.
        // Escrever aqui e subir: e o buffer que o kernel le.
        [[nodiscard]] int* neighborsAt(long long nodeOffset) const;

        // Preenche os slots vazios (-1) com auto-laco, no mapa recem-escrito.
        // Enfileira e volta; quem espera e finish().
        void fillSelfLoops(long long nodeOffset, int nodes);

        // Metadados por jogo. Todos os vetores precisam ter numberOfGames
        // entradas, menos candidates, que e a concatenacao.
        void uploadMetadata(const vector<int>& nodeOffset,
                            const vector<int>& nodeCount,
                            const vector<int>& firstAgent,
                            const vector<int>& healthyOf,
                            const vector<unsigned int>& seeds,
                            const vector<int>& seedMode,
                            const vector<int>& seedTarget,
                            const vector<int>& candidates,
                            const vector<int>& candidateOffset,
                            const vector<int>& candidateCount);

        // Aloca o estado dos agentes e sorteia posicao e estado inicial.
        void seedAgents();

        // Um turno: limpa a mascara, marca os infectados e roda os agentes.
        // NAO espera a GPU - os turnos sao enfileirados um atras do outro, como
        // na CUDA. Sem isso o cronometro mediria ida e volta por turno em vez
        // do tempo de execucao.
        void runTurn(int turn);

        // Espera a fila esvaziar. Obrigatorio antes de parar o cronometro ou de
        // ler states()/gameOf().
        void finish();

        // Memoria unificada: apontam para a MESMA memoria que os kernels
        // escrevem. Validos depois de finish(), sem copia nenhuma.
        [[nodiscard]] const unsigned char* states() const;
        [[nodiscard]] const unsigned char* gameOf() const;

    private:
        struct Impl;
        Impl* impl;
};

#endif //GPU_H
