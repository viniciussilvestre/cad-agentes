#ifndef GPU_H
#define GPU_H
#include <string>
#include <vector>

using namespace std;

// Fachada da GPU. A interface e C++ puro de proposito: game.cpp nao inclui
// header nenhum do Metal e nao precisa ser compilado como Objective-C++. Todo
// tipo do Metal fica escondido atras do pimpl, dentro de gpu.mm.
class GpuRunner
{
    public:
        GpuRunner();
        ~GpuRunner();

        GpuRunner(const GpuRunner&) = delete;
        GpuRunner& operator=(const GpuRunner&) = delete;

        // Prepara device, kernels e buffers, e sobe o estado inicial. Devolve
        // false com o motivo em cerr se faltar device, metallib ou kernel.
        bool initialize(const string& metallibPath,
                        int numberOfAgents,
                        int numberOfNodes,
                        const vector<int>& neighbors,
                        const int* agentPosition,
                        const unsigned char* agentState,
                        const unsigned char* agentCounter,
                        unsigned int baseSeed);

        // Um turno inteiro: limpa infectedAtNode, marca os nos com infectado e
        // roda os agentes. Bloqueia ate a GPU terminar, porque o host le as
        // contagens logo depois.
        void runTurn(int turn);

        // Memoria unificada no Apple Silicon: estes ponteiros apontam para a
        // MESMA memoria que os kernels escrevem. Ler daqui depois de runTurn()
        // nao copia nada e nao passa por barramento nenhum.
        [[nodiscard]] const int* positions() const;
        [[nodiscard]] const unsigned char* states() const;

    private:
        struct Impl;
        Impl* impl;
};

#endif //GPU_H
