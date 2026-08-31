#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu.h"
#include "constants.h"
#include <iostream>

// Espelho exato do StepParams declarado em kernels.metal. Os campos precisam
// bater em ordem e tamanho; todos sao de 4 bytes, entao nao ha buraco de
// alinhamento a discutir.
//
// As duracoes e limiares viajam por aqui em vez de serem reescritos no MSL:
// constants.h e a fonte unica, e o kernel usa literalmente o que o host
// calculou.
struct StepParams
{
    uint32_t numberOfAgents;
    uint32_t numberOfNodes;
    uint32_t baseSeed;
    int32_t  turn;
    uint32_t slots;
    int32_t  exposedTime;
    int32_t  deathTime;
    uint32_t infectionThreshold;
    uint32_t mortalityThreshold;
};

struct GpuRunner::Impl
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> markPipeline = nil;
    id<MTLComputePipelineState> stepPipeline = nil;

    id<MTLBuffer> positionBuffer = nil;
    id<MTLBuffer> stateBuffer = nil;
    id<MTLBuffer> counterBuffer = nil;
    id<MTLBuffer> neighborsBuffer = nil;
    id<MTLBuffer> infectedBuffer = nil;

    StepParams params = {};
};

GpuRunner::GpuRunner()
{
    impl = new Impl();
}

GpuRunner::~GpuRunner()
{
    delete impl;
}

bool GpuRunner::initialize(const string& metallibPath,
                           int numberOfAgents,
                           int numberOfNodes,
                           const vector<int>& neighbors,
                           const int* agentPosition,
                           const unsigned char* agentState,
                           const unsigned char* agentCounter,
                           unsigned int baseSeed)
{
    @autoreleasepool {
        impl->device = MTLCreateSystemDefaultDevice();
        if (impl->device == nil)
        {
            cerr << "Erro: nenhum device Metal disponivel." << endl;
            return false;
        }

        NSError* error = nil;
        NSString* path = [NSString stringWithUTF8String:metallibPath.c_str()];
        id<MTLLibrary> library = [impl->device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                           error:&error];
        if (library == nil)
        {
            cerr << "Erro: nao foi possivel carregar '" << metallibPath << "': "
                 << [[error localizedDescription] UTF8String] << endl;
            return false;
        }

        id<MTLFunction> markFunction = [library newFunctionWithName:@"markInfected"];
        id<MTLFunction> stepFunction = [library newFunctionWithName:@"stepAgents"];
        if (markFunction == nil || stepFunction == nil)
        {
            cerr << "Erro: kernel 'markInfected' ou 'stepAgents' nao encontrado no metallib." << endl;
            return false;
        }

        impl->markPipeline = [impl->device newComputePipelineStateWithFunction:markFunction error:&error];
        if (impl->markPipeline == nil)
        {
            cerr << "Erro: pipeline de markInfected: " << [[error localizedDescription] UTF8String] << endl;
            return false;
        }
        impl->stepPipeline = [impl->device newComputePipelineStateWithFunction:stepFunction error:&error];
        if (impl->stepPipeline == nil)
        {
            cerr << "Erro: pipeline de stepAgents: " << [[error localizedDescription] UTF8String] << endl;
            return false;
        }

        impl->queue = [impl->device newCommandQueue];

        // StorageModeShared: no Apple Silicon CPU e GPU compartilham a mesma
        // memoria fisica, entao "subir para a GPU" e so alocar aqui e escrever.
        // Nao existe copia por barramento como numa placa discreta, e por isso
        // o host pode ler as contagens todo turno sem custo de transferencia.
        // Estado e contador tem um byte cada; so a posicao precisa de int. Sao
        // 6 bytes por agente em vez de 12 - e como o laco de agentes e limitado
        // por memoria e nao por aritmetica, o que se economiza aqui e tempo, e
        // nao so RAM.
        const NSUInteger positionBytes = (NSUInteger) numberOfAgents * sizeof(int);
        const NSUInteger byteFieldBytes = (NSUInteger) numberOfAgents * sizeof(unsigned char);
        impl->positionBuffer = [impl->device newBufferWithBytes:agentPosition
                                                         length:positionBytes
                                                        options:MTLResourceStorageModeShared];
        impl->stateBuffer = [impl->device newBufferWithBytes:agentState
                                                      length:byteFieldBytes
                                                     options:MTLResourceStorageModeShared];
        impl->counterBuffer = [impl->device newBufferWithBytes:agentCounter
                                                        length:byteFieldBytes
                                                       options:MTLResourceStorageModeShared];
        impl->neighborsBuffer = [impl->device newBufferWithBytes:neighbors.data()
                                                          length:neighbors.size() * sizeof(int)
                                                         options:MTLResourceStorageModeShared];
        impl->infectedBuffer = [impl->device newBufferWithLength:(NSUInteger) numberOfNodes * sizeof(int)
                                                         options:MTLResourceStorageModeShared];

        impl->params.numberOfAgents = (uint32_t) numberOfAgents;
        impl->params.numberOfNodes = (uint32_t) numberOfNodes;
        impl->params.baseSeed = baseSeed;
        impl->params.turn = 0;
        impl->params.slots = (uint32_t) Constants::SLOTS;
        impl->params.exposedTime = Constants::EXPOSED_TIME;
        impl->params.deathTime = Constants::DEATH_TIME;
        impl->params.infectionThreshold = Constants::INFECTION_THRESHOLD;
        impl->params.mortalityThreshold = Constants::MORTALITY_THRESHOLD;

        return true;
    }
}

void GpuRunner::runTurn(int turn)
{
    @autoreleasepool {
        impl->params.turn = turn;

        id<MTLCommandBuffer> commandBuffer = [impl->queue commandBuffer];

        // 1. Zera infectedAtNode inteiro.
        //
        // Um fill de alguns milhares de inteiros pela unidade de blit custa
        // microssegundos e dispensa manter a lista esparsa dos nos sujos, que
        // aqui exigiria uma escrita atomica em append. O caminho de CPU tambem
        // deixou de manter essa lista, pelo mesmo motivo: com muitos agentes,
        // zerar numberOfNodes bytes some perto de varrer numberOfAgents.
        id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
        [blit fillBuffer:impl->infectedBuffer
                   range:NSMakeRange(0, [impl->infectedBuffer length])
                   value:0];
        [blit endEncoding];

        // 2 e 3. Marcacao e passo dos agentes, no mesmo encoder.
        //
        // Um MTLComputeCommandEncoder e serial por padrao, entao o Metal poe a
        // barreira entre os dois dispatches sozinho. E a barreira importa: o
        // retrato dos infectados precisa estar completo antes de qualquer
        // agente ler infectedAtNode.
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:impl->markPipeline];
        [encoder setBuffer:impl->stateBuffer offset:0 atIndex:0];
        [encoder setBuffer:impl->positionBuffer offset:0 atIndex:1];
        [encoder setBuffer:impl->infectedBuffer offset:0 atIndex:2];
        [encoder setBytes:&impl->params length:sizeof(StepParams) atIndex:3];
        NSUInteger markWidth = impl->markPipeline.maxTotalThreadsPerThreadgroup;
        [encoder dispatchThreads:MTLSizeMake(impl->params.numberOfAgents, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(markWidth, 1, 1)];

        [encoder setComputePipelineState:impl->stepPipeline];
        [encoder setBuffer:impl->positionBuffer offset:0 atIndex:0];
        [encoder setBuffer:impl->stateBuffer offset:0 atIndex:1];
        [encoder setBuffer:impl->counterBuffer offset:0 atIndex:2];
        [encoder setBuffer:impl->neighborsBuffer offset:0 atIndex:3];
        [encoder setBuffer:impl->infectedBuffer offset:0 atIndex:4];
        [encoder setBytes:&impl->params length:sizeof(StepParams) atIndex:5];
        NSUInteger stepWidth = impl->stepPipeline.maxTotalThreadsPerThreadgroup;
        [encoder dispatchThreads:MTLSizeMake(impl->params.numberOfAgents, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(stepWidth, 1, 1)];

        [encoder endEncoding];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

const int* GpuRunner::positions() const
{
    return (const int*) [impl->positionBuffer contents];
}

const unsigned char* GpuRunner::states() const
{
    return (const unsigned char*) [impl->stateBuffer contents];
}
