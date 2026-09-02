#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "gpu.h"
#include "constants.h"
#include <iostream>
#include <mutex>

// Espelhos exatos dos structs declarados em kernels.metal. Ordem e largura
// precisam bater campo a campo.
//
// As duracoes e limiares viajam por aqui em vez de serem reescritos no MSL:
// constants.h e a fonte unica, e o kernel usa literalmente o que o host
// calculou.
struct FillParams
{
    uint64_t base;
    uint32_t nodes;
    uint32_t padding;
};

struct SeedParams
{
    uint32_t numberOfAgents;
    uint32_t numberOfGames;
};

struct StepParams
{
    uint32_t numberOfAgents;
    uint32_t numberOfGames;
    int32_t  turn;
    uint32_t slots;
    int32_t  exposedTime;
    int32_t  deathTime;
    uint32_t infectionThreshold;
    uint32_t mortalityThreshold;
    uint32_t travelThreshold;
    uint32_t travelLockTurns;
};

struct GpuRunner::Impl
{
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;

    id<MTLComputePipelineState> fillPipeline = nil;
    id<MTLComputePipelineState> seedPipeline = nil;
    id<MTLComputePipelineState> markPipeline = nil;
    id<MTLComputePipelineState> stepPipeline = nil;

    // Estado dos agentes, indexado por id global.
    id<MTLBuffer> positions = nil;
    id<MTLBuffer> states = nil;
    id<MTLBuffer> counters = nil;
    id<MTLBuffer> travelLock = nil;
    id<MTLBuffer> gameOf = nil;

    // Mapas concatenados e a mascara de infectados.
    id<MTLBuffer> neighbors = nil;
    id<MTLBuffer> infectedMask = nil;

    // Metadados por jogo.
    id<MTLBuffer> nodeOffset = nil;
    id<MTLBuffer> nodeCount = nil;
    id<MTLBuffer> firstAgent = nil;
    id<MTLBuffer> healthyOf = nil;
    id<MTLBuffer> seeds = nil;
    id<MTLBuffer> seedMode = nil;
    id<MTLBuffer> seedTarget = nil;
    id<MTLBuffer> candidates = nil;
    id<MTLBuffer> candidateOffset = nil;
    id<MTLBuffer> candidateCount = nil;

    // Ultimo command buffer enfileirado. finish() espera por ele; como a fila e
    // ordenada, esperar o ultimo espera todos.
    //
    // O mutex existe por causa da carga dos mapas: ela roda em paralelo e cada
    // thread enfileira o proprio fillSelfLoops. A MTLCommandQueue ja e segura
    // para criar command buffers de varias threads; quem nao e segura e esta
    // variavel. O laco de turnos e serial e nao disputa nada.
    id<MTLCommandBuffer> lastCommand = nil;
    mutex commandMutex;

    int numberOfAgents = 0;
    int numberOfGames = 0;
    uint64_t maskWords = 0;
    StepParams step = {};
};

GpuRunner::GpuRunner()
{
    impl = new Impl();
}

GpuRunner::~GpuRunner()
{
    delete impl;
}

static id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                                id<MTLLibrary> library,
                                                NSString* name)
{
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil)
    {
        cerr << "Erro: kernel '" << [name UTF8String] << "' nao encontrado no metallib." << endl;
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
    {
        cerr << "Erro: pipeline de '" << [name UTF8String] << "': "
             << [[error localizedDescription] UTF8String] << endl;
    }
    return pipeline;
}

bool GpuRunner::initialize(const string& metallibPath,
                           long long totalNodes,
                           int numberOfAgents,
                           int numberOfGames)
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

        impl->fillPipeline = makePipeline(impl->device, library, @"fillSelfLoops");
        impl->seedPipeline = makePipeline(impl->device, library, @"seedAgents");
        impl->markPipeline = makePipeline(impl->device, library, @"markInfected");
        impl->stepPipeline = makePipeline(impl->device, library, @"stepAgents");
        if (impl->fillPipeline == nil || impl->seedPipeline == nil ||
            impl->markPipeline == nil || impl->stepPipeline == nil)
        {
            return false;
        }

        impl->queue = [impl->device newCommandQueue];
        impl->numberOfAgents = numberOfAgents;
        impl->numberOfGames = numberOfGames;

        // Um BIT por no, e nao um byte: o consumidor e uma leitura aleatoria por
        // agente por turno, entao o que decide o custo e quanto do vetor cabe na
        // cache. Dividir por oito e a diferenca entre caber e nao caber.
        impl->maskWords = (uint64_t) ((totalNodes + 31) / 32);

        const NSUInteger neighborBytes =
            (NSUInteger) totalNodes * Constants::SLOTS * sizeof(int);
        const NSUInteger maskBytes = (NSUInteger) impl->maskWords * sizeof(uint32_t);

        NSUInteger limit = [impl->device maxBufferLength];
        if (neighborBytes > limit)
        {
            cerr << "Erro: os mapas somam " << (neighborBytes >> 20) << " MB, acima do teto de "
                 << (limit >> 20) << " MB por MTLBuffer neste device." << endl;
            return false;
        }

        impl->neighbors = [impl->device newBufferWithLength:neighborBytes
                                                    options:MTLResourceStorageModeShared];
        impl->infectedMask = [impl->device newBufferWithLength:maskBytes
                                                       options:MTLResourceStorageModeShared];
        if (impl->neighbors == nil || impl->infectedMask == nil)
        {
            cerr << "Erro: falha ao alocar os buffers do grafo." << endl;
            return false;
        }

        impl->step.numberOfAgents = (uint32_t) numberOfAgents;
        impl->step.numberOfGames = (uint32_t) numberOfGames;
        impl->step.turn = 0;
        impl->step.slots = (uint32_t) Constants::SLOTS;
        impl->step.exposedTime = Constants::EXPOSED_TIME;
        impl->step.deathTime = Constants::DEATH_TIME;
        impl->step.infectionThreshold = Constants::INFECTION_THRESHOLD;
        impl->step.mortalityThreshold = Constants::MORTALITY_THRESHOLD;
        impl->step.travelThreshold = Constants::TRAVEL_THRESHOLD;
        impl->step.travelLockTurns = (uint32_t) Constants::TRAVEL_LOCK_TURNS;

        return true;
    }
}

int* GpuRunner::neighborsAt(long long nodeOffset) const
{
    int* base = (int*) [impl->neighbors contents];
    return base + (size_t) nodeOffset * Constants::SLOTS;
}

void GpuRunner::fillSelfLoops(long long nodeOffset, int nodes)
{
    @autoreleasepool {
        FillParams params = {};
        params.base = (uint64_t) nodeOffset * Constants::SLOTS;
        params.nodes = (uint32_t) nodes;
        params.padding = 0;

        id<MTLCommandBuffer> command = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:impl->fillPipeline];
        [encoder setBuffer:impl->neighbors offset:0 atIndex:0];
        [encoder setBytes:&params length:sizeof(FillParams) atIndex:1];

        // Grade 2D: x percorre os nos, y percorre os slots. Assim o kernel acha
        // o indice sem divisao.
        NSUInteger width = impl->fillPipeline.threadExecutionWidth;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) nodes, Constants::SLOTS, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];

        lock_guard<mutex> guard(impl->commandMutex);
        impl->lastCommand = command;
    }
}
void GpuRunner::uploadMetadata(const vector<int>& nodeOffset,
                               const vector<int>& nodeCount,
                               const vector<int>& firstAgent,
                               const vector<int>& healthyOf,
                               const vector<unsigned int>& seeds,
                               const vector<int>& seedMode,
                               const vector<int>& seedTarget,
                               const vector<int>& candidates,
                               const vector<int>& candidateOffset,
                               const vector<int>& candidateCount)
{
    @autoreleasepool {
        auto upload = [this](const void* source, size_t bytes) {
            return [impl->device newBufferWithBytes:source
                                             length:(NSUInteger) bytes
                                            options:MTLResourceStorageModeShared];
        };

        impl->nodeOffset = upload(nodeOffset.data(), nodeOffset.size() * sizeof(int));
        impl->nodeCount = upload(nodeCount.data(), nodeCount.size() * sizeof(int));
        impl->firstAgent = upload(firstAgent.data(), firstAgent.size() * sizeof(int));
        impl->healthyOf = upload(healthyOf.data(), healthyOf.size() * sizeof(int));
        impl->seeds = upload(seeds.data(), seeds.size() * sizeof(unsigned int));
        impl->seedMode = upload(seedMode.data(), seedMode.size() * sizeof(int));
        impl->seedTarget = upload(seedTarget.data(), seedTarget.size() * sizeof(int));
        impl->candidates = upload(candidates.data(), candidates.size() * sizeof(int));
        impl->candidateOffset = upload(candidateOffset.data(), candidateOffset.size() * sizeof(int));
        impl->candidateCount = upload(candidateCount.data(), candidateCount.size() * sizeof(int));
    }
}

void GpuRunner::seedAgents()
{
    @autoreleasepool {
        const NSUInteger agents = (NSUInteger) impl->numberOfAgents;
        impl->positions = [impl->device newBufferWithLength:agents * sizeof(int)
                                                    options:MTLResourceStorageModeShared];
        impl->states = [impl->device newBufferWithLength:agents
                                                 options:MTLResourceStorageModeShared];
        impl->counters = [impl->device newBufferWithLength:agents
                                                   options:MTLResourceStorageModeShared];
        impl->travelLock = [impl->device newBufferWithLength:agents
                                                     options:MTLResourceStorageModeShared];
        impl->gameOf = [impl->device newBufferWithLength:agents
                                                 options:MTLResourceStorageModeShared];

        SeedParams params = {};
        params.numberOfAgents = (uint32_t) impl->numberOfAgents;
        params.numberOfGames = (uint32_t) impl->numberOfGames;

        id<MTLCommandBuffer> command = [impl->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:impl->seedPipeline];
        [encoder setBuffer:impl->positions offset:0 atIndex:0];
        [encoder setBuffer:impl->states offset:0 atIndex:1];
        [encoder setBuffer:impl->counters offset:0 atIndex:2];
        [encoder setBuffer:impl->travelLock offset:0 atIndex:3];
        [encoder setBuffer:impl->gameOf offset:0 atIndex:4];
        [encoder setBuffer:impl->firstAgent offset:0 atIndex:5];
        [encoder setBuffer:impl->healthyOf offset:0 atIndex:6];
        [encoder setBuffer:impl->nodeCount offset:0 atIndex:7];
        [encoder setBuffer:impl->seeds offset:0 atIndex:8];
        [encoder setBuffer:impl->seedMode offset:0 atIndex:9];
        [encoder setBuffer:impl->seedTarget offset:0 atIndex:10];
        [encoder setBuffer:impl->candidates offset:0 atIndex:11];
        [encoder setBuffer:impl->candidateOffset offset:0 atIndex:12];
        [encoder setBuffer:impl->candidateCount offset:0 atIndex:13];
        [encoder setBytes:&params length:sizeof(SeedParams) atIndex:14];

        NSUInteger width = impl->seedPipeline.maxTotalThreadsPerThreadgroup;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) impl->numberOfAgents, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        impl->lastCommand = command;
    }
}

void GpuRunner::runTurn(int turn)
{
    @autoreleasepool {
        impl->step.turn = turn;

        id<MTLCommandBuffer> command = [impl->queue commandBuffer];

        // 1. Zera a mascara inteira. Um fill pela unidade de blit custa
        // microssegundos e dispensa manter a lista esparsa dos nos sujos, que
        // aqui exigiria uma escrita atomica em append.
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        [blit fillBuffer:impl->infectedMask
                   range:NSMakeRange(0, [impl->infectedMask length])
                   value:0];
        [blit endEncoding];

        // 2 e 3. Marcacao e passo dos agentes, no mesmo encoder.
        //
        // Um MTLComputeCommandEncoder e serial por padrao, entao o Metal poe a
        // barreira entre os dois dispatches sozinho. E a barreira importa: a
        // mascara precisa estar congelada antes de qualquer agente le-la.
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];

        [encoder setComputePipelineState:impl->markPipeline];
        [encoder setBuffer:impl->positions offset:0 atIndex:0];
        [encoder setBuffer:impl->states offset:0 atIndex:1];
        [encoder setBuffer:impl->gameOf offset:0 atIndex:2];
        [encoder setBuffer:impl->nodeOffset offset:0 atIndex:3];
        [encoder setBuffer:impl->infectedMask offset:0 atIndex:4];
        [encoder setBytes:&impl->step length:sizeof(StepParams) atIndex:5];
        NSUInteger markWidth = impl->markPipeline.maxTotalThreadsPerThreadgroup;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) impl->numberOfAgents, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(markWidth, 1, 1)];

        [encoder setComputePipelineState:impl->stepPipeline];
        [encoder setBuffer:impl->positions offset:0 atIndex:0];
        [encoder setBuffer:impl->states offset:0 atIndex:1];
        [encoder setBuffer:impl->counters offset:0 atIndex:2];
        [encoder setBuffer:impl->travelLock offset:0 atIndex:3];
        [encoder setBuffer:impl->gameOf offset:0 atIndex:4];
        [encoder setBuffer:impl->neighbors offset:0 atIndex:5];
        [encoder setBuffer:impl->infectedMask offset:0 atIndex:6];
        [encoder setBuffer:impl->nodeOffset offset:0 atIndex:7];
        [encoder setBuffer:impl->nodeCount offset:0 atIndex:8];
        [encoder setBuffer:impl->seeds offset:0 atIndex:9];
        // setBytes COPIA para dentro do command buffer, entao cada turno leva o
        // proprio numero de turno mesmo que a execucao aconteca bem depois do
        // commit. E o que permite enfileirar sem esperar.
        [encoder setBytes:&impl->step length:sizeof(StepParams) atIndex:10];
        NSUInteger stepWidth = impl->stepPipeline.maxTotalThreadsPerThreadgroup;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger) impl->numberOfAgents, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(stepWidth, 1, 1)];

        [encoder endEncoding];
        [command commit];
        impl->lastCommand = command;
    }
}

void GpuRunner::finish()
{
    if (impl->lastCommand != nil)
    {
        // A fila e ordenada, entao esperar o ultimo espera todos os anteriores.
        [impl->lastCommand waitUntilCompleted];
        impl->lastCommand = nil;
    }
}

const unsigned char* GpuRunner::states() const
{
    return (const unsigned char*) [impl->states contents];
}

const unsigned char* GpuRunner::gameOf() const
{
    return (const unsigned char*) [impl->gameOf contents];
}
