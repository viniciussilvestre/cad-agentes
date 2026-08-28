#ifndef CONSTANTS_H
#define CONSTANTS_H
#include <cstdint>

namespace Constants
{
    constexpr const int INFECTION_RATE {10};
    constexpr const int INFECTION_TIME {10};
    constexpr const int DEATH_TIME {3};
    constexpr const int EXPOSED_TIME {7};
    constexpr const int RECOVERY_TIME = {10};
    constexpr const double MORTALITY = {4.4};
    constexpr const int lowerResolutionX = {62};
    constexpr const int lowerResolutionY = {60};
    constexpr const double widthPercentage {0.976};
    constexpr const double heightPercentage {0.902};
    constexpr const int resolutionX = {1920};
    constexpr const int resolutionY = {1080};
    constexpr const int SLOTS = {6};
    constexpr const unsigned int DEFAULT_SEED = {62};

    // INFECTION_TIME e o curso MAXIMO da doenca: EXPOSED_TIME turnos de
    // incubacao no pior caso, mais os DEATH_TIME turnos da fase infecciosa. A
    // incubacao tem duracao variavel (ver Player::infection), entao o curso
    // tipico e mais curto que INFECTION_TIME - a identidade abaixo so amarra o
    // teto. Se as tres constantes deixarem de fechar, esse teto passa a mentir.
    static_assert(EXPOSED_TIME + DEATH_TIME == INFECTION_TIME,
                  "EXPOSED_TIME + DEATH_TIME precisa ser igual a INFECTION_TIME");

    // As duas taxas acima, reescritas como limiar inteiro em fracao de 2^32.
    //
    // A comparacao passa a ser feita em inteiro, e nao em double, porque
    // contracao FMA pode dar resultados diferentes em CPU e GPU para a mesma
    // conta em ponto flutuante. Com limiar inteiro a versao CUDA reproduz o
    // mesmo golden que a sequencial, e "a GPU esta certa?" vira uma pergunta de
    // resposta binaria em vez de estatistica.
    //
    // A conversao acontece em tempo de compilacao, entao o double aparece so
    // aqui e nunca no laco.
    constexpr const uint32_t INFECTION_THRESHOLD =
        (uint32_t) (INFECTION_RATE / 100.0 * 4294967296.0);
    constexpr const uint32_t MORTALITY_THRESHOLD =
        (uint32_t) (MORTALITY / 100.0 * 4294967296.0);

    static_assert(INFECTION_RATE >= 0 && INFECTION_RATE < 100,
                  "INFECTION_RATE precisa estar em [0, 100) para caber no limiar de 32 bits");
    static_assert(MORTALITY >= 0.0 && MORTALITY < 100.0,
                  "MORTALITY precisa estar em [0, 100) para caber no limiar de 32 bits");
}

#endif
