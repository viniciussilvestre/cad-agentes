#ifndef ENUMS_H
#define ENUMS_H

// Tipo base explicito: sao cinco valores, cabem num byte. O enum atravessa a
// fronteira host/kernel, entao a largura faz parte do contrato do MTLBuffer -
// int aqui custaria 4 bytes por agente dos dois lados.
enum class States : unsigned char {
    SUSCEPTIBLE = 0,
    EXPOSED = 1,
    INFECTED = 2,
    RECOVERED = 3,
    DEAD = 4
};

// Onde os infectados iniciais comecam. O passeio aleatorio leva centenas de
// turnos para misturar, entao a semeadura muda a dinamica por completo.
enum class SeedMode {
    Uniform = 0,
    SingleRegion = 1,
    SingleNode = 2
};

#endif //ENUMS_H
