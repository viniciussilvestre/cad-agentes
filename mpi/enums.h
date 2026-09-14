#ifndef ENUMS_H
#define ENUMS_H

// Tipo base explicito: sao cinco valores, cabem num byte. Sem isso o enum vira
// int e agentsStatusVector custa 4 bytes por agente - 400 MB com 100 milhoes de
// agentes, para guardar um numero que vai de 0 a 4.
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
