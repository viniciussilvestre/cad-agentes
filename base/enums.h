#ifndef ENUMS_H
#define ENUMS_H

enum class States {
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
