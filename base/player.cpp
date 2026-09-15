#include "player.h"
#include "constants.h"

// Grau fixo: todo no tem exatamente SLOTS slots e os vazios apontam para o
// proprio no. Sorteia-se um slot e pronto - sem laco de rejeicao, sem -1, sem
// checagem de limites, sem branch.
//
// Layout SoA sobre os slots: neighbors[k * numberOfNodes + i]. Agentes que
// sorteiem o mesmo k leem posicoes proximas na memoria.
static void updatePosition(int id, const vector<int>& neighbors, int numberOfNodes,
                           vector<int>& positions, AgentRng& rng)
{
    int slot = (int) rng.below(Constants::SLOTS);
    positions[id] = neighbors[slot * numberOfNodes + positions[id]];
}

static void checkSurroundings(int id, const vector<unsigned char>& infectedAtNode,
                              const vector<int>& positions, vector<States>& states,
                              vector<unsigned char>& counters)
{
    States state = states[id];
    if (state != States::SUSCEPTIBLE && state != States::RECOVERED)
    {
        return;
    }
    // O(1): para fins de contato basta saber se ha algum INFECTED no no. O
    // proprio agente e SUSCEPTIBLE ou RECOVERED aqui, entao nunca se conta a si
    // mesmo. infectedAtNode e reconstruido pelo Game no inicio do turno.
    //
    // REAVALIAR: dois pontos herdados, mantidos de proposito por enquanto.
    //
    // 1. O contato expoe com probabilidade 1 - basta um INFECTED no no. Se
    //    INFECTION_RATE algum dia migrar para ca (ver a nota em infection), e
    //    exatamente aqui que o sorteio entraria.
    //
    // 2. RECOVERED tambem e reexposto pela condicao acima, entao nao existe
    //    imunidade: o modelo e SEIRS, nao SEIR. Cada curso da doenca carrega
    //    ~12,6% de morte (MORTALITY em DEATH_TIME sorteios) e os agentes
    //    reciclam indefinidamente, entao em horizonte longo DEAD e o unico
    //    estado absorvente e quase todo mundo acaba nele: com 5.000 agentes em
    //    400 turnos, 89% morrem. Tirar RECOVERED da condicao acima da imunidade
    //    permanente e derruba esse numero.
    if (infectedAtNode[positions[id]] > 0)
    {
        states[id] = States::EXPOSED;
        counters[id] = 0;
    }
}

static void infection(int id, vector<States>& states, vector<unsigned char>& counters, AgentRng& rng)
{
    // O contador mede turnos NO ESTADO ATUAL e zera em toda transicao:
    //
    //   EXPOSED   ate EXPOSED_TIME turnos, sorteando INFECTION_RATE em cada um
    //             deles. Quem tira o numero cedo adoece em poucos turnos, quem
    //             tira tarde leva quase a janela inteira, e quem nunca tira
    //             volta a SUSCEPTIBLE. A incubacao passa a ter duracao
    //             variavel, que e o comportamento real da doenca: nem todo
    //             contato vira caso, e os que viram nao viram todos no mesmo
    //             dia.
    //
    //   INFECTED  DEATH_TIME turnos, sorteando MORTALITY em cada um deles.
    //             Sobreviveu aos tres -> RECOVERED.
    //
    // Os dois relogios sao independentes, e e por isso que o contador zera na
    // transicao: a duracao da fase infecciosa nao pode depender de quanto tempo
    // o agente levou para incubar. Sem o zeramento, quem incubasse em 1 turno
    // ficaria infeccioso por 9 e quem incubasse em 7 ficaria por 3 - o inverso
    // do pretendido.
    //
    // Com a janela inteira, a chance de uma exposicao virar doenca e
    // 1 - (1 - INFECTION_RATE)^EXPOSED_TIME, e nao INFECTION_RATE.
    //
    // O contador cabe num byte porque nunca passa de EXPOSED_TIME (7): zera em
    // toda transicao e as duas janelas fecham em 7 e em 3.
    //
    // REAVALIAR: o contato em checkSurroundings ainda expoe com probabilidade
    // 1; toda a estocasticidade da transmissao esta concentrada aqui. Ver a
    // nota gemea la.
    unsigned char counter = (unsigned char) (counters[id] + 1);

    // Sorteio como inteiro de 32 bits, comparado contra o limiar equivalente a
    // taxa. Sem double no laco: ver a nota em constants.h sobre reproduzir o
    // mesmo resultado em CPU e GPU.
    uint32_t probability = rng.next();

    if (states[id] == States::EXPOSED)
    {
        if (probability < Constants::INFECTION_THRESHOLD)
        {
            states[id] = States::INFECTED;
            counters[id] = 0;
            return;
        }
        if (counter == Constants::EXPOSED_TIME)
        {
            // A janela fechou sem a infeccao pegar: o agente volta a ser
            // suscetivel e pode ser exposto de novo num turno seguinte.
            states[id] = States::SUSCEPTIBLE;
            counters[id] = 0;
            return;
        }
        counters[id] = counter;
        return;
    }

    // DEAD e terminal, e o return aqui e o que impede a ressurreicao: sem ele,
    // quem morre no ultimo turno da janela cairia no bloco de RECOVERED logo
    // abaixo e voltaria a vida.
    if (probability < Constants::MORTALITY_THRESHOLD)
    {
        states[id] = States::DEAD;
        counters[id] = 0;
        return;
    }

    if (counter == Constants::DEATH_TIME)
    {
        states[id] = States::RECOVERED;
        counters[id] = 0;
        return;
    }
    counters[id] = counter;
}

void agentTurn(int id, const vector<int>& neighbors, int numberOfNodes,
               const vector<unsigned char>& infectedAtNode,
               vector<int>& positions, vector<States>& states,
               vector<unsigned char>& counters, AgentRng& rng)
{
    if (states[id] == States::DEAD)
    {
        return;
    }

    updatePosition(id, neighbors, numberOfNodes, positions, rng);

    States previousState = states[id];
    checkSurroundings(id, infectedAtNode, positions, states, counters);
    if (states[id] != previousState)
    {
        return;
    }

    if (states[id] == States::EXPOSED || states[id] == States::INFECTED)
    {
        infection(id, states, counters, rng);
    }
}
