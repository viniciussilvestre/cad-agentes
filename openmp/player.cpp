#include "player.h"
#include "constants.h"
#include "game.h"

Player::Player(int id, int position, States state, Game* game)
{
    this->id = id;
    this->position = position;
    this->state = state;
    this->game = game;
    // O contador mede turnos no estado atual e zera em toda transicao, entao
    // vale zero tanto para quem nasce SUSCEPTIBLE quanto para quem nasce
    // INFECTED - este ultimo comeca a fase infecciosa do zero, como qualquer
    // outro que acabou de adoecer.
    infectionCounter = 0;
}

int Player::getPosition() const
{
    return position;
}

States Player::getState() const
{
    return state;
}

void Player::infection(mt19937& rng)
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
    // REAVALIAR: o contato em checkSurroundings ainda expoe com probabilidade
    // 1; toda a estocasticidade da transmissao esta concentrada aqui. Ver a
    // nota gemea la.
    infectionCounter++;

    double lower_bound = 0;
    double upper_bound = 100;
    uniform_real_distribution<double> uniform(lower_bound, upper_bound);
    double probability = uniform(rng);

    if (state == States::EXPOSED)
    {
        if (probability <= Constants::INFECTION_RATE)
        {
            state = States::INFECTED;
            infectionCounter = 0;
            game->agentsStatusVector[id] = state;
            return;
        }
        if (infectionCounter == Constants::EXPOSED_TIME)
        {
            // A janela fechou sem a infeccao pegar: o agente volta a ser
            // suscetivel e pode ser exposto de novo num turno seguinte.
            state = States::SUSCEPTIBLE;
            infectionCounter = 0;
            game->agentsStatusVector[id] = state;
        }
        return;
    }

    // DEAD e terminal, e o return aqui e o que impede a ressurreicao: sem ele,
    // quem morre no ultimo turno da janela cairia no bloco de RECOVERED logo
    // abaixo e voltaria a vida.
    if (probability <= Constants::MORTALITY)
    {
        state = States::DEAD;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
        return;
    }

    if (infectionCounter == Constants::DEATH_TIME)
    {
        state = States::RECOVERED;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
    }
}

void Player::commonRoutine(const vector<int>& neighbors, int numberOfNodes, const vector<int>& infectedAtNode, mt19937& rng)
{
    // DEAD e terminal: nao se move, nao transmite, nao transiciona.
    if (state == States::DEAD)
    {
        return;
    }

    updatePosition(neighbors, numberOfNodes, rng);

    // Cada agente executa no maximo uma transicao por turno: se checkSurroundings
    // acabou de expor o agente, ele nao avanca a incubacao no mesmo turno.
    States previousState = state;
    checkSurroundings(infectedAtNode);
    if (state != previousState)
    {
        return;
    }

    if (state == States::EXPOSED || state == States::INFECTED)
    {
        infection(rng);
    }
}

void Player::checkSurroundings(const vector<int>& infectedAtNode)
{
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
    if (infectedAtNode[position] > 0)
    {
        state = States::EXPOSED;
        infectionCounter = 0;
        game->agentsStatusVector[id] = state;
    }
}

void Player::updatePosition(const vector<int>& neighbors, int numberOfNodes, mt19937& rng)
{
    // Grau fixo: todo no tem exatamente SLOTS slots e os vazios apontam para o
    // proprio no. Sorteia-se um slot e pronto - sem laco de rejeicao, sem -1,
    // sem checagem de limites, sem branch.
    //
    // Layout SoA sobre os slots: neighbors[k * numberOfNodes + i]. Agentes que
    // sorteiem o mesmo k leem posicoes proximas na memoria.
    uniform_int_distribution<int> uniform(0, Constants::SLOTS - 1);
    int slot = uniform(rng);
    position = neighbors[slot * numberOfNodes + position];
    game->agentsPositionVector[id] = position;
}
