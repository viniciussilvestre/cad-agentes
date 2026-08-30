# Medicoes

Registro dos numeros medidos, com a configuracao de cada um. Serve para nao
repetir experimento e para nao decidir por intuicao o que ja foi medido.

Regra: nenhum numero aqui e estimativa, salvo onde estiver escrito
**estimado**.

## Maquinas

| apelido | descricao |
|---|---|
| **local** | 14 nucleos / 28 threads, 1 socket. Linux, g++, glibc 2.43. GPU Quadro P6000 (sm_61). |
| **servidor** | 2 sockets x 32 nucleos / 64 threads = 64 nucleos / 128 threads. 2x Tesla V100. |
| **mac** | Apple Silicon, versao Metal. |

Configuracao padrao das medicoes locais, salvo indicacao: `map.txt`,
`--seed 42`.

---

## 1. Bandeira em vez de contagem em infectedAtNode

`infectedAtNode[pos]++` virou `infectedAtNode[pos] = 1`, porque o unico
consumidor testa `> 0`. Isso liberou a passada para ser paralela: varias
threads escrevem o mesmo valor, a corrida e inofensiva e nao precisa de
atomico nem de merge de `touchedNodes` por thread.

**local**, 10M agentes, 20 turnos:

| threads | antes | depois |
|---|---|---|
| 1 | 6,537 s | 6,427 s |
| 14 | 1,533 s | 0,775 s |
| 28 | 1,296 s | 0,598 s |

Escalabilidade contra o proprio serial: **5,0x -> 10,8x**.

Saida conferida identica a da versao sequencial em 1, 2, 7, 14 e 28 threads.

## 2. SoA e fim do ponteiro de volta

`vector<Player>` de 32 bytes com `Game*` dentro, mais a duplicacao entre
`agent.state` e `agentsStatusVector[id]`, viraram tres vetores paralelos de 6
bytes por agente.

**local**, 10M agentes, 20 turnos:

| threads | original | fix 1 | fix 2 |
|---|---|---|---|
| 1 | 6,576 s | 6,682 s | 6,076 s |
| 14 | 1,512 s | 0,664 s | 0,584 s |
| 28 | 1,310 s | 0,565 s | 0,419 s |

Escalabilidade contra o proprio serial: **5,0x -> 14,5x**.

Pico de memoria (RSS) com 10,5M agentes: **723 MB -> 185 MB**.

## 3. Comparacao honesta: sequencial com o mesmo modelo de dados

As duas versoes tinham divergido - `openmp/` com as mudancas, `base/` sem.
Depois de aplicar as mesmas mudancas em `base/`, as duas passam a diferir so
pelo OpenMP.

**local**, 10M agentes, 20 turnos:

| | tempo |
|---|---|
| sequencial, antes | 6,521 s |
| sequencial, depois | 5,923 s |
| omp 1 thread | 5,941 s |
| omp 14 | 0,529 s |
| omp 28 | 0,406 s |

Ganho do paralelismo, mesmo codigo dos dois lados: **14,6x**. Antes das
mudancas a mesma medida dava 4,98x - inflada, porque comparava contra um
sequencial que carregava peso que a versao paralela ja tinha largado.

O que as mudancas de modelo de dados rendem sozinhas, no sequencial: **1,10x**.
O resto so cobrava quando havia threads disputando.

## 4. Escala no servidor (100M agentes)

Antes das mudancas:

| configuracao | tempo |
|---|---|
| 1 thread | 690 s |
| 32 nucleos (1 socket) | 216 s |
| 128 threads (2 sockets) | 310 s |

O segundo socket **piorava** o tempo. Fracao serial derivada por Amdahl a
partir dos dois ultimos pontos: cerca de 44% - o `buildInfectedAtNode` serial.

Depois das mudancas:

| nucleos | tempo | vs 8 | ideal | eficiencia |
|---|---|---|---|---|
| 8 | 64 s | 1,00x | 1x | - |
| 16 | 33 s | 1,94x | 2x | 97% |
| 32 | 18 s | 3,56x | 4x | 89% |
| 64 | 13 s | 4,92x | 8x | 62% |

32 cores passou de 216 s para **17 s** (medicao separada), ou seja **12,7x** na
mesma maquina, e **40x** contra o sequencial original.

Leitura: 8, 16 e 32 cabem em um socket. A queda de eficiencia de 89% para 62%
acontece no unico passo que cruza para o segundo socket. O segundo socket
deixou de custar tempo e passou a render 1,38x - ainda longe de 2x, porque o
first-touch continua serial e todas as paginas nascem no socket 0.

## 5. Metal

**mac**, mesma configuracao de 100M agentes: cerca de **10 s**.

Uma GPU integrada abaixo de um servidor de 64 nucleos com 2 sockets. E o
formato esperado para uma carga com dois acessos aleatorios por agente por
turno e quase nenhuma aritmetica: o servidor tem mais nucleos, a GPU tem mais
requisicoes de memoria em voo.

## 6. Multiplos ambientes: onde colocar o OpenMP

Com `Simulation` rodando N jogos, ha dois lacos: os ambientes por fora, os
agentes de cada um por dentro. Tres variantes medidas.

**local**, 28 threads, 8,4M agentes no total (divididos entre os jogos),
20 turnos:

| jogos | so jogos | aninhado | sem omp no laco interno |
|---|---|---|---|
| 1 | 0,363 s | 0,345 s | **4,977 s** |
| 2 | 3,029 s | **0,608 s** | 2,889 s |
| 8 | 1,425 s | **0,731 s** | 1,384 s |
| 28 | 0,869 s | 1,383 s | **0,783 s** |

- **so jogos**: `parallel for` nos dois lacos, mas com um nivel ativo - o
  interno e serializado pelo OpenMP.
- **aninhado**: `omp_set_max_active_levels(2)`.
- **sem omp no laco interno**: o pragma interno nao existe.

Tres conclusoes:

1. Sem paralelismo no laco interno, `--games 1` fica **completamente serial**
   (13,7x pior). Nao ha o que dividir no laco de fora.
2. A partir de 2 jogos, tirar o pragma interno e 3% a 11% **mais rapido** que
   deixa-lo com o aninhamento desligado: uma regiao paralela inativa ainda
   custa entrada e saida a cada turno de cada jogo.
3. A escolha entre aninhar e nao aninhar vira em `jogos == threads`. Com
   poucos jogos o laco de fora deixa threads paradas (2 jogos usam 2 de 28
   threads); com jogos demais o aninhamento cria 28x28 = 784 threads
   disputando 28 nucleos.

O aninhamento e uma opcao explicita (`--nested`), como `--threads`: a tabela
acima diz qual escolher, mas quem escolhe e quem roda.

### Armadilha do libgomp: regiao paralela com `if` falso

Primeira versao do laco de jogos usava `#pragma omp parallel for if(N > 1)`.
Com um jogo so, a clausula falsa **nao evita** que a regiao exista - ela e so
inativa. As regioes de dentro de `Game::step` passam a ser aninhadas, e o
libgomp so reaproveita o pool de threads quente no nivel de cima: aninhadas,
elas voltam a criar e destruir threads de verdade a cada turno.

Efeito medido: `runTests_omp` (500.000 turnos com 1 agente) saiu de segundos
para **mais de 90 s** com 28 threads.

Correcao: desviar em C++ antes do pragma, e nao com clausula `if`.

## 7. Viagem entre jogos

Lista de presentes de tamanho fixo por jogo:

    capacidade = agentes + agentes / 2

Nunca cresce. Um `push_back` de um agente chegando reallocaria o vetor debaixo
do laco que outro jogo esta percorrendo, e isso nao e corrida inofensiva - e
leitura de memoria liberada. Com capacidade fixa nada muda de lugar; o pior
caso e uma viagem recusada por lotacao.

Conferido com 42.000 agentes em 4 jogos, 60 turnos: nenhum agente perdido nem
duplicado, com 1, 4 e 28 threads. As populacoes oscilam em torno de 10.500 por
jogo e a soma fica exatamente em 42.000 em todo turno.

### Turno duplo: a consequencia que a viagem traz

Cada jogo le o tamanho da propria lista UMA vez, antes do laco. Quem chega
depois disso so e processado no turno seguinte. Mas se o destino ainda **nao**
andou neste turno, a chegada entra no limite dele e o agente faz um segundo
turno completo - anda de novo e avanca a doenca de novo, com o mesmo sorteio,
porque o RNG e indexado por (id, turno).

Medido com 51.000 agentes, 100 turnos:

| jogos | viagens | turnos duplos | |
|---|---|---|---|
| 2 | 34.667 | 17.230 | 49,7% das viagens |
| 4 | 34.838 | 17.245 | 49,5% |
| 8 | 35.409 | 17.497 | 49,4% |

Sempre metade, e faz sentido: no sequencial os jogos andam em ordem, entao
viajar "para frente" cai num jogo que ainda nao andou e viajar "para tras" nao.

Sao 0,34% dos turnos-agente. O travelLock impede viajar duas vezes, mas nao
impede o segundo turno.

**Decidido manter.** Chegar num ambiente exige processar o estado do agente
ali de qualquer forma: o segundo turno nao e trabalho perdido, e a entrada do
agente no jogo novo.

Um detalhe fica em aberto e vale saber: o segundo turno reusa os MESMOS
sorteios do primeiro, porque o RNG e indexado por (id, turno) e a semente e
remontada igual no destino. O agente sorteia o mesmo slot de passeio e compara
o mesmo numero contra o limiar da transicao - nao sao dois turnos
independentes, sao um turno repetido. Se em algum momento isso incomodar,
basta deslocar o indice do fluxo quando travelLock > 0, que e como se sabe que
aquele processamento e de chegada.

### Determinismo

| | |
|---|---|
| 1 jogo | sequencial == OpenMP, com qualquer numero de threads |
| N jogos, 1 thread | sequencial == OpenMP |
| N jogos, N threads | **diverge** |

Com mais de um jogo e mais de uma thread, quem ja andou quando a chegada
acontece depende do escalonamento, entao quais agentes fazem turno duplo muda
de execucao para execucao. A saida continua conservando os agentes, mas deixa
de dar para testar por diff contra a sequencial.

Isso foi uma escolha explicita. Para desfazer bastaria um byte por agente com
a marca do ultimo turno processado: pular quem ja andou no turno corrente
elimina o turno duplo e devolve o determinismo, ao custo de 100 MB com 100
milhoes de agentes.

---

## Ambiente CUDA (bloqueios conhecidos)

Nao sao medicoes, sao impedimentos que ja custaram tempo:

- **local**: nvcc 13.1 nao compila nenhum `.cu`. glibc 2.43 declara `rsqrt`
  com `noexcept` e o `crt/math_functions.h` do CUDA declara sem, o que da
  `error: exception specification is incompatible` ate num arquivo vazio.
- **local**: CUDA 13 removeu Pascal. A P6000 e sm_61 e o nvcc recusa
  (`Unsupported gpu architecture 'sm_61'`).
- **servidor**: a V100 e sm_70, tambem removida no CUDA 13. Conferir com
  `nvcc --list-gpu-arch` antes de comecar; se `compute_70` nao aparecer, e
  preciso um toolkit CUDA 12.x.
