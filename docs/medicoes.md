# Medicoes

Registro dos numeros medidos, com a configuracao de cada um. Serve para nao
repetir experimento e para nao decidir por intuicao o que ja foi medido.

Regra: nenhum numero aqui e estimativa, salvo onde estiver escrito
**estimado**.

## Resultado final

Configuracao de producao: 1,222 bilhao de agentes, 600 turnos, 20 ambientes de
30 milhoes de nos cada - 0,73 trilhao de agente-turnos.

| versao | tempo | vazao | contra 1 thread |
|---|---|---|---|
| sequencial | 60 h (2,5 dias) | 3,4 M/s | 1x |
| OpenMP, `nested=false` | 3 h 02 | 67,1 M/s | **20x** |
| CUDA, Quadro P6000 | **8 min** | 1.527 M/s | **450x** |

CUDA contra OpenMP: **22,8x**. A placa e de 2016.

O caminho ate aqui nao foi a GPU: foi tirar o que atrapalhava. Bandeira em vez
de contagem (secao 1), SoA sem ponteiro de volta (secao 2), carga de mapa 10x
menor (secao 8) - cada uma dessas valia pouco sozinha no sequencial e muito no
paralelo, e sem elas a versao CUDA teria batido nos mesmos limites.

### Margem de memoria: quase nenhuma

| | |
|---|---|
| agentes (8 bytes cada) | 9,1 GiB |
| mapas (6 slots x 4 bytes por no) | 13,5 GiB |
| **total** | **22,6 de 24 GiB** |

94% da placa. Um jogo a mais (+0,7 GiB) ainda entra; dois, nao. Aumentar
agentes custa 8 bytes cada - mais 100 milhoes de agentes sao 0,8 GiB.

Quem paga a conta e o mapa: 726 MB por ambiente, e sao 20. Se a escala precisar
crescer, o caminho e compartilhar o grafo entre jogos que usam o mesmo arquivo,
ou aceitar mapas menores - e nao economizar no agente, que ja esta em 8 bytes.

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

## 5b. MPI contra OpenMP no laco de jogos

A pergunta era: um ou outro, quanto custa trocar memoria compartilhada por
troca de mensagem no MESMO laco. A comparacao estrutural certa e contra
`GameC_omp_games`, que tambem paraleliza os ambientes - `GameC_omp` paraleliza
os agentes de cada um, que e outra decomposicao.

**mac** (10 nucleos logicos), 8 jogos, 880 mil agentes, 100 turnos, melhor de 3:

| unidades | omp-jogos | mpi | mpi / omp |
|---|---|---|---|
| 1 | 3,242 s | 3,237 s | 1,00 |
| 2 | 1,802 s | 1,874 s | 1,04 |
| 4 | 1,002 s | 1,020 s | 1,02 |
| 8 | 0,602 s | 0,637 s | 1,06 |

Sequencial de referencia: 3,191 s.

**A mensagem custa de 2% a 6%.** Muito menos que no gerador (secao 8, onde o
MPI perdia 79%), e a razao e o que se comunica: com TRAVEL_RATE de 1%, sao
~8.800 viajantes por turno a 16 bytes = 141 KB por turno em toda a malha, um
`Alltoallv` so. Em 100 turnos, 14 MB no total. O `mpirun` paga criacao de
processo uma vez e amortiza ao longo da corrida, enquanto no gerador a corrida
inteira durava segundos.

### O que o MPI ganha de quebra

**Determinismo.** A secao 7 registra que a versao OpenMP com varios jogos e
varias threads diverge da sequencial, porque quem faz turno duplo depende do
escalonamento. No MPI isso nao existe: as chegadas sao aplicadas na barreira
entre turnos, entao nao ha "jogo que ainda nao andou". Conferido - `-n 1`,
`-n 2`, `-n 4` e `-n 8` dao saida **identica**, e com `--games 1` a saida bate
byte a byte com a do `GameC`.

O preco disso e que a versao MPI difere da sequencial quando ha varios jogos,
pela ausencia do turno duplo - do mesmo jeito que a Metal e a CUDA.

**Memoria de mapa.** Cada rank carrega so os mapas dos jogos que possui. E o
motivo real para usar MPI aqui: com 28 jogos e 4 nos, cada maquina segura sete
grafos em vez de 28.

### Teto

O paralelismo do MPI para no numero de jogos: o laco de agentes de um jogo nao
e dividido entre ranks. Dividi-lo exigiria replicar `infectedAtNode` e reduzi-lo
todo turno - com mapas de 1M nos sao 125 KB por jogo por turno como bitmask,
tolerave; com os de 100M nos sao 12,5 MB, e ai o coletivo passa a custar mais
que o trabalho que economiza.

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

## 8. Geracao de varios grafos: MPI ou OpenMP

Os N grafos nao compartilham um byte, entao a geracao divide sem nenhuma
sincronizacao. Duas versoes do mesmo fonte:

    ./genGraph --graphs 8                     threads OpenMP
    mpirun -n 8 ./genGraph_mpi                um processo por grafo

Os arquivos saem **byte a byte identicos** nas duas, porque a semente de cada
grafo e `seed + indice`, e nao depende de quem gerou nem da ordem.

**local**, 4 grafos de 600x600 (360.000 nos cada):

| | tempo |
|---|---|
| serial (1 thread) | 2,77 s |
| OpenMP, 4 threads | 0,90 s |
| MPI, `-n 4` | 1,61 s |

Numa maquina so o OpenMP ganha: o `mpirun` paga criacao de processo e nao ha
memoria a ganhar com espacos de endereco separados. O MPI vale quando a
geracao passa de uma maquina - ai a mesma linha de comando distribui entre
nos, o que o OpenMP nao faz.

Com mais grafos do que processos, a divisao e circular (`grafo % ranks`), e
cada processo gera varios. Conferido: `mpirun -n 4 --graphs 8` produz os oito
arquivos.

### Memoria: o que cabe e o que nao cabe

Custo por no, medido:

| | bytes/no |
|---|---|
| antes (vector<vector<int>> + unordered_set) | 170 |
| depois (bloco plano de 6 slots, sem tabela hash) | 51 |

O que sumiu: um `vector<int>` por no custa 24 bytes do objeto mais um bloco no
heap por no; e a tabela hash de arestas guardava ~2 entradas por no a ~40 bytes
cada, so para responder "esta aresta ja existe?". Com grau limitado a 6, essa
pergunta se responde varrendo os vizinhos do proprio no - 6 comparacoes
contiguas, sem estrutura nenhuma. Saida byte a byte identica.

Quanto custa cada grafo, com 51 bytes/no:

| grade | nos | memoria por grafo |
|---|---|---|
| 1000x1000 | 1 M | 51 MB |
| 3000x3000 | 9 M | 460 MB |
| 10000x10000 (padrao) | 100 M | **5,1 GB** |

**O numero de grafos nao e o numero de threads.** `--graphs 64` com o padrao
10000x10000 pede 64 x 5,1 GB = 326 GB se os 64 saem ao mesmo tempo. Quem
controla quantos saem juntos e `OMP_NUM_THREADS`:

    OMP_NUM_THREADS=4 ./genGraph --graphs 64     64 grafos, 4 de cada vez

Antes desta mudanca o mesmo comando pediria 17 GB por grafo, e 64 em paralelo
passavam de 1 TB - foi o que matou o job 182315 no PBS.

### Tamanho do arquivo

O arquivo e texto, entao o custo por no e dominado pelo numero de digitos dos
indices - cresce com a escala. Medido:

| grade | com coluna de peso | sem | |
|---|---|---|---|
| 500x500 | 12,6 MB | 9,5 MB | 39,9 bytes/no |
| 1000x1000 | 52,3 MB | 39,6 MB | 41,5 bytes/no |
| 2000x2000 | 226,4 MB | 175,7 MB | 46,0 bytes/no |

A terceira coluna das arestas era a distancia euclidiana entre os dois nos, e
nenhum leitor a usava: os tres jogos leem `values[0]` e `values[1]` e
descartam o resto. Sao 22% a 25% do arquivo. Conferido que a simulacao da
exatamente o mesmo resultado lendo o formato de duas colunas e o de tres.

Projecao para o padrao 10000x10000 (100 milhoes de nos, ~200 milhoes de
arestas): cerca de **5,5 GB por arquivo**, contra ~6,9 GB antes. Com
`--graphs 64` sao ~350 GB de disco - vale conferir a cota antes de disparar.

### Onde o tempo do gerador vai

Perfilado com 2000x2000 (4 milhoes de nos), antes e depois de trocar o
escritor:

| fase | antes | depois |
|---|---|---|
| 1 coords/regiao | 0,18 s | 0,19 s |
| 2 arestas de grade | 0,08 s | 0,08 s |
| 3 atalhos | 0,09 s | 0,09 s |
| 4 escrita do arquivo | 1,69 s | **0,33 s** |
| 5 relatorio/BFS | 0,57 s | 0,59 s (opcional) |

Construir o grafo eram 0,35 s de 2,61 s - 13%. **Era a escrita que dominava**,
e formatar inteiro pelo ostream era quase toda ela: a mao, o mesmo conteudo sai
5,1x mais rapido.

Isso decide o que vale acelerar. Com a escrita custando 65%, levar a geracao
para a GPU dava no maximo 1,15x por Amdahl, mesmo que a geracao custasse zero.
Com o escritor novo e `--no-report`, a geracao passa a ser mais da metade do
que sobra, e ai a GPU tem o que ganhar.

Tempo de parede, 2000x2000: 3,00 s -> 2,26 s -> 1,37 s com `--no-report`.

## 9. genGraph_cuda

Uma thread por no nas tres fases de construcao. A emissao do arquivo continua
na CPU - GPU nao escreve arquivo.

Duas diferencas obrigatorias em relacao a versao de CPU:

- **RNG.** O mt19937 e sequencial e carrega 2,5 KB de estado. Foi trocado pelo
  splitmix64 indexado por (semente, no, indice), o mesmo do jogo. Por isso
  `genGraph_cuda` **nao produz os mesmos arquivos** que `genGraph` com a mesma
  semente: mesmas propriedades, grafos diferentes.

- **Atalhos.** Na CPU os nos sao processados em ordem e cada aresta criada muda
  o grau visto pelos proximos. Em paralelo a disputa vira rodadas: cada destino
  fica com a proposta do menor id que o quer (atomicMin, independe da ordem das
  threads), e quem recebe nao envia na mesma rodada. O grau sobe no maximo 1
  por no por rodada, entao nunca estoura o teto sem precisar reservar as duas
  pontas de uma aresta de forma atomica.

  A decisao fica em kernel separado da aplicacao. Juntos, uma thread escreveria
  degree[i] enquanto outra o le para avaliar o predicado do emissor, e as duas
  pontas da mesma aresta poderiam discordar.

### Desempenho: a GPU perde

Um grafo por vez, na Quadro P6000:

| grade | CPU 1 thread | GPU |
|---|---|---|
| 1000x1000 | 0,22 s | 0,89 s |
| 2000x2000 | 0,73 s | 1,23 s |
| 3000x3000 | 1,60 s | 1,82 s |
| 4000x4000 | 2,77 s | 2,66 s |

Oito grafos de 1500x1500, que e o caso de uso real:

| | tempo |
|---|---|
| CPU 1 thread | 2,96 s |
| CPU 8 threads | **0,50 s** |
| CPU 28 threads | 0,49 s |
| GPU | 2,83 s |

**5,7x mais lenta que a versao OpenMP.** Tres motivos, todos estruturais:

1. **Contexto CUDA custa 0,84 s fixos.** Medido com uma grade de 100x100, onde
   nao ha trabalho nenhum a fazer. Com um grafo so, e mais que a geracao
   inteira na CPU.

2. **A escrita continua na CPU** e e a maior parte do tempo. GPU nao escreve
   arquivo, e nenhum kernel muda isso.

3. **A versao CUDA faz um grafo por vez**, enquanto a OpenMP faz N em paralelo.
   Como os grafos sao independentes, a CPU tem N vezes mais paralelismo
   disponivel do que a GPU consegue usar num grafo so.

Era o que o perfil ja dizia: com a geracao valendo 13% a 28% do tempo, nao ha
o que ganhar. A versao existe, esta correta e e deterministica - mas para
gerar mapas, o caminho e `./genGraph --graphs N`.

Onde ela poderia virar: grafos muito maiores (a A100 tem banda e L2 muito
maiores que a P6000), ou sobrepor a escrita de um grafo com os kernels do
proximo.

### Estado da verificacao

Compila e roda na P6000 com CUDA 12.9, `-ccbin g++-14` e o cabecalho de
compatibilidade `tools/cuda_compat` (ver adiante). Com 300x300:

| verificacao | resultado |
|---|---|
| cabecalho x conteudo | 90.000 nos e 180.851 arestas, batem |
| teto de grau | maximo 6, respeitado |
| simetria da adjacencia | as duas pontas se veem |
| conectividade | 90.000 de 90.000 alcancados |
| regioes | 10 regioes de 9.000 nos |
| determinismo | duas execucoes na GPU dao o mesmo arquivo |
| semente diferente | da grafo diferente |
| leitura pelo jogo | mapa carrega e simula normalmente |

E o teste que fecha o desenho: a saida da GPU e **byte a byte igual** a do
mesmo algoritmo rodado serialmente (os lacos sao grid-stride, entao com
gridDim=1/blockDim=1 uma thread percorre tudo, e o mesmo fonte compila como C++
contra um cuda_runtime.h de mentira). Ou seja, a eleicao por rodadas realmente
nao depende da ordem em que as threads chegam - que era o ponto onde a versao
paralela poderia divergir.

---

## Ambiente CUDA (bloqueios conhecidos)

Nao sao medicoes, sao impedimentos que ja custaram tempo:

**RESOLVIDO** por `tools/cuda_compat/math.h` - ver o fim desta secao. O que
segue e o diagnostico, para nao se repetir a investigacao.

Sem esse cabecalho, esta maquina **nao compila nenhum `.cu`**, com nenhum dos
dois toolkits instalados. Vale ate para um arquivo com um kernel vazio e zero
includes - nao e problema do nosso codigo.

A causa e a glibc 2.43, mais nova que os dois toolkits. Ela declara um punhado
de funcoes de C23 (`cospi`, `sinpi`, `rsqrt`) com `noexcept`, e o
`crt/math_functions.h` do CUDA declara as mesmas sem, o que da
`error: exception specification is incompatible`. O `cuda_runtime.h` e
pre-incluido em todo `.cu`, entao nao ha como escapar.

| toolkit | situacao |
|---|---|
| CUDA 13.1 | guarda `sinpi`/`cospi`, mas nao `rsqrt` - falha. E removeu Pascal e Volta, entao nem sm_61 nem sm_70 |
| CUDA 12.9 | nao tem guarda nenhuma - falha nas tres. Cobre sm_50 ate sm_90 |

Tentado e sem efeito: `-allow-unsupported-compiler`, `-U_GNU_SOURCE`,
`-D__STDC_WANT_IEC_60559_FUNCS_EXT__`, e um cabecalho de `-include` que
neutraliza `__USE_GNU` antes do `math.h` (chega tarde: o `cuda_runtime.h` ja
foi pre-incluido).

O `-ccbin g++-14` resolve **outro** problema, esse sim real: o CUDA 12.9 aceita
GCC ate 14 e a maquina tem g++-15 como padrao. E necessario, mas nao
suficiente.

### O que resolveu

Fazer a glibc nao declarar as tres. A guarda em `bits/libc-header-start.h` e

    __GLIBC_USE (IEC_60559_FUNCS_EXT) || __GLIBC_USE (ISOC23)

e as **duas** pontas precisam cair. A primeira sai desligando `__USE_GNU`; a
segunda nao vem junto, porque `_GNU_SOURCE` tambem define `_ISOC23_SOURCE` e o
`features.h` tem guarda de inclusao - desligar `_GNU_SOURCE` depois nao refaz a
conta. Por isso `__GLIBC_USE_ISOC23` e forcado a 0 na mao.

`tools/cuda_compat/math.h` entra na frente do `<math.h>` do sistema por `-I`,
desliga as duas pontas, chama o de verdade com `include_next` e restaura os
macros. Em glibc anterior a 2.41 nao muda nada.

Com isso, `make genGraph_cuda` compila e roda na P6000.

Para o servidor: a A100 e sm_80, coberta pelo CUDA 12.x.
`make genGraph_cuda CUDA_ARCH=sm_80`.
