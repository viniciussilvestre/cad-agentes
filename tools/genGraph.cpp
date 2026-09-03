// genGraph - gerador de mapas de mundo pequeno para o PlagueGame.
//
// Emite uma grade width x height com atalhos de longo alcance de Kleinberg
// (destino com probabilidade proporcional a 1/d^2, d = distancia de Manhattan
// em celulas), respeitando um teto de grau de 6 arestas por no.
//
// Gera N grafos de uma vez, um arquivo por grafo:
//
//   map_0.txt, map_1.txt, ...
//
// Os grafos sao independentes entre si - nenhum dado e compartilhado - entao a
// geracao e dividida entre processos ou threads sem nenhuma sincronizacao:
//
//   mpirun -n 8 ./genGraph_mpi --graphs 8    um grafo por rank
//   ./genGraph --graphs 8                    um grafo por thread OpenMP
//
// A semente de cada grafo e seed + indice do grafo. Sem isso os N grafos
// sairiam identicos, que e o oposto do que ter varios ambientes serve.
//
// Formato de saida:
//   <nNos> <nArestas> <nRegioes>
//   <linha em branco>
//   <x> <y> <regiao>     x nNos
//   <linha em branco>
//   <origem> <destino>   x nArestas
//
// A aresta nao carrega peso. Ele existia como terceira coluna, era a distancia
// euclidiana entre os dois nos, e nenhum leitor usava: os tres jogos leem
// values[0] e values[1] e descartam o resto. Eram ~7 bytes por aresta, 20% do
// arquivo - com 100 milhoes de nos, mais de 1 GB por grafo escrito para nada.
// Arquivos antigos, com a coluna, continuam sendo lidos: o parser exige duas
// colunas e ignora o que vier depois.
//
// Cada aresta nao-direcionada aparece uma unica vez, com origem < destino, e
// as arestas saem ordenadas por origem crescente e contiguas por origem.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#endif

using namespace std;

// Teto de grau. Igual ao PlagueGame::SLOTS: 4 vizinhos de grade + 2 atalhos.
static const int MAX_DEGREE = 6;
// Slots de atalho de saida por no. O no pode receber atalhos de outros, e como
// a aresta e nao-direcionada o receptor tambem consome um slot - por isso o
// teto de grau e verificado nos dois extremos.
static const int SHORTCUT_SLOTS = 2;
// Tentativas de sortear um destino que caia dentro da grade.
static const int MAX_ATTEMPTS = 30;

struct GraphConfig
{
    int width = 1000;
    int height = 1000;
    double shortcuts = 0.01;
    int regionsX = 5;
    int regionsY = 2;
    int spacing = 100;
    int jitter = 40;
    unsigned int seed = 42;

    // A conectividade e o diametro saem de duas varreduras BFS sobre o grafo
    // inteiro - 22% do tempo com 2000x2000. Sao verificacao, nao producao:
    // desligar so tira as duas ultimas linhas do relatorio.
    bool report = true;
};

// Numa grade o numero de nos a distancia exata d e aproximadamente 4d, entao
// P(d) proporcional a 4d * (1/d^2) = 1/d. Amostrar d desta cumulativa e depois
// sortear um ponto do losango a distancia d evita a amostragem por rejeicao
// sobre nos uniformes, que numa grade 200x200 rejeitaria praticamente tudo
// (1/d^2 ~ 6e-5 na distancia tipica) e so produziria atalhos curtos.
static vector<double> buildDistanceCumulative(int maxDistance)
{
    vector<double> cumulative(maxDistance, 0.0);
    double total = 0.0;
    for (int d = 1; d <= maxDistance; d++)
    {
        total += 1.0 / (double) d;
        cumulative[d - 1] = total;
    }
    return cumulative;
}

static int sampleDistance(const vector<double>& cumulative, mt19937& rng)
{
    uniform_real_distribution<double> uniform(0.0, cumulative.back());
    double target = uniform(rng);
    int index = (int) (upper_bound(cumulative.begin(), cumulative.end(), target) - cumulative.begin());
    if (index >= (int) cumulative.size())
    {
        index = (int) cumulative.size() - 1;
    }
    return index + 1;
}

// A vizinhanca e um bloco plano de MAX_DEGREE slots por no, e nao um
// vector<vector<int>>.
//
// Nao e arrumacao: e o que torna grades grandes viaveis. Um vector por no custa
// 24 bytes do objeto mais um bloco no heap por no (cabecalho do malloc, o mesmo
// arredondamento e as realocacoes do crescimento) - da ordem de 70 bytes por no
// para guardar 4 inteiros. Plano, sao 24 bytes fixos, e os vizinhos de um no
// ficam na mesma linha de cache.
static inline int neighborSlot(int node, int slot)
{
    return node * MAX_DEGREE + slot;
}

static bool hasEdge(const vector<int>& adjacency, const vector<int>& degree, int from, int to)
{
    // O grau e limitado a MAX_DEGREE, entao varrer os vizinhos custa no maximo
    // 6 comparacoes contiguas - mais barato que consultar uma tabela hash, e
    // sem a tabela inteira na memoria.
    for (int slot = 0; slot < degree[from]; slot++)
    {
        if (adjacency[neighborSlot(from, slot)] == to)
        {
            return true;
        }
    }
    return false;
}

static void addEdge(vector<int>& adjacency, vector<int>& degree, int from, int to)
{
    adjacency[neighborSlot(from, degree[from])] = to;
    degree[from]++;
    adjacency[neighborSlot(to, degree[to])] = from;
    degree[to]++;
}

static vector<int> bfsDistances(const vector<int>& adjacency, const vector<int>& degree, int source)
{
    vector<int> distance(degree.size(), -1);
    queue<int> pending;
    distance[source] = 0;
    pending.push(source);
    while (!pending.empty())
    {
        int node = pending.front();
        pending.pop();
        for (int slot = 0; slot < degree[node]; slot++)
        {
            int neighbor = adjacency[neighborSlot(node, slot)];
            if (distance[neighbor] == -1)
            {
                distance[neighbor] = distance[node] + 1;
                pending.push(neighbor);
            }
        }
    }
    return distance;
}

// ---------------------------------------------------------------------------
// Escrita do arquivo.
//
// Formatar inteiro em texto pelo ostream custa caro: com 2000x2000 a emissao
// era 1,69 s de um total de 2,61 s - 65% do programa, contra 0,35 s para
// construir o grafo inteiro. Convertendo a mao para um buffer grande e
// despejando com fwrite, cai para 0,30 s, com saida byte a byte identica.
//
// Nao e microtuning: era o gargalo, e sem isso qualquer aceleracao da geracao
// (GPU inclusive) esbarra em 1,15x por Amdahl.
// ---------------------------------------------------------------------------

// Coordenadas podem ser negativas: o jitter desloca para os dois lados, entao o
// no 0 sai em -40 com frequencia. Sem tratar o sinal aqui o arquivo sairia
// corrompido justamente nas bordas.
static inline char* writeInt(char* cursor, int value)
{
    if (value < 0)
    {
        *cursor++ = '-';
        value = -value;
    }
    if (value == 0)
    {
        *cursor++ = '0';
        return cursor;
    }
    char digits[12];
    int count = 0;
    while (value > 0)
    {
        digits[count++] = (char) ('0' + value % 10);
        value /= 10;
    }
    while (count > 0)
    {
        *cursor++ = digits[--count];
    }
    return cursor;
}

// Buffer de saida com despejo automatico. A margem existe para que uma linha
// inteira caiba sempre depois da conferencia, sem testar campo a campo.
class OutputBuffer
{
    public:
        static const size_t CAPACITY = 1u << 22;   // 4 MB
        static const size_t MARGIN = 64;           // maior linha possivel

        explicit OutputBuffer(FILE* target) : file(target), storage(CAPACITY + MARGIN)
        {
            cursor = storage.data();
        }

        void endLine()
        {
            *cursor++ = '\n';
            if ((size_t) (cursor - storage.data()) >= CAPACITY)
            {
                flush();
            }
        }

        void field(int value)
        {
            cursor = writeInt(cursor, value);
        }

        void separator()
        {
            *cursor++ = '\t';
        }

        void flush()
        {
            fwrite(storage.data(), 1, (size_t) (cursor - storage.data()), file);
            cursor = storage.data();
        }

    private:
        FILE* file;
        vector<char> storage;
        char* cursor;
};

static int farthestNode(const vector<int>& distance)
{
    int best = 0;
    int bestDistance = -1;
    for (int i = 0; i < (int) distance.size(); i++)
    {
        if (distance[i] > bestDistance)
        {
            bestDistance = distance[i];
            best = i;
        }
    }
    return best;
}

// Gera um grafo inteiro e escreve o arquivo. Nao imprime nada: o relatorio sai
// em report, para ser impresso de uma vez so por quem chamou.
//
// Toda a memoria e local - adjacencia, graus, conjunto de arestas. E o que
// permite rodar varias chamadas destas em paralelo sem nenhuma trava: dois
// grafos nao tem um byte em comum.
static bool generateGraph(const GraphConfig& config, const string& outputFile, string& report)
{
    ostringstream out;

    int numberOfNodes = config.width * config.height;
    int numberOfRegions = config.regionsX * config.regionsY;
    mt19937 rng(config.seed);

    // Coordenadas e regiao de cada no.
    vector<int> coordinateX(numberOfNodes);
    vector<int> coordinateY(numberOfNodes);
    vector<int> nodeRegion(numberOfNodes);
    uniform_int_distribution<int> jitterDistribution(-config.jitter, config.jitter);
    for (int i = 0; i < numberOfNodes; i++)
    {
        int row = i / config.width;
        int column = i % config.width;
        coordinateX[i] = column * config.spacing + jitterDistribution(rng);
        coordinateY[i] = row * config.spacing + jitterDistribution(rng);
        int regionColumn = (int) ((long long) column * config.regionsX / config.width);
        int regionRow = (int) ((long long) row * config.regionsY / config.height);
        nodeRegion[i] = regionRow * config.regionsX + regionColumn;
    }

    // Bloco plano de MAX_DEGREE slots por no. -1 marca slot livre.
    vector<int> adjacency((size_t) numberOfNodes * MAX_DEGREE, -1);
    vector<int> degree(numberOfNodes, 0);

    // Arestas de grade.
    for (int i = 0; i < numberOfNodes; i++)
    {
        if (i % config.width != config.width - 1)
        {
            addEdge(adjacency, degree, i, i + 1);
        }
        if (i + config.width < numberOfNodes)
        {
            addEdge(adjacency, degree, i, i + config.width);
        }
    }

    // Atalhos de longo alcance.
    int shortcutsCreated = 0;
    int shortcutsRejected = 0;
    int maxDistance = config.width + config.height - 2;
    if (maxDistance < 1)
    {
        maxDistance = 1;
    }
    vector<double> cumulative = buildDistanceCumulative(maxDistance);
    uniform_real_distribution<double> slotDraw(0.0, 1.0);
    uniform_int_distribution<int> signDraw(0, 1);

    for (int i = 0; i < numberOfNodes; i++)
    {
        for (int slot = 0; slot < SHORTCUT_SLOTS; slot++)
        {
            if (slotDraw(rng) >= config.shortcuts)
            {
                continue;
            }
            if (degree[i] >= MAX_DEGREE)
            {
                shortcutsRejected++;
                continue;
            }
            int row = i / config.width;
            int column = i % config.width;
            for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++)
            {
                int distance = sampleDistance(cumulative, rng);
                uniform_int_distribution<int> splitDraw(0, distance);
                int deltaX = splitDraw(rng);
                int deltaY = distance - deltaX;
                if (signDraw(rng) == 1) { deltaX = -deltaX; }
                if (signDraw(rng) == 1) { deltaY = -deltaY; }
                int targetColumn = column + deltaX;
                int targetRow = row + deltaY;
                if (targetColumn < 0 || targetColumn >= config.width || targetRow < 0 || targetRow >= config.height)
                {
                    continue;
                }
                int target = targetRow * config.width + targetColumn;
                if (target == i)
                {
                    continue;
                }
                if (hasEdge(adjacency, degree, i, target))
                {
                    continue;
                }
                // O teto de grau vale nos dois extremos: como a aresta e
                // nao-direcionada, o receptor tambem gasta um slot. Sem isso o
                // grau estoura (numa grade 200x200, shortcuts=0.5 chega a
                // milhares de arestas num unico no).
                if (degree[target] >= MAX_DEGREE)
                {
                    shortcutsRejected++;
                    break;
                }
                addEdge(adjacency, degree, i, target);
                shortcutsCreated++;
                break;
            }
        }
    }

    // Emissao: origem crescente, contigua por origem, origem < destino.
    FILE* output = fopen(outputFile.c_str(), "wb");
    if (output == nullptr)
    {
        report = "Erro: nao foi possivel escrever em '" + outputFile + "'.\n";
        return false;
    }
    OutputBuffer buffer(output);

    // Cada aresta gasta um slot nos dois extremos, entao a soma dos graus e o
    // dobro do numero de arestas. Antes esse numero saia do tamanho da tabela
    // hash, que era o unico motivo para ela sobreviver ate aqui.
    long long degreeTotal = 0;
    for (int value : degree)
    {
        degreeTotal += value;
    }
    int numberOfEdges = (int) (degreeTotal / 2);

    buffer.field(numberOfNodes);
    buffer.separator();
    buffer.field(numberOfEdges);
    buffer.separator();
    buffer.field(numberOfRegions);
    buffer.endLine();
    buffer.endLine();

    for (int i = 0; i < numberOfNodes; i++)
    {
        buffer.field(coordinateX[i]);
        buffer.separator();
        buffer.field(coordinateY[i]);
        buffer.separator();
        buffer.field(nodeRegion[i]);
        buffer.endLine();
    }
    buffer.endLine();

    // Os vizinhos de um no cabem em MAX_DEGREE, entao a ordenacao e sobre no
    // maximo 6 elementos num vetor de pilha - sem alocacao por no.
    for (int i = 0; i < numberOfNodes; i++)
    {
        int targets[MAX_DEGREE];
        int count = 0;
        for (int slot = 0; slot < degree[i]; slot++)
        {
            int neighbor = adjacency[neighborSlot(i, slot)];
            if (neighbor > i)
            {
                targets[count++] = neighbor;
            }
        }
        // Insercao direta: sao no maximo MAX_DEGREE elementos. std::sort aqui
        // custa mais em codigo do que o laco, e ainda faz o GCC avisar de
        // limites por causa do caminho de 16 elementos que nunca e tomado.
        for (int a = 1; a < count; a++)
        {
            int value = targets[a];
            int b = a - 1;
            while (b >= 0 && targets[b] > value)
            {
                targets[b + 1] = targets[b];
                b--;
            }
            targets[b + 1] = value;
        }
        for (int index = 0; index < count; index++)
        {
            buffer.field(i);
            buffer.separator();
            buffer.field(targets[index]);
            buffer.endLine();
        }
    }

    buffer.flush();
    fclose(output);

    // Relatorio.
    int minimumDegree = degree.empty() ? 0 : degree[0];
    int maximumDegree = degree.empty() ? 0 : degree[0];
    long long degreeSum = 0;
    for (int value : degree)
    {
        minimumDegree = min(minimumDegree, value);
        maximumDegree = max(maximumDegree, value);
        degreeSum += value;
    }

    out << "arquivo: " << outputFile << "\n";
    out << "grade: " << config.width << "x" << config.height << "  spacing=" << config.spacing
        << "  jitter=" << config.jitter << "  seed=" << config.seed << "\n";
    out << "nos: " << numberOfNodes << "\n";
    out << "arestas: " << numberOfEdges << "\n";
    out << "grau medio: " << fixed << setprecision(3)
        << (numberOfNodes > 0 ? (double) degreeSum / (double) numberOfNodes : 0.0) << "\n";
    out << "grau minimo: " << minimumDegree << "  grau maximo: " << maximumDegree << "\n";
    out << "atalhos criados: " << shortcutsCreated << "  rejeitados: " << shortcutsRejected << "\n";
    out << "regioes: " << numberOfRegions << " (" << config.regionsX << "x" << config.regionsY << ")\n";

    if (config.report)
    {
        vector<int> firstSweep = bfsDistances(adjacency, degree, 0);
        int reached = 0;
        for (int value : firstSweep)
        {
            if (value >= 0)
            {
                reached++;
            }
        }
        bool connected = (reached == numberOfNodes);
        int source = farthestNode(firstSweep);
        vector<int> secondSweep = bfsDistances(adjacency, degree, source);
        int diameter = secondSweep[farthestNode(secondSweep)];

        out << "conexo: " << (connected ? "sim" : "NAO") << "\n";
        out << "diametro estimado (BFS de varredura dupla): " << diameter << "\n";
        if (!connected)
        {
            out << "AVISO: o grafo saiu desconexo (" << reached << " de " << numberOfNodes
                << " nos alcancados a partir do no 0). O arquivo foi gerado mesmo assim.\n";
        }
    }

    report = out.str();
    return true;
}

int main(int argc, char** argv)
{
    int rank = 0;
    int worldSize = 1;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
#endif

    GraphConfig config;
    // Por padrao, um grafo por processo: com mpirun -n 8 saem 8 grafos sem
    // precisar repetir o numero na linha de comando. Sem MPI, um grafo.
    int numberOfGraphs = worldSize;
    string outputPrefix = "map_";

    for (int i = 1; i < argc; i++)
    {
        string flag = argv[i];
        bool hasValue = (i + 1 < argc);
        if (flag == "--width" && hasValue) { config.width = atoi(argv[++i]); }
        else if (flag == "--height" && hasValue) { config.height = atoi(argv[++i]); }
        else if (flag == "--shortcuts" && hasValue) { config.shortcuts = atof(argv[++i]); }
        else if (flag == "--regions-x" && hasValue) { config.regionsX = atoi(argv[++i]); }
        else if (flag == "--regions-y" && hasValue) { config.regionsY = atoi(argv[++i]); }
        else if (flag == "--spacing" && hasValue) { config.spacing = atoi(argv[++i]); }
        else if (flag == "--jitter" && hasValue) { config.jitter = atoi(argv[++i]); }
        else if (flag == "--seed" && hasValue) { config.seed = (unsigned int) strtoul(argv[++i], nullptr, 10); }
        else if (flag == "--graphs" && hasValue) { numberOfGraphs = atoi(argv[++i]); }
        else if (flag == "--out-prefix" && hasValue) { outputPrefix = argv[++i]; }
        else if (flag == "--no-report") { config.report = false; }
        else
        {
            if (rank == 0)
            {
                cerr << "Erro: flag desconhecida ou sem valor: '" << flag << "'." << endl;
            }
#ifdef USE_MPI
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
            return EXIT_FAILURE;
        }
    }

    bool valid = true;
    string problem;
    if (config.width < 1 || config.height < 1)
    {
        problem = "Erro: --width e --height precisam ser >= 1.";
        valid = false;
    }
    else if (config.regionsX < 1 || config.regionsY < 1 || config.regionsX > config.width || config.regionsY > config.height)
    {
        problem = "Erro: --regions-x/--regions-y precisam estar entre 1 e a dimensao correspondente.";
        valid = false;
    }
    else if (config.shortcuts < 0.0 || config.shortcuts > 1.0)
    {
        problem = "Erro: --shortcuts precisa estar em [0, 1].";
        valid = false;
    }
    else if (config.jitter < 0)
    {
        problem = "Erro: --jitter precisa ser >= 0.";
        valid = false;
    }
    else if (numberOfGraphs < 1)
    {
        problem = "Erro: --graphs precisa ser >= 1.";
        valid = false;
    }

    if (!valid)
    {
        if (rank == 0)
        {
            cerr << problem << endl;
        }
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
        return EXIT_FAILURE;
    }

    // Divisao circular: o grafo g fica com o processo g % worldSize. Com
    // --graphs igual ao numero de processos, e um grafo por processo; com mais
    // grafos que processos, cada um pega varios. Nada precisa ser combinado
    // entre eles, porque nao ha nada compartilhado.
    int failures = 0;

#ifdef USE_MPI
    for (int graph = rank; graph < numberOfGraphs; graph += worldSize)
    {
        GraphConfig local = config;
        // Semente propria por grafo, senao os N sairiam identicos.
        local.seed = config.seed + (unsigned int) graph;
        string report;
        if (!generateGraph(local, outputPrefix + to_string(graph) + ".txt", report))
        {
            failures++;
        }
        // Um unico << por relatorio: com varios processos escrevendo em cerr,
        // montar a saida linha a linha misturaria os relatorios.
        cerr << report << flush;
    }

    int totalFailures = 0;
    MPI_Reduce(&failures, &totalFailures, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    if (rank == 0 && totalFailures > 0)
    {
        return EXIT_FAILURE;
    }
#else
    // Sem MPI os grafos saem em threads OpenMP quando o binario foi compilado
    // com -fopenmp, e em sequencia quando nao. O laco e o mesmo nos dois casos:
    // as iteracoes nao compartilham nada.
    #pragma omp parallel for schedule(static) reduction(+:failures)
    for (int graph = 0; graph < numberOfGraphs; graph++)
    {
        GraphConfig local = config;
        local.seed = config.seed + (unsigned int) graph;
        string report;
        if (!generateGraph(local, outputPrefix + to_string(graph) + ".txt", report))
        {
            failures++;
        }
        #pragma omp critical
        cerr << report << flush;
    }

    if (failures > 0)
    {
        return EXIT_FAILURE;
    }
#endif

    return EXIT_SUCCESS;
}
