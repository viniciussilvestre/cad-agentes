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
//   <x> <y> <regiao>          x nNos
//   <linha em branco>
//   <origem> <destino> <peso> x nArestas
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
#include <unordered_set>
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
    int width = 10000;
    int height = 10000;
    double shortcuts = 0.01;
    int regionsX = 5;
    int regionsY = 2;
    int spacing = 100;
    int jitter = 40;
    unsigned int seed = 42;
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

static vector<int> bfsDistances(const vector<vector<int>>& adjacency, int source)
{
    vector<int> distance(adjacency.size(), -1);
    queue<int> pending;
    distance[source] = 0;
    pending.push(source);
    while (!pending.empty())
    {
        int node = pending.front();
        pending.pop();
        for (int neighbor : adjacency[node])
        {
            if (distance[neighbor] == -1)
            {
                distance[neighbor] = distance[node] + 1;
                pending.push(neighbor);
            }
        }
    }
    return distance;
}

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

    vector<vector<int>> adjacency(numberOfNodes);
    vector<int> degree(numberOfNodes, 0);
    unordered_set<long long> edgeKeys;

    // Arestas de grade.
    for (int i = 0; i < numberOfNodes; i++)
    {
        if (i % config.width != config.width - 1)
        {
            adjacency[i].push_back(i + 1);
            adjacency[i + 1].push_back(i);
            degree[i]++;
            degree[i + 1]++;
            edgeKeys.insert((long long) i * numberOfNodes + (i + 1));
        }
        if (i + config.width < numberOfNodes)
        {
            adjacency[i].push_back(i + config.width);
            adjacency[i + config.width].push_back(i);
            degree[i]++;
            degree[i + config.width]++;
            edgeKeys.insert((long long) i * numberOfNodes + (i + config.width));
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
                long long key = (long long) min(i, target) * numberOfNodes + max(i, target);
                if (edgeKeys.find(key) != edgeKeys.end())
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
                adjacency[i].push_back(target);
                adjacency[target].push_back(i);
                degree[i]++;
                degree[target]++;
                edgeKeys.insert(key);
                shortcutsCreated++;
                break;
            }
        }
    }

    // Emissao: origem crescente, contigua por origem, origem < destino.
    ofstream output(outputFile);
    if (!output.is_open())
    {
        report = "Erro: nao foi possivel escrever em '" + outputFile + "'.\n";
        return false;
    }

    int numberOfEdges = (int) edgeKeys.size();
    output << numberOfNodes << "\t" << numberOfEdges << "\t" << numberOfRegions << "\n";
    output << "\n";
    for (int i = 0; i < numberOfNodes; i++)
    {
        output << coordinateX[i] << "\t" << coordinateY[i] << "\t" << nodeRegion[i] << "\n";
    }
    output << "\n";
    output << fixed << setprecision(2);
    for (int i = 0; i < numberOfNodes; i++)
    {
        vector<int> targets;
        for (int neighbor : adjacency[i])
        {
            if (neighbor > i)
            {
                targets.push_back(neighbor);
            }
        }
        sort(targets.begin(), targets.end());
        for (int target : targets)
        {
            double deltaX = (double) (coordinateX[target] - coordinateX[i]);
            double deltaY = (double) (coordinateY[target] - coordinateY[i]);
            output << i << "\t" << target << "\t" << sqrt(deltaX * deltaX + deltaY * deltaY) << "\n";
        }
    }
    output.close();

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

    vector<int> firstSweep = bfsDistances(adjacency, 0);
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
    vector<int> secondSweep = bfsDistances(adjacency, source);
    int diameter = secondSweep[farthestNode(secondSweep)];

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
    out << "conexo: " << (connected ? "sim" : "NAO") << "\n";
    out << "diametro estimado (BFS de varredura dupla): " << diameter << "\n";
    if (!connected)
    {
        out << "AVISO: o grafo saiu desconexo (" << reached << " de " << numberOfNodes
            << " nos alcancados a partir do no 0). O arquivo foi gerado mesmo assim.\n";
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
