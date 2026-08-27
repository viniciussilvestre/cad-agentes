// genGraph - gerador de mapas de mundo pequeno para o PlagueGame.
//
// Emite uma grade width x height com atalhos de longo alcance de Kleinberg
// (destino com probabilidade proporcional a 1/d^2, d = distancia de Manhattan
// em celulas), respeitando um teto de grau de 6 arestas por no.
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
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

// Teto de grau. Igual ao PlagueGame::SLOTS: 4 vizinhos de grade + 2 atalhos.
static const int MAX_DEGREE = 6;
// Slots de atalho de saida por no. O no pode receber atalhos de outros, e como
// a aresta e nao-direcionada o receptor tambem consome um slot - por isso o
// teto de grau e verificado nos dois extremos.
static const int SHORTCUT_SLOTS = 2;
// Tentativas de sortear um destino que caia dentro da grade.
static const int MAX_ATTEMPTS = 30;

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

int main(int argc, char** argv)
{
    int width = 200;
    int height = 200;
    double shortcuts = 0.01;
    int regionsX = 5;
    int regionsY = 2;
    int spacing = 100;
    int jitter = 40;
    unsigned int seed = 42;
    string outputFile = "map.txt";

    for (int i = 1; i < argc; i++)
    {
        string flag = argv[i];
        bool hasValue = (i + 1 < argc);
        if (flag == "--width" && hasValue) { width = atoi(argv[++i]); }
        else if (flag == "--height" && hasValue) { height = atoi(argv[++i]); }
        else if (flag == "--shortcuts" && hasValue) { shortcuts = atof(argv[++i]); }
        else if (flag == "--regions-x" && hasValue) { regionsX = atoi(argv[++i]); }
        else if (flag == "--regions-y" && hasValue) { regionsY = atoi(argv[++i]); }
        else if (flag == "--spacing" && hasValue) { spacing = atoi(argv[++i]); }
        else if (flag == "--jitter" && hasValue) { jitter = atoi(argv[++i]); }
        else if (flag == "--seed" && hasValue) { seed = (unsigned int) strtoul(argv[++i], nullptr, 10); }
        else if (flag == "--out" && hasValue) { outputFile = argv[++i]; }
        else
        {
            cerr << "Erro: flag desconhecida ou sem valor: '" << flag << "'." << endl;
            return EXIT_FAILURE;
        }
    }

    if (width < 1 || height < 1)
    {
        cerr << "Erro: --width e --height precisam ser >= 1." << endl;
        return EXIT_FAILURE;
    }
    if (regionsX < 1 || regionsY < 1 || regionsX > width || regionsY > height)
    {
        cerr << "Erro: --regions-x/--regions-y precisam estar entre 1 e a dimensao correspondente." << endl;
        return EXIT_FAILURE;
    }
    if (shortcuts < 0.0 || shortcuts > 1.0)
    {
        cerr << "Erro: --shortcuts precisa estar em [0, 1]." << endl;
        return EXIT_FAILURE;
    }
    if (jitter < 0)
    {
        cerr << "Erro: --jitter precisa ser >= 0." << endl;
        return EXIT_FAILURE;
    }

    int numberOfNodes = width * height;
    int numberOfRegions = regionsX * regionsY;
    mt19937 rng(seed);

    // Coordenadas e regiao de cada no.
    vector<int> coordinateX(numberOfNodes);
    vector<int> coordinateY(numberOfNodes);
    vector<int> nodeRegion(numberOfNodes);
    uniform_int_distribution<int> jitterDistribution(-jitter, jitter);
    for (int i = 0; i < numberOfNodes; i++)
    {
        int row = i / width;
        int column = i % width;
        coordinateX[i] = column * spacing + jitterDistribution(rng);
        coordinateY[i] = row * spacing + jitterDistribution(rng);
        int regionColumn = (int) ((long long) column * regionsX / width);
        int regionRow = (int) ((long long) row * regionsY / height);
        nodeRegion[i] = regionRow * regionsX + regionColumn;
    }

    vector<vector<int>> adjacency(numberOfNodes);
    vector<int> degree(numberOfNodes, 0);
    unordered_set<long long> edgeKeys;

    // Arestas de grade.
    for (int i = 0; i < numberOfNodes; i++)
    {
        if (i % width != width - 1)
        {
            adjacency[i].push_back(i + 1);
            adjacency[i + 1].push_back(i);
            degree[i]++;
            degree[i + 1]++;
            edgeKeys.insert((long long) i * numberOfNodes + (i + 1));
        }
        if (i + width < numberOfNodes)
        {
            adjacency[i].push_back(i + width);
            adjacency[i + width].push_back(i);
            degree[i]++;
            degree[i + width]++;
            edgeKeys.insert((long long) i * numberOfNodes + (i + width));
        }
    }

    // Atalhos de longo alcance.
    int shortcutsCreated = 0;
    int shortcutsRejected = 0;
    int maxDistance = width + height - 2;
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
            if (slotDraw(rng) >= shortcuts)
            {
                continue;
            }
            if (degree[i] >= MAX_DEGREE)
            {
                shortcutsRejected++;
                continue;
            }
            int row = i / width;
            int column = i % width;
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
                if (targetColumn < 0 || targetColumn >= width || targetRow < 0 || targetRow >= height)
                {
                    continue;
                }
                int target = targetRow * width + targetColumn;
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
        cerr << "Erro: nao foi possivel escrever em '" << outputFile << "'." << endl;
        return EXIT_FAILURE;
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

    cerr << "arquivo: " << outputFile << "\n";
    cerr << "grade: " << width << "x" << height << "  spacing=" << spacing << "  jitter=" << jitter
         << "  seed=" << seed << "\n";
    cerr << "nos: " << numberOfNodes << "\n";
    cerr << "arestas: " << numberOfEdges << "\n";
    cerr << "grau medio: " << fixed << setprecision(3)
         << (numberOfNodes > 0 ? (double) degreeSum / (double) numberOfNodes : 0.0) << "\n";
    cerr << "grau minimo: " << minimumDegree << "  grau maximo: " << maximumDegree << "\n";
    cerr << "atalhos criados: " << shortcutsCreated << "  rejeitados: " << shortcutsRejected << "\n";
    cerr << "regioes: " << numberOfRegions << " (" << regionsX << "x" << regionsY << ")\n";
    cerr << "conexo: " << (connected ? "sim" : "NAO") << "\n";
    cerr << "diametro estimado (BFS de varredura dupla): " << diameter << "\n";
    if (!connected)
    {
        cerr << "AVISO: o grafo saiu desconexo (" << reached << " de " << numberOfNodes
             << " nos alcancados a partir do no 0). O arquivo foi gerado mesmo assim." << endl;
    }

    return EXIT_SUCCESS;
}
