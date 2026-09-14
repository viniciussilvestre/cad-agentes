#!/bin/sh
# Varredura de escalabilidade em numero de agentes.
#
# Roda cada versao DISPONIVEL para uma serie crescente de agentes, N repeticoes
# por ponto, e no fim desenha tempo e speedup.
#
# NENHUMA VERSAO E OBRIGATORIA. O script descobre sozinho o que da para rodar
# nesta maquina e trabalha com isso: numa maquina so de CPU sai base + OpenMP,
# numa com Radeon entra o ROCm, numa com NVIDIA entra o CUDA, e numa com as
# duas entram os quatro. Binario que nao existe ou que existe mas nao roda -
# GameC_cuda compila sem placa da NVIDIA e falha ao executar - sai da varredura
# com recado, e nao com dezenas de linhas de falha no CSV.
#
# MEDIR E DESENHAR SAO SEPARADOS DE PROPOSITO. Este script so mede e grava; os
# graficos saem do plot_scaling.py lendo o que ficou em disco. Uma varredura
# completa custa dezenas de minutos, entao mudar a escala de um eixo nao pode
# exigir medir tudo de novo.
#
# Uso:
#   scripts/bench_scaling.sh [-n REP] [-t TURNOS] [-g JOGOS] [-a INICIO]
#                            [-A PASSO] [-p PONTOS] [-v LISTA] [-o PASTA] [-s] [-d]
#
#   -n REP     repeticoes por ponto (padrao: 5)
#   -t TURNOS  turnos por corrida (padrao: 600)
#   -g JOGOS   jogos (padrao: 28)
#   -a INICIO  suscetiveis do primeiro ponto (padrao: 100000)
#   -A PASSO   quanto somar de suscetiveis a cada ponto (padrao: 50000)
#   -p PONTOS  quantos pontos na serie (padrao: 19)
#   -r RANKS   processos do mpirun na versao mpi (padrao: min(jogos, nucleos)).
#              Acima do numero de jogos nao ha o que ganhar - o teto do MPI e
#              um jogo por rank, ver mpi/simulation.h.
#
#              CUIDADO AO LER O GRAFICO COM POUCOS TURNOS. O que vai para o
#              desenho e o tempo TOTAL do processo, e o mpirun cobra um
#              arranque fixo que nao encolhe com a carga - medido em ~76 ms a
#              mais que o binario direto. Com os 600 turnos padrao isso dilui;
#              com -t 60 ele domina, e a versao mpi parece pior do que e. A
#              coluna `laco` do timings.csv separa as duas coisas: num teste de
#              60 turnos o laco deu 0,141 s contra 0,144 s do omp-jogos - ou
#              seja, empate - enquanto o total deu 0,410 s contra 0,338 s.
#   -v LISTA   roda SO estas versoes, separadas por virgula. Nomes validos:
#              base, omp, omp-jogos, omp-nested, mpi, cuda, rocm, metal.
#              Padrao: todas as que rodarem nesta maquina.
#                -v rocm            so a Radeon
#                -v metal           so o Metal
#                -v base,rocm       uma GPU e a referencia sequencial
#                -v omp,omp-jogos,omp-nested   comparar as tres de OpenMP
#              Sem a base na lista nao ha referencia, e o grafico de speedup
#              sai com aviso em vez de curvas.
#   -o PASTA   pasta de saida (padrao: bench/<data-hora>)
#   -s         so mede, nao desenha
#   -d         so mostra o plano (versoes, pontos, comandos) e sai
#
# Os infectados sao sempre INICIO/10, crescendo PASSO/10 por ponto.
#
# CUIDADO COM O CUSTO. A versao base e sequencial e domina a varredura: o custo
# dela cresce com agentes x turnos, e cada ponto ainda roda REP vezes. Antes de
# uma varredura grande, use -d para ver o plano e leia a estimativa que o
# aquecimento imprime - ela mede o maior ponto de verdade antes de comecar.
# Se so a curva paralela interessa, -v omp,cuda,rocm corta o grosso do tempo.
#
# Saida:
#   PASTA/timings.csv        uma linha por repeticao - a fonte dos graficos
#   PASTA/csv/<v>_S<n>.csv   o CSV por ponto, como o bench.sh escreve
#   PASTA/runs/<v>/S<n>/     stdout e stderr de cada repeticao
#   PASTA/meta.txt           maquina, GPU, compilador, commit
#   PASTA/plots/*.png        os graficos
#
# Para redesenhar sem medir de novo:
#   scripts/plot_scaling.py PASTA

set -u

# Locale C para as contas de tempo. Em pt_BR o awk imprime "0,123" com virgula
# decimal, e o CSV e separado por virgula: o campo se parte em dois, todas as
# colunas deslocam e o status vira lixo - a varredura roda inteira, sem erro
# nenhum no caminho, e o CSV sai imprestavel. Exportado, entao vale tambem para
# o bench.sh, que e quem formata o tempo de parede.
export LC_ALL=C

# PADROES FIXOS: suscetiveis de 100 mil a 1 milhao de 50 mil em 50 mil, 19
# pontos, infectados sempre um decimo, 28 jogos, 600 turnos.
#
# NAO MUDAR. Varredura e caro e os pontos so se comparam entre si com a MESMA
# configuracao: mexer aqui invalida em silencio todo CSV medido antes, e quem
# roda o script pelado nao tem como perceber. Configuracao diferente vai por
# flag na linha de comando, que fica registrada no meta.txt da varredura.
#
# Por que 28 jogos: e o numero de threads da maquina de referencia, o que da
# exatamente um jogo por thread na variante GameC_omp_games.
repetitions=5
turns=600
games=28
firstHealthy=100000
stepHealthy=50000
points=19
outDir=""
skipPlots=0
dryRun=0
onlyVersions=""
ranks=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n) repetitions="$2"; shift 2 ;;
        -t) turns="$2"; shift 2 ;;
        -g) games="$2"; shift 2 ;;
        -a) firstHealthy="$2"; shift 2 ;;
        -A) stepHealthy="$2"; shift 2 ;;
        -p) points="$2"; shift 2 ;;
        -r) ranks="$2"; shift 2 ;;
        -v) onlyVersions="$2"; shift 2 ;;
        -o) outDir="$2"; shift 2 ;;
        -s) skipPlots=1; shift ;;
        -d) dryRun=1; shift ;;
        -h|--help) sed -n '2,49p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Erro: opcao desconhecida '$1'. Use -h." >&2; exit 1 ;;
    esac
done

for value in "$repetitions" "$turns" "$games" "$firstHealthy" "$stepHealthy" "$points" "$ranks"; do
    case "$value" in
        ''|*[!0-9]*) echo "Erro: -n, -t, -g, -a, -A, -p e -r precisam ser inteiros nao negativos." >&2; exit 1 ;;
    esac
done
[ "$repetitions" -lt 1 ] && { echo "Erro: -n precisa ser >= 1." >&2; exit 1; }
[ "$points" -lt 1 ] && { echo "Erro: -p precisa ser >= 1." >&2; exit 1; }
[ "$firstHealthy" -lt 10 ] && { echo "Erro: -a precisa ser >= 10 (infectados sao -a/10)." >&2; exit 1; }

scriptDir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
rootDir=$(CDPATH= cd -- "$scriptDir/.." && pwd)
cd "$rootDir" || exit 1

# Relogio com fracao de segundo. O %N e do coreutils da GNU; num date que nao
# o entenda a saida vem com um "N" literal no lugar dos nanossegundos, e o sed
# a transforma em zero em vez de deixar a conta virar NaN.
now() {
    date +%s.%N 2>/dev/null | sed 's/\.N$/.0/'
}

# O -Is tambem e da GNU; sem ele, a data sai no formato que houver.
stamp() {
    date -Is 2>/dev/null || date
}

[ -z "$outDir" ] && outDir="bench/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$outDir/csv" "$outDir/runs" "$outDir/plots" || exit 1

# ---------------------------------------------------------------------------
# A serie: PONTOS pontos, comecando em INICIO suscetiveis e somando PASSO a
# cada ponto. O padrao vai de 100 mil a 1 milhao de 50 mil em 50 mil.
#
# Os infectados sao sempre um decimo dos suscetiveis. A razao 10:1 fica fixa de
# proposito: o que se quer medir e escala, e com a razao constante a fracao de
# agentes em cada estado fica parecida entre os pontos - a curva mostra o custo
# crescendo com o numero de agentes, e nao a epidemia mudando de regime no meio
# da varredura.
#
# Sao TOTAIS, divididos entre os jogos: com o padrao de 28 jogos, o primeiro
# ponto (110 mil agentes) da cerca de 3900 por jogo, e o ultimo (1,1 milhao) da
# cerca de 39 mil.
# ---------------------------------------------------------------------------
firstHealthy=$(( firstHealthy ))
stepHealthy=$(( stepHealthy ))
firstInfected=$(( firstHealthy / 10 ))
stepInfected=$(( stepHealthy / 10 ))

# ---------------------------------------------------------------------------
# Deteccao: o que roda nesta maquina.
#
# Duas perguntas diferentes, e as duas importam. "O binario existe?" nao
# responde nada sobre GPU: GameC_cuda compila numa maquina sem NVIDIA e so
# falha na hora de rodar. Entao alem do teste de existencia ha uma corrida
# minima de verdade, e quem nao passar nas duas fica de fora.
# ---------------------------------------------------------------------------
# As tres variantes de OpenMP saem do mesmo fonte, mudando so onde entra o
# paralelismo (ver Makefile e openmp/simulation.cpp): agentes, jogos, ou os dois
# aninhados. Sao opcionais como as GPUs - quem nao compilou fica de fora.
candidates="base:GameC omp:GameC_omp omp-jogos:GameC_omp_games omp-nested:GameC_omp_nested mpi:GameC_mpi cuda:GameC_cuda rocm:GameC_rocm metal:GameC_metal"

knownLabels=$(for pair in $candidates; do printf '%s ' "${pair%%:*}"; done)

# Nome errado em -v nao pode passar em silencio. Sem esta checagem, "-v rcm"
# simplesmente nao casaria com nada e a varredura morreria dizendo que nenhuma
# versao roda nesta maquina - mensagem que aponta para o hardware quando o
# problema e um typo.
if [ -n "$onlyVersions" ]; then
    for wanted in $(echo "$onlyVersions" | tr ',' ' '); do
        case " $knownLabels " in
            *" $wanted "*) ;;
            *)
                echo "Erro: versao desconhecida em -v: '$wanted'." >&2
                echo "Validas: $knownLabels" >&2
                exit 1
                ;;
        esac
    done
fi

# ---------------------------------------------------------------------------
# Lancador: como cada versao e posta para rodar.
#
# Quase todas SAO o proprio binario. A de MPI nao: e `mpirun -n N` na frente
# dele, e o N e mais uma dimensao da varredura. O padrao e o menor entre os
# jogos e os nucleos - acima do numero de jogos o MPI nao tem o que dividir
# (ver mpi/simulation.h), e acima dos nucleos so ha disputa.
#
# O --oversubscribe e do Open MPI e nao existe no MPICH, entao nao pode ser
# fixo: a deteccao tenta com e sem, e fica com o que rodar.
# ---------------------------------------------------------------------------
cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
if [ "$ranks" -eq 0 ]; then
    ranks=$games
    [ "$cores" -lt "$ranks" ] && ranks=$cores
fi
[ "$ranks" -lt 1 ] && ranks=1

mpiLauncher=""

launcherFor() {
    case "$1" in
        mpi) printf '%s' "$mpiLauncher" ;;
        *)   printf '' ;;
    esac
}

available=""
skipped=""
for pair in $candidates; do
    label=${pair%%:*}
    binary=${pair#*:}

    if [ -n "$onlyVersions" ]; then
        case ",$onlyVersions," in
            *",$label,"*) ;;
            *) continue ;;
        esac
    fi

    if [ ! -x "./$binary" ]; then
        skipped="$skipped $label(sem binario)"
        continue
    fi

    if [ "$label" = "mpi" ]; then
        if ! command -v mpirun >/dev/null 2>&1; then
            skipped="$skipped mpi(sem mpirun)"
            continue
        fi
        mpiLauncher=""
        for attempt in "mpirun -n $ranks --oversubscribe" "mpirun -n $ranks"; do
            if $attempt ./"$binary" --games 1 --agents 1000 --infected 100 --turns 1 \
                 >/dev/null 2>&1; then
                mpiLauncher="$attempt"
                break
            fi
        done
        if [ -z "$mpiLauncher" ]; then
            skipped="$skipped mpi(mpirun falhou)"
            continue
        fi
        available="$available $label:$binary"
        continue
    fi

    if ./"$binary" --games 1 --agents 1000 --infected 100 --turns 1 >/dev/null 2>&1; then
        available="$available $label:$binary"
    else
        skipped="$skipped $label(nao roda aqui)"
    fi
done

if [ -z "$available" ]; then
    # Diz POR QUE cada uma ficou fora. Com -v rocm numa maquina sem Radeon, um
    # "rode make" generico manda o usuario compilar o que nao ia rodar de todo
    # jeito.
    echo "Erro: nenhuma das versoes pedidas roda nesta maquina." >&2
    echo "Fora:${skipped:- nenhuma}" >&2
    echo "(sem binario = falta compilar; nao roda aqui = compilou mas falhou," >&2
    echo " tipicamente GPU ausente ou driver de outro fabricante)" >&2
    exit 1
fi

versionList=$(echo "$available" | sed 's/:[^ ]*//g' | sed 's/^ *//')
versionCount=$(echo "$available" | wc -w)

# ---------------------------------------------------------------------------
# Os mapas precisam existir ANTES de comecar.
#
# A lista de mapas esta fixa no main de cada versao: com -g N sao lidos
# map_0.txt ate map_(N-1).txt. Numa maquina nova esses arquivos nao vem do
# repositorio - estao no .gitignore porque sao gerados e somam dezenas de GB.
#
# Sem esta checagem, a varredura falharia N x REP vezes seguidas, cada uma
# depois de ja ter aberto o binario, e a pasta de saida terminaria cheia de
# erro identico. Melhor parar aqui e dizer o comando que resolve.
# ---------------------------------------------------------------------------
missing=0
index=0
while [ "$index" -lt "$games" ]; do
    [ -f "map_$index.txt" ] || missing=$(( missing + 1 ))
    index=$(( index + 1 ))
done
if [ "$missing" -gt 0 ]; then
    echo "Erro: faltam $missing dos $games mapas (map_0.txt ... map_$(( games - 1 )).txt)." >&2
    echo "Gere com:  ./genGraph --graphs $games --out-prefix map_" >&2
    echo "(sao ~42 MB por mapa, ~$(( games * 42 )) MB no total)" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Procedencia. Sem isto, um CSV de daqui a um mes e um numero sem maquina.
# ---------------------------------------------------------------------------
{
    echo "data:            $(stamp)"
    echo "host:            $(uname -sr)"
    echo "cpu:             $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2- | sed 's/^ *//')"
    echo "nproc:           $(nproc 2>/dev/null || echo '?')"
    echo "OMP_NUM_THREADS: ${OMP_NUM_THREADS:-<nao definido, usa nproc>}"
    echo "gpu amd:         $(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || echo 'nenhuma')"
    # O nvidia-smi imprime o erro de driver no STDOUT, entao filtrar por '^GPU'
    # e o que separa "achei uma placa" de "achei uma reclamacao".
    echo "gpu nvidia:      $(nvidia-smi -L 2>/dev/null | grep -m1 '^GPU' || echo 'nenhuma')"
    echo "git:             $(git rev-parse --short HEAD 2>/dev/null || echo '?') $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
    echo "git sujo:        $(git status --porcelain 2>/dev/null | wc -l) arquivos modificados"
    echo "turnos:          $turns"
    echo "jogos:           $games"
    echo "repeticoes:      $repetitions"
    echo "pontos:          $points"
    echo "mpi ranks:       ${mpiLauncher:-<mpi fora da varredura>}"
    echo "versoes medidas: $versionList"
    echo "versoes fora:   ${skipped:- nenhuma}"
} > "$outDir/meta.txt"

totalPoints=$(( points * versionCount ))

master="$outDir/timings.csv"
echo "versao,binario,suscetiveis,infectados,agentes,jogos,turnos,rep,laco,total,status" > "$master"

echo
cat "$outDir/meta.txt"
echo
echo "saida: $outDir"
echo

if [ "$dryRun" -eq 1 ]; then
    echo "plano ($totalPoints corridas de $repetitions repeticoes):"
    for entry in $available; do
        label=${entry%%:*}
        binary=${entry#*:}
        index=0
        while [ "$index" -lt "$points" ]; do
            healthy=$(( firstHealthy + index * stepHealthy ))
            infected=$(( firstInfected + index * stepInfected ))
            index=$(( index + 1 ))
            launcher=$(launcherFor "$label")
            printf '  %-9s %s./%s --games %s --agents %s --infected %s --turns %s\n' \
                "$label" "${launcher:+$launcher }" "$binary" "$games" "$healthy" \
                "$infected" "$turns"
        done
    done
    rm -rf "$outDir"
    exit 0
fi

# ---------------------------------------------------------------------------
# Aquecimento no MAIOR ponto. Serve a duas coisas: poe os mapas na cache de
# pagina do sistema - a primeira leitura de 20 arquivos de 41 MB e bem mais
# lenta que as seguintes, e sem isso a primeira repeticao de cada versao sairia
# artificialmente ruim - e da a constante para estimar a varredura.
# ---------------------------------------------------------------------------
maxHealthy=$(( firstHealthy + (points - 1) * stepHealthy ))
maxInfected=$(( firstInfected + (points - 1) * stepInfected ))

echo "aquecendo no maior ponto e estimando..."
estimate=0
for entry in $available; do
    label=${entry%%:*}
    binary=${entry#*:}
    launcher=$(launcherFor "$label")
    start=$(now)
    $launcher ./"$binary" --games "$games" --agents "$maxHealthy" --infected "$maxInfected" \
                --turns "$turns" >/dev/null 2>&1
    end=$(now)
    seconds=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }')
    estimate=$(awk -v e="$estimate" -v s="$seconds" -v n="$repetitions" -v p="$points" \
                   -v f="$firstHealthy" -v d="$stepHealthy" \
        'BEGIN {
             # A varredura custa o maior ponto vezes a soma das fracoes
             # (agentes_i / agentes_max), assumindo custo linear em agentes.
             fraction = 0
             for (i = 0; i < p; i++) fraction += (f + i * d) / (f + (p - 1) * d)
             printf "%.0f", e + s * fraction * n
         }')
    printf '  %-9s %ss no maior ponto\n' "$label" "$seconds"
done
printf 'estimativa da varredura: ~%s min\n\n' \
    "$(awk -v e="$estimate" 'BEGIN { printf "%.0f", (e + 59) / 60 }')"

# ---------------------------------------------------------------------------
# A varredura.
# ---------------------------------------------------------------------------
sweepStart=$(date +%s)
step=0
totalSteps=$totalPoints

for entry in $available; do
    label=${entry%%:*}
    binary=${entry#*:}

    index=0
    while [ "$index" -lt "$points" ]; do
        healthy=$(( firstHealthy + index * stepHealthy ))
        infected=$(( firstInfected + index * stepInfected ))
        agents=$(( healthy + infected ))
        index=$(( index + 1 ))
        step=$(( step + 1 ))

        pointCsv="$outDir/csv/${label}_S${healthy}.csv"
        pointDir="$outDir/runs/$label/S$healthy"
        mkdir -p "$pointDir"

        elapsed=$(( $(date +%s) - sweepStart ))
        printf '[%2d/%2d] %-9s S=%-8s I=%-7s (decorrido %dm%02ds)\n' \
            "$step" "$totalSteps" "$label" "$healthy" "$infected" \
            $(( elapsed / 60 )) $(( elapsed % 60 ))

        # $launcher sem aspas de proposito: quando nao e vazio, precisa
        # quebrar em palavras (mpirun, -n, 8, ...).
        launcher=$(launcherFor "$label")
        "$scriptDir/bench.sh" -n "$repetitions" -o "$pointCsv" -k "$pointDir" -q -- \
            $launcher ./"$binary" --games "$games" --agents "$healthy" --infected "$infected" \
                        --turns "$turns" > "$pointDir/bench.log" 2>&1

        # O bench.sh escreve run,laco,total,status. Aqui isso vira uma linha
        # larga, com a configuracao junto - o formato que o grafico consome.
        awk -F, -v v="$label" -v b="$binary" -v h="$healthy" -v i="$infected" \
                -v a="$agents" -v g="$games" -v t="$turns" \
            'NR > 1 { print v","b","h","i","a","g","t","$1","$2","$3","$4 }' \
            "$pointCsv" >> "$master"
    done
done

sweepEnd=$(date +%s)
printf '\nvarredura terminada em %dm%02ds\n' \
    $(( (sweepEnd - sweepStart) / 60 )) $(( (sweepEnd - sweepStart) % 60 ))
echo "medicoes: $master"

if [ "$skipPlots" -eq 0 ]; then
    echo
    if command -v python3 >/dev/null 2>&1; then
        python3 "$scriptDir/plot_scaling.py" "$outDir"
    else
        echo "python3 nao encontrado: graficos nao gerados." >&2
        echo "Depois, em outra maquina: scripts/plot_scaling.py $outDir" >&2
    fi
fi
