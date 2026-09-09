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
#   scripts/bench_scaling.sh [-n REP] [-t TURNOS] [-g JOGOS] [-v LISTA] [-o PASTA] [-s]
#
#   -n REP     repeticoes por ponto (padrao: 5)
#   -t TURNOS  turnos por corrida (padrao: 600)
#   -g JOGOS   jogos (padrao: 20)
#   -v LISTA   restringe as versoes, separadas por virgula (padrao: todas as
#              que rodarem). Ex: -v base,rocm
#   -o PASTA   pasta de saida (padrao: bench/<data-hora>)
#   -s         so mede, nao desenha
#   -d         so mostra o plano (versoes, pontos, comandos) e sai
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

repetitions=5
turns=600
games=20
outDir=""
skipPlots=0
dryRun=0
onlyVersions=""

while [ $# -gt 0 ]; do
    case "$1" in
        -n) repetitions="$2"; shift 2 ;;
        -t) turns="$2"; shift 2 ;;
        -g) games="$2"; shift 2 ;;
        -v) onlyVersions="$2"; shift 2 ;;
        -o) outDir="$2"; shift 2 ;;
        -s) skipPlots=1; shift ;;
        -d) dryRun=1; shift ;;
        -h|--help) sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Erro: opcao desconhecida '$1'. Use -h." >&2; exit 1 ;;
    esac
done

for value in "$repetitions" "$turns" "$games"; do
    case "$value" in
        ''|*[!0-9]*) echo "Erro: -n, -t e -g precisam ser inteiros positivos." >&2; exit 1 ;;
    esac
done
[ "$repetitions" -lt 1 ] && { echo "Erro: -n precisa ser >= 1." >&2; exit 1; }

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
# A serie. 19 pontos: suscetiveis de 100 mil a 1 milhao, de 50 mil em 50 mil, e
# infectados sempre a um decimo disso.
#
# A razao 10:1 fica constante de proposito. O que se quer medir e escala, e com
# a razao fixa a fracao de agentes em cada estado fica parecida entre os pontos
# - a curva mostra o custo crescendo com o numero de agentes, e nao a epidemia
# mudando de regime no meio da varredura.
#
# Sao TOTAIS, divididos entre os jogos: com -g 20, o primeiro ponto da 5500
# agentes por jogo e o ultimo da 55000.
# ---------------------------------------------------------------------------
points=19
firstHealthy=100000
stepHealthy=50000
firstInfected=10000
stepInfected=5000

# ---------------------------------------------------------------------------
# Deteccao: o que roda nesta maquina.
#
# Duas perguntas diferentes, e as duas importam. "O binario existe?" nao
# responde nada sobre GPU: GameC_cuda compila numa maquina sem NVIDIA e so
# falha na hora de rodar. Entao alem do teste de existencia ha uma corrida
# minima de verdade, e quem nao passar nas duas fica de fora.
# ---------------------------------------------------------------------------
candidates="base:GameC omp:GameC_omp cuda:GameC_cuda rocm:GameC_rocm"

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
    if ./"$binary" --games 1 --agents 1000 --infected 100 --turns 1 >/dev/null 2>&1; then
        available="$available $label:$binary"
    else
        skipped="$skipped $label(nao roda aqui)"
    fi
done

if [ -z "$available" ]; then
    echo "Erro: nenhuma versao executavel. Rode 'make' primeiro." >&2
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
            printf '  %-5s ./%s --games %s --agents %s --infected %s --turns %s\n' \
                "$label" "$binary" "$games" "$healthy" "$infected" "$turns"
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
    start=$(now)
    ./"$binary" --games "$games" --agents "$maxHealthy" --infected "$maxInfected" \
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
    printf '  %-5s %ss no maior ponto\n' "$label" "$seconds"
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
        printf '[%2d/%2d] %-5s S=%-8s I=%-7s (decorrido %dm%02ds)\n' \
            "$step" "$totalSteps" "$label" "$healthy" "$infected" \
            $(( elapsed / 60 )) $(( elapsed % 60 ))

        "$scriptDir/bench.sh" -n "$repetitions" -o "$pointCsv" -k "$pointDir" -q -- \
            ./"$binary" --games "$games" --agents "$healthy" --infected "$infected" \
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
