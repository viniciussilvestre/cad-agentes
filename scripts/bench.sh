#!/bin/bash
# Roda o mesmo comando N vezes e resume os tempos.
#
# Uso:
#   scripts/bench.sh [-n N] [-o saida.csv] [-k pasta] [-q] -- <comando...>
#
#   -n N        quantas repeticoes (padrao: 5)
#   -o arquivo  grava run,laco,total,status em CSV
#   -k pasta    guarda stdout e stderr de cada repeticao
#   -q          so o resumo, sem a linha de cada repeticao
#
# Tudo depois do -- e executado como esta, entao prefixos entram normalmente:
#
#   scripts/bench.sh -n 10 -- ./GameC_cuda --games 4 --turns 100
#   scripts/bench.sh -n 5  -- taskset -c 0-23 ./GameC_omp --games 8
#   OMP_NUM_THREADS=24 scripts/bench.sh -n 5 -- ./GameC_omp --turns 50
#
# Duas medidas por repeticao, e a diferenca entre elas importa:
#
#   laco   o que o programa imprime em "tempo:", so o laco de turnos;
#   total  parede do processo inteiro, incluindo ler os mapas e subir para a
#          GPU. Com mapas grandes a carga domina, e comparar so o laco esconde
#          isso.

set -u

repetitions=5
csvFile=""
keepDir=""
quiet=0

while [ $# -gt 0 ]; do
    case "$1" in
        -n) repetitions="$2"; shift 2 ;;
        -o) csvFile="$2"; shift 2 ;;
        -k) keepDir="$2"; shift 2 ;;
        -q) quiet=1; shift ;;
        --) shift; break ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)  echo "Erro: opcao desconhecida '$1'. Use -h." >&2; exit 1 ;;
    esac
done

if [ $# -eq 0 ]; then
    echo "Erro: falta o comando. Use: $0 [-n N] -- <comando...>" >&2
    exit 1
fi

case "$repetitions" in
    ''|*[!0-9]*) echo "Erro: -n precisa ser um inteiro positivo." >&2; exit 1 ;;
esac
if [ "$repetitions" -lt 1 ]; then
    echo "Erro: -n precisa ser >= 1." >&2
    exit 1
fi

if [ -n "$keepDir" ]; then
    mkdir -p "$keepDir"
fi
if [ -n "$csvFile" ]; then
    echo "run,laco,total,status" > "$csvFile"
fi

echo "comando: $*"
echo "repeticoes: $repetitions"
echo

if [ "$quiet" -eq 0 ]; then
    printf "%-5s %12s %12s  %s\n" "run" "laco (s)" "total (s)" "status"
fi

failures=0
loopTimes=""
wallTimes=""

for run in $(seq 1 "$repetitions"); do
    if [ -n "$keepDir" ]; then
        outFile="$keepDir/run_$run.out"
        errFile="$keepDir/run_$run.err"
    else
        outFile=$(mktemp)
        errFile=$(mktemp)
    fi

    start=$(date +%s.%N)
    "$@" >"$outFile" 2>"$errFile"
    status=$?
    end=$(date +%s.%N)

    wall=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.3f", b - a }')

    # "tempo: 0.374 s" - o cronometro interno do proprio programa.
    loop=$(grep -o 'tempo:[[:space:]]*[0-9.]*' "$errFile" | grep -o '[0-9.]*' | head -1)
    [ -z "$loop" ] && loop="-"

    if [ "$status" -ne 0 ]; then
        failures=$((failures + 1))
        label="FALHOU ($status)"
    else
        label="ok"
        [ "$loop" != "-" ] && loopTimes="$loopTimes $loop"
        wallTimes="$wallTimes $wall"
    fi

    if [ "$quiet" -eq 0 ]; then
        printf "%-5s %12s %12s  %s\n" "$run" "$loop" "$wall" "$label"
    fi
    if [ -n "$csvFile" ]; then
        echo "$run,$loop,$wall,$status" >> "$csvFile"
    fi
    if [ -z "$keepDir" ]; then
        rm -f "$outFile" "$errFile"
    fi
done

# Media, desvio, minimo e maximo. A mediana entra junto porque uma repeticao
# fria - primeira leitura dos mapas, contexto de CUDA ainda nao criado - puxa a
# media para cima sem dizer nada sobre o regime.
summarize() {
    local label="$1"
    shift
    if [ -z "${*// /}" ]; then
        printf "  %-10s sem amostra\n" "$label"
        return
    fi
    printf '%s\n' $* | sort -g | awk -v label="$label" '
        { value[NR] = $1; sum += $1 }
        END {
            n = NR
            mean = sum / n
            for (i = 1; i <= n; i++) { d = value[i] - mean; variance += d * d }
            sd = (n > 1) ? sqrt(variance / (n - 1)) : 0
            median = (n % 2) ? value[(n + 1) / 2] : (value[n / 2] + value[n / 2 + 1]) / 2
            printf "  %-10s media %8.3f   dp %7.3f   mediana %8.3f   min %8.3f   max %8.3f\n",
                   label, mean, sd, median, value[1], value[n]
        }'
}

echo
echo "resumo de $repetitions repeticoes:"
summarize "laco" $loopTimes
summarize "total" $wallTimes

if [ "$failures" -gt 0 ]; then
    echo
    echo "$failures de $repetitions repeticoes falharam." >&2
    exit 1
fi
