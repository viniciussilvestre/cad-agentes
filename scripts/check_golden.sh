#!/bin/sh
# Oracle de regressao da Fase 0.
#
# Toda versao paralela (OpenMP, MPI, CUDA) precisa reproduzir este CSV byte a
# byte. O RNG e indexado por (id do agente, turno), nunca por um fluxo
# sequencial compartilhado, entao a ordem em que os agentes rodam nao pode
# mudar a saida. Se mudar, ha race.
#
# A configuracao e densa de proposito: nesta escala todos os cinco estados
# ficam populados o tempo todo, entao o CSV exercita todas as transicoes. Com
# os 590 agentes do padrao a epidemia nem pega, e o oracle nao pegaria erro
# nenhum na maquina de estados.
#
# Uso: scripts/check_golden.sh [binario]     (padrao: ./GameC)

set -e

BINARY="${1:-./GameC}"
GOLDEN="golden/baseline_v1.csv"
ACTUAL=$(mktemp)

ARGS="--map A1100a.txt --agents 5000 --infected 100 --turns 400 --seed 42"

"$BINARY" $ARGS --out-csv "$ACTUAL" > /dev/null

if diff -q "$GOLDEN" "$ACTUAL" > /dev/null 2>&1; then
    echo "golden OK: '$BINARY' reproduz $GOLDEN"
    rm -f "$ACTUAL"
    exit 0
fi

echo "golden FALHOU: '$BINARY' divergiu de $GOLDEN"
diff "$GOLDEN" "$ACTUAL" | head -20
rm -f "$ACTUAL"
exit 1
