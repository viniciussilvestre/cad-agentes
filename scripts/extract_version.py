#!/usr/bin/env python3
"""Extrai uma ou mais versoes de uma varredura para uma pasta nova.

    scripts/extract_version.py ORIGEM DESTINO VERSAO[,VERSAO...]

Caso de uso: a varredura do desktop mediu base, OpenMP e ROCm, mas para juntar
com a do servidor so o ROCm interessa - as outras ja foram medidas la, e deixar
as duas no CSV faria a media do grafico misturar maquinas.

    scripts/extract_version.py bench/20260910-090000 bench/so-rocm rocm
    scripts/merge_sweeps.py bench/juntas bench/20260910-133038 bench/so-rocm

O DESTINO sai no mesmo formato de uma varredura normal, entao o merge_sweeps.py
e o plot_scaling.py o leem sem saber a diferenca. O meta.txt de origem vai junto
com a nota do que foi extraido, e os brutos de cada corrida (runs/) tambem - sem
eles o CSV nao e reconstruivel se alguem apagar a origem.

Nao filtra por status nem por configuracao: quem checa isso e o merge_sweeps.py,
e este script nao deve esconder dele uma corrida que falhou.

SO BIBLIOTECA PADRAO.
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extrai versoes de uma varredura para uma pasta nova.",
        epilog="Depois: scripts/merge_sweeps.py DESTINO_JUNTO ORIGEM_A DESTINO")
    parser.add_argument("origem", type=Path)
    parser.add_argument("destino", type=Path)
    parser.add_argument("versoes",
                        help="nomes separados por virgula, ex: rocm ou omp,cuda")
    args = parser.parse_args()

    csvPath = args.origem / "timings.csv"
    if not csvPath.is_file():
        print(f"Erro: {csvPath} nao existe.", file=sys.stderr)
        return 1

    wanted = [v.strip() for v in args.versoes.split(",") if v.strip()]
    if not wanted:
        print("Erro: nenhuma versao informada.", file=sys.stderr)
        return 2

    with open(csvPath, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "versao" not in fields:
            print(f"Erro: {csvPath} nao tem coluna 'versao'.", file=sys.stderr)
            return 1
        rows = list(reader)

    present = Counter(r["versao"] for r in rows)
    unknown = [v for v in wanted if v not in present]
    if unknown:
        print(f"Erro: versao nao encontrada em {csvPath}: {', '.join(unknown)}",
              file=sys.stderr)
        print(f"Presentes: {', '.join(sorted(present))}", file=sys.stderr)
        return 1

    kept = [r for r in rows if r["versao"] in wanted]
    ok = sum(1 for r in kept if r.get("status", "").strip() in ("0", "ok"))

    destino = args.destino
    (destino / "plots").mkdir(parents=True, exist_ok=True)

    # Mesmas colunas da origem, na mesma ordem: se ela veio de uma juncao e tem
    # a coluna 'fonte', ela sobrevive.
    with open(destino / "timings.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)

    # csv/ e runs/ so das versoes extraidas. Os nomes que o bench_scaling.sh usa
    # sao csv/<versao>_S<n>.csv e runs/<versao>/S<n>/.
    for version in wanted:
        originRuns = args.origem / "runs" / version
        if originRuns.is_dir():
            shutil.copytree(originRuns, destino / "runs" / version,
                            dirs_exist_ok=True)
    originCsv = args.origem / "csv"
    if originCsv.is_dir():
        (destino / "csv").mkdir(exist_ok=True)
        for version in wanted:
            for pointCsv in originCsv.glob(f"{version}_S*.csv"):
                shutil.copy2(pointCsv, destino / "csv" / pointCsv.name)

    with open(destino / "meta.txt", "w", encoding="utf-8") as handle:
        handle.write(f"extraido de {args.origem}\n")
        handle.write(f"versoes extraidas: {', '.join(wanted)}\n")
        handle.write(f"deixadas de fora:  "
                     f"{', '.join(sorted(set(present) - set(wanted))) or 'nenhuma'}\n")
        handle.write(f"linhas: {len(kept)} ({ok} com status ok)\n")
        handle.write("\n--- meta.txt da origem ---\n")
        originMeta = args.origem / "meta.txt"
        handle.write(originMeta.read_text(encoding="utf-8")
                     if originMeta.is_file() else "(sem meta.txt)\n")

    print(f"origem:  {csvPath}")
    print(f"         versoes presentes: "
          + ", ".join(f"{v}({c})" for v, c in sorted(present.items())))
    print(f"extraiu: {', '.join(wanted)} -> {len(kept)} linhas ({ok} com status ok)")
    print(f"destino: {destino}")
    print(f"\njunte com: scripts/merge_sweeps.py DESTINO_JUNTO OUTRA_PASTA {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
