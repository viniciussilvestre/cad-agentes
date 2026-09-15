#!/usr/bin/env python3
"""Junta varreduras de maquinas diferentes num CSV para plotar juntas.

    scripts/merge_sweeps.py DESTINO PASTA1 PASTA2 [PASTA3 ...]
    scripts/plot_scaling.py DESTINO

Caso de uso: a base, o OpenMP e o CUDA foram medidos no servidor, o ROCm na
maquina com Radeon e o Metal no Mac. Cada maquina produziu sua propria pasta
bench/<data-hora>, e as curvas so fazem sentido juntas.

O QUE ELE CHECA, E POR QUE RECUSA.

Ponto de varredura so se compara com ponto de MESMA configuracao. Se uma
maquina rodou 20 jogos com 600 turnos e outra 28 jogos com 200, as duas curvas
estao medindo trabalho diferente por ponto, e sobrepor as duas produz um grafico
de speedup que parece certo e esta errado - ninguem olhando o PNG teria como
saber. Entao a juncao compara jogos, turnos e a serie de agentes, e para se
divergirem.

Com --force ele junta de todo jeito e grava o aviso no meta.txt do destino, para
que o grafico ao menos carregue a ressalva junto.

SO BIBLIOTECA PADRAO, como o plot_scaling.py: a maquina que junta pode nao ter
pip.
"""

import argparse
import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

COLUMNS = ["versao", "binario", "suscetiveis", "infectados", "agentes",
           "jogos", "turnos", "rep", "laco", "total", "status"]


def read_sweep(folder):
    """Le uma pasta de varredura. Devolve (linhas, resumo) ou (None, erro)."""
    csvPath = folder / "timings.csv"
    if not csvPath.is_file():
        return None, f"{csvPath} nao existe"

    rows = []
    with open(csvPath, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return None, f"{csvPath} sem as colunas {', '.join(missing)}"
        for row in reader:
            rows.append(row)

    if not rows:
        return None, f"{csvPath} nao tem nenhuma linha"

    good = [r for r in rows if r["status"].strip() in ("0", "ok")]
    if not good:
        # E o sintoma exato do bug de locale: a varredura rodou inteira, o CSV
        # tem linhas, e o status veio deslocado porque o awk imprimiu "0,123"
        # com virgula decimal num arquivo separado por virgula.
        return None, (f"{csvPath} tem {len(rows)} linhas mas nenhuma com status ok "
                      f"(status visto: {sorted({r['status'].strip() for r in rows})}). "
                      f"Se o separador decimal da maquina e virgula, o CSV saiu "
                      f"deslocado - rode a varredura de novo com o LC_ALL=C do "
                      f"bench_scaling.sh corrigido")

    summary = {
        "pasta": folder,
        "linhas": len(good),
        "descartadas": len(rows) - len(good),
        "versoes": sorted({r["versao"] for r in good}),
        "jogos": sorted({r["jogos"] for r in good}),
        "turnos": sorted({r["turnos"] for r in good}),
        "serie": tuple(sorted({int(r["suscetiveis"]) for r in good})),
        "reps": len({r["rep"] for r in good}),
    }
    return good, summary


def configuration_of(summary):
    """A chave que precisa bater entre varreduras."""
    return (tuple(summary["jogos"]), tuple(summary["turnos"]), summary["serie"])


def describe(summary):
    serie = summary["serie"]
    faixa = f"{serie[0]}..{serie[-1]} ({len(serie)} pontos)" if serie else "vazia"
    return (f"{summary['pasta'].name:<22} "
            f"jogos={','.join(summary['jogos']):<6} "
            f"turnos={','.join(summary['turnos']):<6} "
            f"serie={faixa:<28} "
            f"reps={summary['reps']}  "
            f"versoes={' '.join(summary['versoes'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Junta varreduras de maquinas diferentes num CSV so.",
        epilog="Depois: scripts/plot_scaling.py DESTINO")
    parser.add_argument("destino", type=Path)
    parser.add_argument("origens", type=Path, nargs="+")
    parser.add_argument("--force", action="store_true",
                        help="junta mesmo com configuracoes diferentes, gravando "
                             "o aviso no meta.txt do destino")
    args = parser.parse_args()

    if len(args.origens) < 1:
        print("Erro: informe ao menos uma pasta de origem.", file=sys.stderr)
        return 2

    allRows = []
    summaries = []
    for folder in args.origens:
        rows, summary = read_sweep(folder)
        if rows is None:
            print(f"Erro: {summary}", file=sys.stderr)
            return 1
        for row in rows:
            row["fonte"] = folder.name
        allRows.extend(rows)
        summaries.append(summary)

    print("varreduras lidas:")
    for summary in summaries:
        print("  " + describe(summary))
        if summary["descartadas"]:
            print(f"    ({summary['descartadas']} linhas sem status ok, descartadas)")

    # --- versao medida em mais de uma origem
    origemOf = defaultdict(set)
    for row in allRows:
        origemOf[row["versao"]].add(row["fonte"])
    duplicated = {v: s for v, s in origemOf.items() if len(s) > 1}
    if duplicated:
        print("\nAviso: versao medida em mais de uma origem (as duas entram no "
              "CSV, e a media do grafico vai misturar as maquinas):")
        for version, sources in sorted(duplicated.items()):
            print(f"  {version}: {', '.join(sorted(sources))}")

    # --- configuracoes incompativeis
    configurations = {configuration_of(s) for s in summaries}
    warning = None
    if len(configurations) > 1:
        message = ("configuracoes diferentes entre as varreduras: cada ponto "
                   "mediu trabalho diferente, e as curvas nao se comparam")
        if not args.force:
            print(f"\nErro: {message}.", file=sys.stderr)
            print("As varreduras precisam bater em jogos, turnos e serie de "
                  "agentes. Re-meca as que estao fora, ou use --force para "
                  "juntar com ressalva.", file=sys.stderr)
            return 1
        warning = message
        print(f"\nAVISO (--force): {message}.")

    destino = args.destino
    (destino / "plots").mkdir(parents=True, exist_ok=True)

    master = destino / "timings.csv"
    with open(master, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS + ["fonte"])
        writer.writeheader()
        writer.writerows(allRows)

    with open(destino / "meta.txt", "w", encoding="utf-8") as handle:
        handle.write("juncao de varreduras\n\n")
        if warning:
            handle.write(f"ATENCAO: {warning}.\n")
            handle.write("Juntado com --force. Os graficos desta pasta NAO sao "
                         "comparaveis entre versoes.\n\n")
        for summary in summaries:
            handle.write(describe(summary) + "\n")
        handle.write("\nmeta.txt de cada origem:\n")
        for folder in args.origens:
            origin = folder / "meta.txt"
            handle.write(f"\n--- {folder} ---\n")
            if origin.is_file():
                handle.write(origin.read_text(encoding="utf-8"))
            else:
                handle.write("(sem meta.txt)\n")

    # Os brutos de cada corrida vao junto: sem eles o CSV nao e reconstruivel se
    # alguem apagar as pastas de origem.
    for folder in args.origens:
        runs = folder / "runs"
        if runs.is_dir():
            shutil.copytree(runs, destino / "runs" / folder.name,
                            dirs_exist_ok=True)

    print(f"\n{len(allRows)} medicoes -> {master}")
    print(f"desenhe com: scripts/plot_scaling.py {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
