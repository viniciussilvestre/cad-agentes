#!/usr/bin/env python3
"""Desenha tempo e speedup a partir de uma varredura do bench_scaling.sh.

Le PASTA/timings.csv e escreve PASTA/plots/*.png. Nao mede nada: pode rodar
quantas vezes quiser, em outra maquina, meses depois - a medicao esta no CSV.

    scripts/plot_scaling.py bench/20260909-143000

As versoes saem do proprio CSV, entao uma varredura so de CPU desenha duas
curvas e uma com as duas GPUs desenha quatro, sem mudar nada aqui.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ordem e aparencia fixas por versao: com cor estavel entre varreduras, dois
# graficos de datas diferentes podem ser comparados lado a lado.
STYLE = {
    "base": ("#444444", "o", "base (1 core)"),
    "omp":  ("#1f77b4", "s", "OpenMP"),
    "cuda": ("#2ca02c", "^", "CUDA"),
    "rocm": ("#d62728", "D", "ROCm"),
}
ORDER = ["base", "omp", "cuda", "rocm"]

# As duas medidas que o bench.sh grava, e a diferenca importa: "laco" e so o
# laco de turnos, "total" inclui ler os mapas e subir para a GPU. Numa GPU
# rapida a carga chega a dominar o total, e um grafico so do laco esconderia
# isso.
MEASURES = [("laco", "laco de turnos"), ("total", "processo inteiro")]


def load(path):
    """CSV -> {medida: {versao: {agentes: [amostras]}}} e o cabecalho da config."""
    samples = {m: defaultdict(lambda: defaultdict(list)) for m, _ in MEASURES}
    config = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # O bench.sh grava o codigo de saida do processo, nao um rotulo:
            # "0" e sucesso. O "ok" fica aceito para o caso de alguem gerar o
            # CSV a mao.
            if row.get("status", "").strip() not in ("0", "ok"):
                continue
            version = row["versao"]
            agents = int(row["agentes"])
            config.setdefault("jogos", row["jogos"])
            config.setdefault("turnos", row["turnos"])
            for measure, _ in MEASURES:
                value = row[measure].strip()
                if value and value != "-":
                    samples[measure][version][agents].append(float(value))
    return samples, config


def curve(perVersion, version):
    """(agentes, media, desvio) ordenados por numero de agentes."""
    points = sorted(perVersion[version])
    mean = np.array([np.mean(perVersion[version][a]) for a in points])
    std = np.array([np.std(perVersion[version][a]) for a in points])
    return np.array(points, dtype=float), mean, std


def versions_in(perVersion):
    return [v for v in ORDER if v in perVersion] + \
           sorted(v for v in perVersion if v not in ORDER)


def style_of(version):
    return STYLE.get(version, ("#7f7f7f", "x", version))


def annotate(axis, config, subtitle):
    axis.set_xlabel("agentes (suscetiveis + infectados)")
    axis.set_title(subtitle)
    axis.grid(True, which="both", alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=9)


def plot_time(samples, config, outDir):
    figure, axes = plt.subplots(1, len(MEASURES), figsize=(13, 5), constrained_layout=True)
    for axis, (measure, subtitle) in zip(np.atleast_1d(axes), MEASURES):
        perVersion = samples[measure]
        for version in versions_in(perVersion):
            color, marker, label = style_of(version)
            x, mean, std = curve(perVersion, version)
            axis.errorbar(x, mean, yerr=std, color=color, marker=marker,
                          markersize=4, linewidth=1.4, capsize=3, label=label)
        axis.set_yscale("log")
        axis.set_ylabel("tempo (s, escala log)")
        annotate(axis, config, subtitle)
    figure.suptitle(
        f"Escalonamento em numero de agentes "
        f"({config.get('jogos', '?')} jogos, {config.get('turnos', '?')} turnos, "
        f"media +/- desvio)")
    target = outDir / "escalonamento.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def plot_speedup(samples, config, outDir):
    figure, axes = plt.subplots(1, len(MEASURES), figsize=(13, 5), constrained_layout=True)
    drew = False
    for axis, (measure, subtitle) in zip(np.atleast_1d(axes), MEASURES):
        perVersion = samples[measure]
        if "base" not in perVersion:
            axis.text(0.5, 0.5, "sem a versao base:\nnao ha referencia de speedup",
                      ha="center", va="center", transform=axis.transAxes, fontsize=10)
            annotate(axis, config, subtitle)
            continue
        baseX, baseMean, _ = curve(perVersion, "base")
        reference = dict(zip(baseX, baseMean))
        for version in versions_in(perVersion):
            if version == "base":
                continue
            color, marker, label = style_of(version)
            x, mean, _ = curve(perVersion, version)
            shared = [(a, t) for a, t in zip(x, mean) if a in reference]
            if not shared:
                continue
            xs = np.array([a for a, _ in shared])
            ys = np.array([reference[a] / t for a, t in shared])
            axis.plot(xs, ys, color=color, marker=marker, markersize=4,
                      linewidth=1.4, label=label)
            drew = True
        axis.axhline(1.0, color="#444444", linewidth=1.0, linestyle="--",
                     label="base (1x)")
        axis.set_yscale("log")
        axis.set_ylabel("speedup sobre a base (escala log)")
        annotate(axis, config, subtitle)
    if not drew:
        plt.close(figure)
        return None
    figure.suptitle(
        f"Speedup sobre a versao sequencial "
        f"({config.get('jogos', '?')} jogos, {config.get('turnos', '?')} turnos)")
    target = outDir / "speedup.png"
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def summary(samples, config):
    perVersion = samples["laco"]
    order = versions_in(perVersion)
    if not order:
        return
    allAgents = sorted({a for v in order for a in perVersion[v]})
    if not allAgents:
        return
    smallest, largest = allAgents[0], allAgents[-1]

    print("\nresumo (laco de turnos, media em s):")
    header = f"{'agentes':>10}" + "".join(f"{style_of(v)[2]:>16}" for v in order)
    print(header)
    for agents in (smallest, largest):
        line = f"{agents:>10,}".replace(",", ".")
        for version in order:
            values = perVersion[version].get(agents)
            line += f"{np.mean(values):>16.3f}" if values else f"{'-':>16}"
        print(line)

    if "base" in perVersion:
        print("\nspeedup no maior ponto:")
        reference = perVersion["base"].get(largest)
        if reference:
            for version in order:
                if version == "base":
                    continue
                values = perVersion[version].get(largest)
                if values:
                    print(f"  {style_of(version)[2]:<14} "
                          f"{np.mean(reference) / np.mean(values):>8.1f}x")


def main():
    if len(sys.argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    outDir = Path(sys.argv[1])
    csvPath = outDir / "timings.csv"
    if not csvPath.is_file():
        print(f"Erro: {csvPath} nao existe.", file=sys.stderr)
        return 1

    samples, config = load(csvPath)
    if not any(samples[m] for m, _ in MEASURES):
        print(f"Erro: {csvPath} nao tem nenhuma medicao com status ok.", file=sys.stderr)
        return 1

    plotDir = outDir / "plots"
    plotDir.mkdir(parents=True, exist_ok=True)

    written = [plot_time(samples, config, plotDir),
               plot_speedup(samples, config, plotDir)]
    summary(samples, config)

    print()
    for target in written:
        if target:
            print(f"grafico: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
