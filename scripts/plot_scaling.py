#!/usr/bin/env python3
"""Desenha tempo e speedup a partir de uma varredura do bench_scaling.sh.

Le PASTA/timings.csv e escreve PASTA/plots/*.svg. Nao mede nada: pode rodar
quantas vezes quiser, em outra maquina, meses depois - a medicao esta no CSV.

    scripts/plot_scaling.py bench/20260910-133038

Dois graficos, os dois em escala LOGARITMICA no eixo y:

    escalonamento.svg   tempo total por numero de agentes
    speedup.svg         tempo total da base dividido pelo de cada versao

O log existe porque a faixa medida cobre duas ordens de grandeza - a base leva
mais de um minuto onde a GPU leva segundos. Em escala linear as curvas rapidas
encostam todas no eixo e viram uma linha so; no log cada uma fica legivel, e a
distancia vertical entre duas curvas passa a ser a razao entre elas.

O tempo plotado e o TOTAL: parede do processo inteiro, incluindo ler os mapas e
subir para a GPU. A coluna "laco" (o cronometro interno do programa, so o laco
de turnos) continua no CSV para quem quiser isolar a computacao, mas fora do
grafico.

SO BIBLIOTECA PADRAO. Nada de matplotlib nem numpy: a maquina que roda a
varredura costuma ser justamente a que nao tem pip - um no de cluster com
python de sistema e mais nada. O grafico sai em SVG escrito na mao, que abre em
qualquer navegador, entra em LaTeX e escala sem perder qualidade.

Para PNG, se precisar:  rsvg-convert -w 1600 grafico.svg -o grafico.png

As versoes saem do proprio CSV, entao uma varredura so de CPU desenha duas
curvas e uma com as tres GPUs desenha seis, sem mudar nada aqui.
"""

import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Ordem e aparencia fixas por versao: com cor estavel entre varreduras, dois
# graficos de datas diferentes podem ser comparados lado a lado.
STYLE = {
    "base":       ("#444444", "circle", "base (1 core)"),
    "omp":        ("#1f77b4", "square", "OpenMP (agentes)"),
    "omp-jogos":  ("#9467bd", "diamond", "OpenMP (jogos)"),
    "omp-nested": ("#8c564b", "triangle", "OpenMP (aninhado)"),
    "mpi":        ("#ff7f0e", "circle", "MPI (jogos)"),
    "cuda":       ("#2ca02c", "triangle", "CUDA"),
    "rocm":       ("#d62728", "diamond", "ROCm"),
    "metal":      ("#e377c2", "square", "Metal"),
}
ORDER = ["base", "omp", "omp-jogos", "omp-nested", "mpi", "cuda", "rocm", "metal"]

# As duas medidas que o bench.sh grava. Vai para o grafico a "total"; a "laco"
# fica disponivel no CSV.
MEASURES = [("laco", "laco de turnos"), ("total", "processo inteiro")]
PLOTTED = "total"

WIDTH, HEIGHT = 1040, 600
PLOT_LEFT, PLOT_WIDTH = 104, 850
PLOT_TOP, PLOT_BOTTOM = 84, 452


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def load(path):
    """CSV -> ({medida: {versao: {agentes: [amostras]}}}, config)."""
    samples = {m: defaultdict(lambda: defaultdict(list)) for m, _ in MEASURES}
    config = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # O bench.sh grava o codigo de saida do processo, nao um rotulo:
            # "0" e sucesso. O "ok" fica aceito para CSV feito a mao.
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
    """[(agentes, media, desvio)] ordenado por numero de agentes."""
    result = []
    for agents in sorted(perVersion[version]):
        values = perVersion[version][agents]
        spread = statistics.pstdev(values) if len(values) > 1 else 0.0
        result.append((float(agents), statistics.fmean(values), spread))
    return result


def versions_in(perVersion):
    known = [v for v in ORDER if v in perVersion]
    return known + sorted(v for v in perVersion if v not in ORDER)


def style_of(version):
    return STYLE.get(version, ("#7f7f7f", "circle", version))


# ---------------------------------------------------------------------------
# SVG na mao
# ---------------------------------------------------------------------------

def escape(raw):
    return (str(raw).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def text(x, y, content, size=12, anchor="middle", color="#222222", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'font-family="DejaVu Sans, Helvetica, Arial, sans-serif" '
            f'font-weight="{weight}" text-anchor="{anchor}" '
            f'fill="{color}">{escape(content)}</text>')


def marker(shape, x, y, color, size=3.8):
    if shape == "square":
        return (f'<rect x="{x - size:.1f}" y="{y - size:.1f}" '
                f'width="{2 * size:.1f}" height="{2 * size:.1f}" fill="{color}"/>')
    if shape == "triangle":
        pts = (f"{x:.1f},{y - size * 1.2:.1f} {x - size:.1f},{y + size * .9:.1f} "
               f"{x + size:.1f},{y + size * .9:.1f}")
        return f'<polygon points="{pts}" fill="{color}"/>'
    if shape == "diamond":
        pts = (f"{x:.1f},{y - size * 1.3:.1f} {x + size * 1.1:.1f},{y:.1f} "
               f"{x:.1f},{y + size * 1.3:.1f} {x - size * 1.1:.1f},{y:.1f}")
        return f'<polygon points="{pts}" fill="{color}"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{color}"/>'


def decade_ticks(low, high):
    """Decadas 10^k cobrindo [low, high], com as marcas menores 2..9.

    Devolve (expoente de baixo, expoente de cima, maiores, menores). Os limites
    sao decadas inteiras para o eixo comecar e terminar num rotulo redondo.
    """
    if low <= 0:
        low = high / 100.0 if high > 0 else 0.01
    if high <= low:
        high = low * 10.0
    start = math.floor(math.log10(low))
    stop = math.ceil(math.log10(high))
    if stop == start:
        stop = start + 1
    major, minor = [], []
    for power in range(start, stop + 1):
        base = 10.0 ** power
        major.append(base)
        for digit in range(2, 10):
            value = base * digit
            if 10.0 ** start <= value <= 10.0 ** stop:
                minor.append(value)
    return start, stop, major, minor


def format_value(value):
    if value == 0:
        return "0"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"
    if value >= 1:
        return f"{value:g}"
    return f"{value:.10f}".rstrip("0").rstrip(".")


def format_agents(value):
    return f"{value / 1e6:.1f}M"


class Panel:
    """Area de plotagem: x linear em agentes, y logaritmico."""

    def __init__(self, series, ylabel):
        self.series = series          # [(label, color, shape, [(x, y, err)])]
        self.ylabel = ylabel
        self.left, self.right = PLOT_LEFT, PLOT_LEFT + PLOT_WIDTH
        self.top, self.bottom = PLOT_TOP, PLOT_BOTTOM

        xs = [p[0] for _, _, _, points in series for p in points]
        # O topo considera a barra de erro, senao ela sai cortada; o piso ignora
        # a barra, que num ponto rapido pode chegar a zero e nao cabe no log.
        highs = [p[1] + p[2] for _, _, _, points in series for p in points]
        lows = [p[1] for _, _, _, points in series for p in points if p[1] > 0]
        self.xMin, self.xMax = (min(xs), max(xs)) if xs else (0.0, 1.0)
        if self.xMin == self.xMax:
            self.xMax = self.xMin + 1.0
        self.powLow, self.powHigh, self.major, self.minor = decade_ticks(
            min(lows) if lows else 0.01, max(highs) if highs else 1.0)

    def px(self, value):
        return self.left + (value - self.xMin) / (self.xMax - self.xMin) * PLOT_WIDTH

    def py(self, value):
        # Zero e negativo nao existem no log: prendem no piso do eixo.
        floor = 10.0 ** self.powLow
        if value <= floor:
            value = floor
        fraction = ((math.log10(value) - self.powLow)
                    / (self.powHigh - self.powLow))
        return self.bottom - fraction * (self.bottom - self.top)

    def render(self):
        out = []

        # Menores primeiro, sem rotulo: dao a leitura de que a escala e log.
        for value in self.minor:
            y = self.py(value)
            out.append(f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.right}" '
                       f'y2="{y:.1f}" stroke="#f0f0f0" stroke-width="0.6"/>')
        for value in self.major:
            y = self.py(value)
            out.append(f'<line x1="{self.left}" y1="{y:.1f}" x2="{self.right}" '
                       f'y2="{y:.1f}" stroke="#cccccc" stroke-width="0.9"/>')
            out.append(text(self.left - 10, y + 4, format_value(value), 11,
                            anchor="end", color="#555555"))

        for step in range(7):
            value = self.xMin + (self.xMax - self.xMin) * step / 6
            x = self.px(value)
            out.append(f'<line x1="{x:.1f}" y1="{self.top}" x2="{x:.1f}" '
                       f'y2="{self.bottom}" stroke="#eeeeee" stroke-width="0.8"/>')
            out.append(text(x, self.bottom + 20, format_agents(value), 11,
                            color="#555555"))

        out.append(f'<rect x="{self.left}" y="{self.top}" width="{PLOT_WIDTH}" '
                   f'height="{self.bottom - self.top}" fill="none" '
                   f'stroke="#888888" stroke-width="1"/>')

        out.append(text((self.left + self.right) / 2, self.bottom + 46,
                        "agentes (suscetiveis + infectados)", 12))
        cx, cy = self.left - 66, (self.top + self.bottom) / 2
        out.append(f'<g transform="rotate(-90 {cx:.1f} {cy:.1f})">'
                   f'{text(cx, cy, self.ylabel, 12)}</g>')

        for _, color, shape, points in self.series:
            if not points:
                continue
            path = " ".join(f"{self.px(x):.1f},{self.py(y):.1f}"
                            for x, y, _ in points)
            out.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                       f'stroke-width="1.9"/>')
            for x, y, err in points:
                if err > 0:
                    top, bottom = self.py(y + err), self.py(max(y - err, 0.0))
                    px = self.px(x)
                    out.append(f'<line x1="{px:.1f}" y1="{top:.1f}" x2="{px:.1f}" '
                               f'y2="{bottom:.1f}" stroke="{color}" stroke-width="1"/>')
                    for edge in (top, bottom):
                        out.append(f'<line x1="{px - 3:.1f}" y1="{edge:.1f}" '
                                   f'x2="{px + 3:.1f}" y2="{edge:.1f}" '
                                   f'stroke="{color}" stroke-width="1"/>')
                out.append(marker(shape, self.px(x), self.py(y), color))
        return "\n".join(out)


def legend_row(series, y):
    """Legenda numa faixa horizontal abaixo do grafico.

    Fora da area de plotagem porque em escala linear nao ha canto livre
    garantido: no grafico de tempo a base ocupa o alto, no de speedup a GPU
    ocupa o alto e o OpenMP a base.
    """
    drawn = [s for s in series if s[3]]
    if not drawn:
        return ""
    slot = min(210.0, (PLOT_WIDTH - 20) / len(drawn))
    startX = PLOT_LEFT + (PLOT_WIDTH - slot * len(drawn)) / 2
    out = []
    for index, (label, color, shape, _) in enumerate(drawn):
        x = startX + index * slot
        out.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 22:.1f}" '
                   f'y2="{y:.1f}" stroke="{color}" stroke-width="1.9"/>')
        out.append(marker(shape, x + 11, y, color))
        out.append(text(x + 28, y + 4, label, 11, anchor="start"))
    return "\n".join(out)


def write_figure(path, title, subtitle, panel):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
           f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">\n'
           f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>\n'
           f'{text(WIDTH / 2, 32, title, 16, weight="bold")}\n'
           f'{text(WIDTH / 2, 54, subtitle, 11, color="#666666")}\n'
           f'{panel.render()}\n'
           f'{legend_row(panel.series, PLOT_BOTTOM + 82)}\n'
           f'</svg>\n')
    path.write_text(svg, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def context(config):
    return (f"{config.get('jogos', '?')} jogos, {config.get('turnos', '?')} turnos, "
            f"tempo total do processo, media +/- desvio, eixo y em escala log")


def plot_time(samples, config, outDir):
    perVersion = samples[PLOTTED]
    series = []
    for version in versions_in(perVersion):
        color, shape, label = style_of(version)
        series.append((label, color, shape, curve(perVersion, version)))
    if not any(points for _, _, _, points in series):
        return None
    return write_figure(outDir / "escalonamento.svg",
                        "Tempo total por numero de agentes",
                        context(config),
                        Panel(series, "tempo total (s, escala log)"))


def plot_speedup(samples, config, outDir):
    perVersion = samples[PLOTTED]
    if "base" not in perVersion:
        return None
    reference = {x: y for x, y, _ in curve(perVersion, "base")}
    series = []
    for version in versions_in(perVersion):
        color, shape, label = style_of(version)
        points = [(x, reference[x] / y, 0.0)
                  for x, y, _ in curve(perVersion, version)
                  if x in reference and y > 0]
        if points:
            series.append((label, color, shape, points))
    if len(series) < 2:
        return None
    return write_figure(outDir / "speedup.svg",
                        "Speedup sobre a versao sequencial",
                        context(config),
                        Panel(series, "speedup (vezes a base, escala log)"))


def summary(samples):
    perVersion = samples[PLOTTED]
    order = versions_in(perVersion)
    if not order:
        return
    allAgents = sorted({a for v in order for a in perVersion[v]})
    if not allAgents:
        return

    print("\nresumo (tempo total, media em s):")
    print(f"{'agentes':>12}" + "".join(f"{style_of(v)[2]:>19}" for v in order))
    for agents in (allAgents[0], allAgents[len(allAgents) // 2], allAgents[-1]):
        line = f"{agents:>12,}".replace(",", ".")
        for version in order:
            values = perVersion[version].get(agents)
            line += f"{statistics.fmean(values):>19.3f}" if values else f"{'-':>19}"
        print(line)

    if "base" in perVersion:
        largest = allAgents[-1]
        reference = perVersion["base"].get(largest)
        if reference:
            print("\nspeedup no maior ponto:")
            for version in order:
                if version == "base":
                    continue
                values = perVersion[version].get(largest)
                if values:
                    print(f"  {style_of(version)[2]:<19}"
                          f"{statistics.fmean(reference) / statistics.fmean(values):>8.1f}x")


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
    if not samples[PLOTTED]:
        print(f"Erro: {csvPath} nao tem nenhuma medicao com status ok.",
              file=sys.stderr)
        return 1

    plotDir = outDir / "plots"
    plotDir.mkdir(parents=True, exist_ok=True)

    written = [plot_time(samples, config, plotDir),
               plot_speedup(samples, config, plotDir)]
    summary(samples)

    print()
    for target in written:
        if target:
            print(f"grafico: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
