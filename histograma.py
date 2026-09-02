#!/usr/bin/env python3
"""
histograma.py

Recebe um numero TOTAL de celulas (== totalThreads) e uma distribuicao
de threads por bloco em x/y/z, junta TODAS as execucoes do
data-thread-factorial que batem com essa configuracao EXATA e plota um
histograma do tempo de simulacao (e do tempo de generateCubes, quando
disponivel).

Uso:
    python3 histograma.py <totalCells> <nxThreads> <nyThreads> <nzThreads>

Exemplo:
    python3 histograma.py 1048576 512 2 1
    -> procura data-thread-factorial/dataOpt_1048576_*_*.txt, filtra as
       execucoes com nxThreads=512 nyThreads=2 nzThreads=1, calcula o
       histograma e salva
       data-thread-factorial/histograma_total1048576_threads512x2x1.png

Cada barra do histograma tem largura (max - min) / n, onde n e' o
numero de execucoes daquela config -- ou seja, exatamente n barras
entre o menor e o maior tempo.

O PNG e' salvo na pasta data-thread-factorial/ e, logo depois, a figura
e' exibida na tela (plt.show()).

O parser dos arquivos e' reaproveitado de analyse-thread-factorial.py
(mesmo formato de dataFile, mesmo tratamento de ValidSimulation).

Requer: numpy, matplotlib  (mesmas dependencias dos scripts analyse-*)
"""
import os
import sys
import importlib.util

import numpy as np
import matplotlib

# Tenta garantir um backend interativo ANTES de importar o pyplot, para
# que o plt.show() do final abra uma janela mesmo se o ambiente tiver
# forcado um backend nao-interativo ("Agg").
if matplotlib.get_backend().lower() == "agg":
    for _bk in ("QtAgg", "TkAgg", "GTK3Agg", "MacOSX", "Qt5Agg"):
        try:
            matplotlib.use(_bk, force=True)
            break
        except Exception:
            continue

import matplotlib.pyplot as plt  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Cor das execucoes marcadas como invalidas (ValidSimulation=0), igual
# aos scripts analyse-thread-factorial*.
INVALID_COLOR = "#000000"
VALID_COLOR = "#4C72B0"

def _make_bin_edges(values):
    """Bordas das fatias do histograma.

    Largura de cada barra = (max - min) / n, onde n e' o numero de
    execucoes -- ou seja, EXATAMENTE n barras cobrindo [min, max].
    Ex.: min=0.90, max=0.95, n=15  ->  15 barras de (0.95-0.90)/15.

    Se todos os tempos forem iguais (max == min) abre uma faixa minima
    artificial so' para conseguir desenhar uma barra.
    """
    n = max(int(values.size), 1)
    vmin = float(values.min())
    vmax = float(values.max())

    if vmax <= vmin:
        pad = max(abs(vmin) * 1e-9, 1e-9)
        return np.linspace(vmin - pad, vmax + pad, 2)

    return np.linspace(vmin, vmax, n + 1)


def _load_analyse_module():
    """Reaproveita o parser de analyse-thread-factorial.py (o nome com
    hifens nao pode ser importado com 'import' direto)."""
    path = os.path.join(SCRIPT_DIR, "analyse-thread-factorial.py")
    if not os.path.isfile(path):
        print(f"Erro: nao encontrei {path} -- necessario para ler os dados.",
              file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("analyse_thread_factorial", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv):
    prog = os.path.basename(argv[0])
    if len(argv) != 5:
        print(
            f"Uso: python3 {prog} <totalCells> <nxThreads> <nyThreads> <nzThreads>\n\n"
            f"Exemplo: python3 {prog} 1048576 512 2 1",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        total = int(argv[1])
        nx, ny, nz = int(argv[2]), int(argv[3]), int(argv[4])
    except ValueError:
        print("Erro: todos os argumentos devem ser inteiros.", file=sys.stderr)
        sys.exit(1)
    if min(total, nx, ny, nz) <= 0:
        print("Erro: todos os argumentos devem ser positivos.", file=sys.stderr)
        sys.exit(1)
    return total, nx, ny, nz


def collect_entries(atf, total, nx, ny, nz):
    """Le todos os dataOpt_<total>_*_*.txt e devolve as execucoes cuja
    distribuicao de threads por bloco e' exatamente (nx, ny, nz)."""
    files = atf.resolve_data_paths(total)  # [(path, numBlocks, numThreads), ...] (ou sys.exit)
    matched = []
    contributing = []
    for path, numBlocks, numThreads in files:
        with open(path, "r") as f:
            text = f.read()
        entries = atf.parse_data(text, fallback_numBlocks=numBlocks,
                                 fallback_numThreads=numThreads)
        hit = [e for e in entries
               if (e["nxThreads"], e["nyThreads"], e["nzThreads"]) == (nx, ny, nz)]
        if hit:
            contributing.append((os.path.basename(path), len(hit)))
            matched.extend(hit)
    return matched, contributing


def _hist_axis(ax, values, invalid_mask, label, unit="s"):
    """Desenha um histograma de `values` no eixo `ax`, separando em
    barras empilhadas as execucoes validas / invalidas e marcando
    media / mediana / faixa de +-1 desvio."""
    values = np.asarray(values, dtype=float)
    n = values.size
    edges = _make_bin_edges(values)
    nbins = len(edges) - 1
    # com muitas fatias, a borda branca some com a barra -- so desenha
    # contorno quando as fatias sao largas o bastante.
    edgekw = dict(edgecolor="white", linewidth=0.4) if nbins <= 60 else dict(edgecolor="none")

    if invalid_mask.any() and (~invalid_mask).any():
        ax.hist([values[~invalid_mask], values[invalid_mask]], bins=edges,
                stacked=True, color=[VALID_COLOR, INVALID_COLOR],
                label=["ValidSimulation=1", "ValidSimulation=0"], **edgekw)
    else:
        color = INVALID_COLOR if invalid_mask.all() else VALID_COLOR
        ax.hist(values, bins=edges, color=color, **edgekw)

    # "rug": um tracinho no eixo x para cada execucao, ajuda a enxergar
    # os valores quando ficam muito colados.
    ax.plot(values, np.full(n, -0.02), "|", color="#222222",
            markeredgewidth=1.0, markersize=10, clip_on=False,
            transform=ax.get_xaxis_transform())

    ax.set_xlim(edges[0], edges[-1])

    mean = float(np.mean(values))
    std = float(np.std(values))
    median = float(np.median(values))
    ax.axvspan(mean - std, mean + std, color="red", alpha=0.08,
               label=f"media +- desvio ({std:.6f})")
    ax.axvline(mean, color="red", linewidth=1.6, label=f"media = {mean:.6f}")
    ax.axvline(median, color="orange", linewidth=1.3, linestyle="--",
               label=f"mediana = {median:.6f}")

    bin_width = float(edges[1] - edges[0])
    ax.set_xlabel(f"{label} ({unit})")
    ax.set_ylabel("Frequencia (nº de execucoes)")
    ax.set_title(
        f"{label}\n"
        f"n={n}  media={mean:.6f}  desvio={std:.6f}  "
        f"min={values.min():.6f}  max={values.max():.6f}\n"
        f"{nbins} fatias de {bin_width:.2e} {unit}"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)


def main():
    total, nx, ny, nz = parse_args(sys.argv)
    atf = _load_analyse_module()

    entries, contributing = collect_entries(atf, total, nx, ny, nz)
    numThreadsBlock = nx * ny * nz

    print(f"totalCells={total}  threads/bloco = {nx}x{ny}x{nz} (= {numThreadsBlock})")

    if not entries:
        print(
            f"Erro: nenhuma execucao com nxThreads={nx} nyThreads={ny} "
            f"nzThreads={nz} encontrada para totalCells={total} em "
            f"{atf.DATA_DIR}.",
            file=sys.stderr,
        )
        sys.exit(1)

    for fname, cnt in contributing:
        print(f"  - {fname}: {cnt} execucao(oes)")
    print(f"{len(entries)} execucao(oes) no total")

    times = np.array([e["time"] for e in entries], dtype=float)
    invalid_mask = np.array([not e["validSimulation"] for e in entries])
    n_invalid = int(invalid_mask.sum())
    if n_invalid:
        print(f"  {n_invalid} execucao(oes) com ValidSimulation=0 (em preto no histograma)")

    gencubes_pairs = [(e["gencubesTime"], not e["validSimulation"])
                      for e in entries if e["gencubesTime"] is not None]

    npanels = 2 if gencubes_pairs else 1
    fig, axes = plt.subplots(1, npanels, figsize=(7 * npanels, 5), squeeze=False)
    axes = axes[0]

    _hist_axis(axes[0], times, invalid_mask, "Tempo de simulacao")

    if gencubes_pairs:
        gvals = np.array([p[0] for p in gencubes_pairs], dtype=float)
        gmask = np.array([p[1] for p in gencubes_pairs])
        _hist_axis(axes[1], gvals, gmask, "Tempo de generateCubes")

    numBlocks_txt = ",".join(str(b) for b in sorted({e["numBlocks"] for e in entries}))
    fig.suptitle(
        f"Histograma — totalCells={total}, threads/bloco {nx}x{ny}x{nz} "
        f"(numThreads={numThreadsBlock}, numBlocks={numBlocks_txt})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    outname = f"histograma_total{total}_threads{nx}x{ny}x{nz}.png"
    outpath = os.path.join(atf.DATA_DIR, outname)
    fig.savefig(outpath, dpi=200)
    print(f"\n  -> {outpath}")

    # exibe na tela depois de salvar
    try:
        plt.show()
    except Exception as exc:  # backend sem GUI disponivel, etc.
        print(f"(nao foi possivel abrir a janela do matplotlib: {exc})",
              file=sys.stderr)


if __name__ == "__main__":
    main()
