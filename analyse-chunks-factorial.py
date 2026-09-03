"""
analyse-chunks-factorial.py

Analisa os arquivos gerados pelo alvo "chunks-factorial" do Makefile
(data-chunks-factorial/dataOpt_<totalThreads>_<numBlocks>_<numThreads>.txt).

Diferente do thread-factorial, aqui a distribuicao de THREADS e' FIXA e o
que varia e' a particao do grid de BLOCOS em chunks (nxChunks x nyChunks x
nzChunks). Como todas as combinacoes de chunk compartilham o mesmo
numBlocks/numThreads, elas caem no MESMO arquivo dataOpt_*.txt -- por isso
o binario precisa imprimir, no bloco de configuracao:

    Chunks: nxChunks=..  nyChunks=..  nzChunks=..  numChunks=..
    Chunk size (blocos): NxNxN

Sem essas linhas o script nao consegue separar as configuracoes e aborta
com uma mensagem explicando o que falta.

METRICA
-------
Somente o TEMPO DE SIMULACAO. O tempo de generateCubes nao e' plotado:
ele e' setup, roda uma vez e nao depende da particao em chunks.
O skippedWarps tambem nao e' usado como escala de cor -- com threadsDim
fixo ele e' identico em todas as barras.

GRAFICOS GERADOS
----------------
O MESMO grafico de barras (media +- 3*desvio/sqrt(n), ordenado por tempo)
e' salvo varias vezes, mudando apenas o criterio de COR:

  ..._cor_numChunks.png       cor por numero total de chunks (x*y*z)
  ..._cor_nxChunks.png        cor por numero de chunks no eixo X
  ..._cor_nyChunks.png        cor por numero de chunks no eixo Y
  ..._cor_nzChunks.png        cor por numero de chunks no eixo Z
  ..._cor_blocosPorChunk.png  cor por blocos dentro de cada chunk

Mais um grafico de sintese:

  speedup_numchunks_....png   ganho relativo a numChunks=1, usando a
                              melhor particao de cada numChunks

EIXO Y
------
Nos graficos de barras o eixo Y NAO comeca em zero -- ele e' ajustado a
faixa dos dados para que diferencas pequenas fiquem visiveis. Isso amplia
visualmente as diferencas: ao usar essas figuras num texto, diga na
legenda que a origem foi suprimida. Use ZERO_BASE=True abaixo se quiser
o comportamento convencional.

Uso:
    python3 analyse-chunks-factorial.py <totalThreads> [numThreads]

    O segundo argumento e' opcional e filtra por numThreads, util se a
    pasta acumular varreduras com THREADSDIM diferentes.

Requer: numpy, matplotlib
"""
import os
import re
import sys
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data-chunks-factorial")

INVALID_COLOR = "#000000"
ERRORBAR_SIGMAS = 3.0

# False -> eixo Y ajustado a faixa dos dados (padrao, mostra detalhe).
# True  -> eixo Y comecando em zero (convencional, esconde detalhe).
ZERO_BASE = False
# Folga acima/abaixo da faixa dos dados, como fracao da amplitude.
ZOOM_PAD = 0.08
# ======================================================================

RECORD_START_RE = re.compile(r"===== t=")
FILENAME_RE = re.compile(r"^dataOpt_(\d+)_(\d+)_(\d+)\.txt$")

NUM = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"

FIELD_RES = {
    "t":               re.compile(r"===== t=(" + NUM + r") s, iter=(\d+), velFlux=(" + NUM + r") ====="),
    "numThreads":      re.compile(r"numThreads=(\d+)"),
    "numBlocks":       re.compile(r"numBlocks=(\d+)"),
    "blocksDim":       re.compile(r"Blocks:\s*nxBlock=(\d+)\s+nyBlock=(\d+)\s+nzBlock=(\d+)"),
    "chunksDim":       re.compile(r"Chunks:\s*nxChunks=(\d+)\s+nyChunks=(\d+)\s+nzChunks=(\d+)\s+numChunks=(\d+)"),
    "chunkSize":       re.compile(r"Chunk size \(blocos\):\s*(\d+)x(\d+)x(\d+)"),
    "threadsDim":      re.compile(r"Threads per block:\s*nxThreads=(\d+)\s+nyThreads=(\d+)\s+nzThreads=(\d+)"),
    "totalGrid":       re.compile(r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"),
    "totalThreads":    re.compile(r"totalThreads=(\d+)"),
    "validSimulation": re.compile(r"ValidSimulation=(-?\d+)"),
    "cubesInfo":       re.compile(r"Cubes Info:\s*numCubes=(\d+)\s+occupiedVolume=(" + NUM + r")%\s+skippedWarps=(" + NUM + r")%"),
    "gencubesTime":    re.compile(r"generateCubes time \(s\):\s*(" + NUM + r")"),
    "simTime":         re.compile(r"Total simulation time \(s\):\s*(" + NUM + r")"),
}


# ----------------------------------------------------------------------
# leitura
# ----------------------------------------------------------------------

def resolve_data_paths(totalThreads, numThreadsFilter=None):
    pattern = os.path.join(DATA_DIR, f"dataOpt_{totalThreads}_*_*.txt")
    paths = glob.glob(pattern)

    found = []
    for path in paths:
        m = FILENAME_RE.match(os.path.basename(path))
        if not m:
            continue
        file_total, numBlocks, numThreads = (int(v) for v in m.groups())
        if file_total != totalThreads:
            continue
        if numThreadsFilter is not None and numThreads != numThreadsFilter:
            continue
        found.append((path, numBlocks, numThreads))

    if not found:
        print(
            f"Erro: nenhum arquivo encontrado para totalThreads={totalThreads}"
            + (f", numThreads={numThreadsFilter}" if numThreadsFilter else "")
            + f" (padrao: '{pattern}').\n"
            f"Verifique se a varredura rodou e se os arquivos estao em '{DATA_DIR}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    found.sort(key=lambda item: item[2])
    return found


def split_records(text):
    starts = [m.start() for m in RECORD_START_RE.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def parse_record(record, fallback_numBlocks=None, fallback_numThreads=None):
    m_sim_time = FIELD_RES["simTime"].search(record)
    if not m_sim_time:
        return None
    m_chunks = FIELD_RES["chunksDim"].search(record)
    if not m_chunks:
        return "NO_CHUNK_FIELD"

    cx, cy, cz, ncTotal = (int(v) for v in m_chunks.groups())

    m_t = FIELD_RES["t"].search(record)
    m_nt = FIELD_RES["numThreads"].search(record)
    m_nb = FIELD_RES["numBlocks"].search(record)
    m_bd = FIELD_RES["blocksDim"].search(record)
    m_cs = FIELD_RES["chunkSize"].search(record)
    m_td = FIELD_RES["threadsDim"].search(record)
    m_grid = FIELD_RES["totalGrid"].search(record)
    m_total = FIELD_RES["totalThreads"].search(record)
    m_cubes = FIELD_RES["cubesInfo"].search(record)
    m_gencubes = FIELD_RES["gencubesTime"].search(record)
    m_valid = FIELD_RES["validSimulation"].search(record)

    xT, yT, zT = (int(v) for v in m_grid.groups()) if m_grid else (None, None, None)
    total_cells = xT * yT * zT if m_grid else None

    nx, ny, nz = (int(v) for v in m_td.groups()) if m_td else (None, None, None)
    bx, by, bz = (int(v) for v in m_bd.groups()) if m_bd else (None, None, None)
    sx, sy, sz = (int(v) for v in m_cs.groups()) if m_cs else (None, None, None)

    return {
        "nxChunks": cx, "nyChunks": cy, "nzChunks": cz, "numChunks": ncTotal,
        "chunkSizeX": sx, "chunkSizeY": sy, "chunkSizeZ": sz,
        "blocksPerChunk": (sx * sy * sz) if m_cs else None,
        "nxBlock": bx, "nyBlock": by, "nzBlock": bz,
        "nxThreads": nx, "nyThreads": ny, "nzThreads": nz,
        "numThreads": int(m_nt.group(1)) if m_nt else fallback_numThreads,
        "numBlocks": int(m_nb.group(1)) if m_nb else fallback_numBlocks,
        "totalCells": total_cells,
        "totalThreadsDeclared": int(m_total.group(1)) if m_total else None,
        "validSimulation": (int(m_valid.group(1)) != 0) if m_valid else True,
        "hasValidField": m_valid is not None,
        "iter": int(m_t.group(2)) if m_t else None,
        "time": float(m_sim_time.group(1)),
        "gencubesTime": float(m_gencubes.group(1)) if m_gencubes else None,
        "numCubes": int(m_cubes.group(1)) if m_cubes else None,
        "occupiedVolumePct": float(m_cubes.group(2)) if m_cubes else None,
        "skippedWarpsPct": float(m_cubes.group(3)) if m_cubes else None,
    }


def parse_data(text, fallback_numBlocks=None, fallback_numThreads=None):
    entries, legacy = [], 0
    for record in split_records(text):
        parsed = parse_record(record, fallback_numBlocks, fallback_numThreads)
        if parsed == "NO_CHUNK_FIELD":
            legacy += 1
        elif parsed is not None:
            entries.append(parsed)
    return entries, legacy


def aggregate_by_chunks(entries):
    """Aglutina as REPEAT repeticoes da MESMA configuracao de chunks."""
    groups = {}
    for e in entries:
        key = (e["numBlocks"], e["numThreads"],
               e["nxChunks"], e["nyChunks"], e["nzChunks"])
        groups.setdefault(key, []).append(e)

    stats = []
    for (numBlocks, numThreads, cx, cy, cz), elist in groups.items():
        times = np.array([e["time"] for e in elist])
        gencubes = np.array([e["gencubesTime"] for e in elist if e["gencubesTime"] is not None])
        skipped = np.array([e["skippedWarpsPct"] for e in elist if e["skippedWarpsPct"] is not None])

        efficiencies = np.array([
            (e["totalCells"] * e["iter"] / e["time"])
            for e in elist
            if e["totalCells"] is not None and e["iter"] is not None and e["time"] > 0
        ])

        n_invalid = sum(1 for e in elist if not e["validSimulation"])
        first = elist[0]

        stats.append({
            "numBlocks": numBlocks, "numThreads": numThreads,
            "nxChunks": cx, "nyChunks": cy, "nzChunks": cz,
            "numChunks": cx * cy * cz,
            "chunkSizeX": first["chunkSizeX"], "chunkSizeY": first["chunkSizeY"],
            "chunkSizeZ": first["chunkSizeZ"],
            "blocksPerChunk": first["blocksPerChunk"],
            "nxThreads": first["nxThreads"], "nyThreads": first["nyThreads"],
            "nzThreads": first["nzThreads"],
            "n": len(elist),
            "valid": n_invalid == 0,
            "nInvalid": n_invalid,
            "meanTime": float(np.mean(times)),
            "stdTime": float(np.std(times)),
            "minTime": float(np.min(times)),
            "maxTime": float(np.max(times)),
            "meanEfficiency": float(np.mean(efficiencies)) if efficiencies.size else None,
            "stdEfficiency": float(np.std(efficiencies)) if efficiencies.size else None,
            "meanSkippedWarpsPct": float(np.mean(skipped)) if skipped.size else None,
            "meanGencubesTime": float(np.mean(gencubes)) if gencubes.size else None,
            "stdGencubesTime": float(np.std(gencubes)) if gencubes.size else None,
        })

    stats.sort(key=lambda s: s["meanTime"])
    return stats


# ----------------------------------------------------------------------
# saida em texto
# ----------------------------------------------------------------------

def print_table(stats):
    header = (f"{'chunksDim':>12} {'nChunks':>8} {'blocos/chunk':>13} {'nBlocks':>8} "
              f"{'nThreads':>9} {'n':>3} {'valid':>7}   "
              f"{'tempo medio (s)':>16} {'desvio (s)':>12}   {'min (s)':>10} {'max (s)':>10}")
    print(header)
    print("-" * len(header))
    for s in stats:
        flag = "sim" if s["valid"] else f"NAO({s['nInvalid']})"
        dim = f'{s["nxChunks"]}x{s["nyChunks"]}x{s["nzChunks"]}'
        bpc = s["blocksPerChunk"] if s["blocksPerChunk"] is not None else -1
        print(
            f"{dim:>12} {s['numChunks']:>8} {bpc:>13} {s['numBlocks']:>8} "
            f"{s['numThreads']:>9} {s['n']:>3} {flag:>7}   "
            f"{s['meanTime']:>16.6f} {s['stdTime']:>12.6f}   "
            f"{s['minTime']:>10.6f} {s['maxTime']:>10.6f}"
        )


def save_csv(stats, outpath):
    fields = ["numBlocks", "numThreads", "nxChunks", "nyChunks", "nzChunks", "numChunks",
              "chunkSizeX", "chunkSizeY", "chunkSizeZ", "blocksPerChunk",
              "nxThreads", "nyThreads", "nzThreads", "n", "valid", "nInvalid",
              "meanTime", "stdTime", "minTime", "maxTime",
              "meanEfficiency", "stdEfficiency",
              "meanSkippedWarpsPct", "meanGencubesTime", "stdGencubesTime"]
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in stats:
            writer.writerow({k: s.get(k) for k in fields})
    print(f"  -> {outpath}")


# ----------------------------------------------------------------------
# graficos
# ----------------------------------------------------------------------

_ORDERED_CMAP = plt.colormaps["jet_r"]


def build_discrete_color_map(values):
    """Cor SOLIDA por valor, ordenada: menor valor vermelho, maior azul."""
    uniq = sorted(set(values))
    n = len(uniq)
    positions = [0.0] if n == 1 else [i / (n - 1) for i in range(n)]
    return {v: _ORDERED_CMAP(p) for v, p in zip(uniq, positions)}


def _mean_error_bars(stats):
    return np.array([
        (ERRORBAR_SIGMAS * s["stdTime"] / np.sqrt(s["n"])) if s.get("n") else 0.0
        for s in stats
    ])


def _bar_labels(stats):
    labels = []
    for s in stats:
        bpc = s["blocksPerChunk"]
        label = f'{s["nxChunks"]}×{s["nyChunks"]}×{s["nzChunks"]}  (nC={s["numChunks"]}'
        label += f', {bpc} bl/chunk)' if bpc is not None else ')'
        if not s["valid"]:
            label += " [invalido]"
        labels.append(label)
    return labels


def _style_invalid_ticks(ax, stats):
    for tick, s in zip(ax.get_xticklabels(), stats):
        if not s["valid"]:
            tick.set_color(INVALID_COLOR)
            tick.set_fontweight("bold")


def _invalid_legend_handle():
    return mpatches.Patch(facecolor=INVALID_COLOR, edgecolor="black",
                          label="ValidSimulation=0 (invalido)")


def _apply_ylim(ax, means, errs):
    """Ajusta o eixo Y a faixa dos dados (nao comeca em zero), a menos que
    ZERO_BASE esteja ligado. Retorna True se a origem foi suprimida."""
    if ZERO_BASE:
        return False
    lo = float(np.min(means - errs))
    hi = float(np.max(means + errs))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1e-9) * 0.1
    pad = span * ZOOM_PAD
    bottom = max(0.0, lo - pad)
    if bottom <= 0.0:
        return False
    ax.set_ylim(bottom, hi + pad)
    return True


def plot_time_bars(stats, totalThreads, color_key, color_title, outpath):
    """Grafico de barras do TEMPO DE SIMULACAO por particao de chunks,
    sempre ordenado pelo proprio tempo. Muda apenas o criterio de cor."""
    usable = [s for s in stats if s.get("meanTime") is not None and s.get(color_key) is not None]
    if not usable:
        print(f"  ({color_key} ausente nos dados -- grafico nao gerado)")
        return

    ordered = sorted(usable, key=lambda s: s["meanTime"])
    labels = _bar_labels(ordered)
    means = np.array([s["meanTime"] for s in ordered])
    errs = _mean_error_bars(ordered)
    color_values = [s[color_key] for s in ordered]
    ns = [s["n"] for s in ordered]

    color_map = build_discrete_color_map(color_values)
    bar_colors = [color_map[v] if s["valid"] else INVALID_COLOR
                  for v, s in zip(color_values, ordered)]
    n_invalid = sum(1 for s in ordered if not s["valid"])

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.45), 6))
    x = np.arange(len(ordered))

    ax.bar(x, means, yerr=errs, capsize=3, color=bar_colors, alpha=0.9,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    _style_invalid_ticks(ax, ordered)

    zoomed = _apply_ylim(ax, means, errs)

    ax.set_xlabel("Particao em chunks (nxChunks × nyChunks × nzChunks)")
    ax.set_ylabel(f"Tempo de simulacao (s) — media ± {ERRORBAR_SIGMAS:.0f}·desvio/√n")
    title = (f"Tempo de simulacao por particao de chunks — cor por {color_title}\n"
             f"totalThreads={totalThreads}  "
             f"(repeticoes por barra: min={min(ns)}, max={max(ns)}; ordenado por tempo)")
    if zoomed:
        title += "  [eixo Y nao comeca em zero]"
    if n_invalid:
        title += f"\n{n_invalid} combinacao(oes) invalida(s) em preto"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=color_map[v], label=f"{color_title}={v}")
               for v in sorted(color_map)]
    if n_invalid:
        handles.append(_invalid_legend_handle())
    ax.legend(handles=handles, title=color_title, fontsize=8,
              title_fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_speedup(stats, totalThreads, outpath):
    """Speedup relativo a numChunks=1 (a versao sem chunking)."""
    valid = [s for s in stats if s["valid"]]
    base = [s for s in valid if s["numChunks"] == 1]
    if not base:
        print("  (sem baseline numChunks=1 -- speedup nao gerado)")
        return
    t0 = min(s["meanTime"] for s in base)

    best = {}
    for s in valid:
        nc = s["numChunks"]
        if nc not in best or s["meanTime"] < best[nc]["meanTime"]:
            best[nc] = s

    ncs = sorted(best)
    speedups = [t0 / best[nc]["meanTime"] for nc in ncs]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#27ae60" if sp >= 1.0 else "#c0392b" for sp in speedups]
    ax.bar([str(n) for n in ncs], speedups, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)

    for i, sp in enumerate(speedups):
        ax.annotate(f"{sp:.3f}", (i, sp), textcoords="offset points",
                    xytext=(0, 4 if sp >= 1 else -12), ha="center", fontsize=8)

    lo, hi = min(speedups), max(speedups)
    span = max(hi - lo, 1e-6)
    ax.set_ylim(min(lo, 1.0) - span * 0.15, max(hi, 1.0) + span * 0.15)

    ax.set_xlabel("numChunks")
    ax.set_ylabel("Speedup relativo a numChunks=1")
    ax.set_title("Ganho do pipeline por chunks contra a versao sem chunking\n"
                 f"totalThreads={totalThreads}  (baseline = {t0:.6f} s)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  -> {outpath}")


# ----------------------------------------------------------------------

def main():
    if len(sys.argv) not in (2, 3):
        print(
            f"Uso: python3 {os.path.basename(sys.argv[0])} <totalThreads> [numThreads]\n\n"
            f"Exemplo: python3 {os.path.basename(sys.argv[0])} 4194304\n"
            f"         python3 {os.path.basename(sys.argv[0])} 4194304 1024\n"
            f"(procura {DATA_DIR}/dataOpt_<totalThreads>_*_*.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        totalThreads = int(sys.argv[1])
        numThreadsFilter = int(sys.argv[2]) if len(sys.argv) == 3 else None
    except ValueError:
        print("Erro: os argumentos devem ser inteiros.", file=sys.stderr)
        sys.exit(1)

    files = resolve_data_paths(totalThreads, numThreadsFilter)
    print(f"totalThreads={totalThreads}: {len(files)} arquivo(s) encontrado(s)")
    for path, numBlocks, numThreads in files:
        print(f"  - {os.path.basename(path)}  (numBlocks={numBlocks}, numThreads={numThreads})")

    all_entries, legacy_total = [], 0
    for path, numBlocks, numThreads in files:
        with open(path, "r") as f:
            text = f.read()
        entries, legacy = parse_data(text, numBlocks, numThreads)
        all_entries.extend(entries)
        legacy_total += legacy

    print(f"\n{len(all_entries)} execucoes parseadas")
    if legacy_total:
        print(f"  {legacy_total} registro(s) sem a linha 'Chunks:' foram IGNORADOS "
              f"(formato antigo do binario)")

    if not all_entries:
        print(
            "\nNenhuma execucao com informacao de chunks encontrada.\n"
            "O binario precisa imprimir, no bloco de configuracao:\n"
            "    Chunks: nxChunks=..  nyChunks=..  nzChunks=..  numChunks=..\n"
            "    Chunk size (blocos): NxNxN\n"
            "tanto no printf do terminal quanto no fprintf do dataFile.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_invalid_runs = sum(1 for e in all_entries if not e["validSimulation"])
    print(f"  {n_invalid_runs} execucao(oes) com ValidSimulation=0 (plotadas em preto)")

    stats = aggregate_by_chunks(all_entries)
    n_invalid_groups = sum(1 for s in stats if not s["valid"])
    print(f"{len(stats)} particoes distintas -- {n_invalid_groups} invalida(s)\n")

    print_table(stats)
    print()

    tag = f"total{totalThreads}" + (f"_nt{numThreadsFilter}" if numThreadsFilter else "")

    def out(name):
        return os.path.join(DATA_DIR, f"{name}_{tag}.png")

    save_csv(stats, os.path.join(DATA_DIR, f"chunks_{tag}.csv"))

    # O MESMO grafico de tempo, salvo uma vez por criterio de cor.
    color_keys = [
        ("numChunks",      "numChunks"),
        ("nxChunks",       "nxChunks"),
        ("nyChunks",       "nyChunks"),
        ("nzChunks",       "nzChunks"),
        ("blocksPerChunk", "blocosPorChunk"),
    ]
    for key, title in color_keys:
        plot_time_bars(stats, totalThreads, key, title,
                       out(f"tempo_sim_por_chunks_cor_{title}"))

    plot_speedup(stats, totalThreads, out("speedup_numchunks"))


if __name__ == "__main__":
    main()