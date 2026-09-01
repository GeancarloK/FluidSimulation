"""
analyse-factorial-cut.py

Analisa os arquivos gerados pelo alvo "thread-factorial" do Makefile
(data-thread-factorial/dataOpt_<totalThreads>_<numBlocks>_<numThreads>.txt).

Agora o script recebe apenas TOTALTHREADS (numBlocks * numThreads) e
busca TODOS os arquivos dataOpt_<totalThreads>_*_*.txt dentro de
data-thread-factorial/ -- ou seja, todas as combinacoes de
numBlocks/numThreads que resultam nesse mesmo total. Cada arquivo
acumula varias execucoes (uma por combinacao valida de
XDIV x YDIV x ZDIV cujo produto bate com numThreads, repetida REPEAT
vezes).

O agrupamento das barras do grafico e' pela distribuicao de threads em
x/y/z (nxThreads, nyThreads, nzThreads) dentro de cada arquivo. Varias
repeticoes (REPEAT) da MESMA distribuicao sao aglutinadas em media e
desvio padrao do tempo de simulacao.

Cada barra e' colorida de acordo com o numThreads do arquivo de onde
ela veio -- cores DISCRETAS (uma cor fixa por valor de numThreads),
sem degrade.

Uso:
    py analyze_thread_factorial.py <totalThreads>

Exemplo:
    py analyze_thread_factorial.py 1048576
    (pega dataOpt_1048576_1024_1024.txt, dataOpt_1048576_2048_512.txt, etc.)

Os arquivos sao procurados em data-thread-factorial/, na mesma pasta
deste script (ajuste DATA_DIR abaixo se o seu Makefile usar outro nome
de pasta).

Requer: numpy, matplotlib
    py -m pip install numpy matplotlib
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
DATA_DIR = os.path.join(SCRIPT_DIR, "data-thread-factorial")

# Sufixo adicionado a TODOS os arquivos de saida (CSV e PNGs) deste
# script, para diferencia-los dos gerados pela versao sem o corte de
# nxThreads < 8.
OUTPUT_SUFFIX = "-cut"
# ======================================================================


# Cada execucao anexada ao arquivo comeca com essa linha (escrita pelo
# main.cu a cada chamada de run()). Usamos ela como delimitador de
# registro em vez de depender de uma linha de tracos, que pode nao
# estar presente na versao do dataFile que voce esta usando.
RECORD_START_RE = re.compile(r"===== t=")

# Nome de arquivo: dataOpt_<totalThreads>_<numBlocks>_<numThreads>.txt
FILENAME_RE = re.compile(r"^dataOpt_(\d+)_(\d+)_(\d+)\.txt$")

FIELD_RES = {
    "t":               re.compile(r"===== t=([\d.]+) s, iter=(\d+), velFlux=([\d.]+) ====="),
    "numThreads":      re.compile(r"numThreads=(\d+)"),
    "numBlocks":       re.compile(r"numBlocks=(\d+)"),
    "threadsDim":      re.compile(r"Threads per block:\s*nxThreads=(\d+)\s+nyThreads=(\d+)\s+nzThreads=(\d+)"),
    "totalGrid":       re.compile(r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"),
    "totalThreads":    re.compile(r"totalThreads=(\d+)"),
    "cubesInfo":       re.compile(r"Cubes Info:\s*numCubes=(\d+)\s+occupiedVolume=([\d.]+)%\s+skippedWarps=([\d.]+)%"),
    "gencubesTime":    re.compile(r"generateCubes time \(s\):\s*([\d.]+)"),
    "simTime":         re.compile(r"Total simulation time \(s\):\s*([\d.]+)"),
}


def resolve_data_paths(totalThreads):
    """Encontra todos os arquivos data-thread-factorial/dataOpt_<totalThreads>_*_*.txt
    (uma combinacao numBlocks x numThreads diferente por arquivo, todas
    com o mesmo total) e retorna [(path, numBlocks, numThreads), ...]
    ordenado por numThreads."""
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
        found.append((path, numBlocks, numThreads))

    if not found:
        print(
            f"Erro: nenhum arquivo encontrado para totalThreads={totalThreads} "
            f"(padrao procurado: '{pattern}').\n"
            f"Verifique se totalThreads esta certo e se os arquivos estao dentro "
            f"de '{DATA_DIR}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    found.sort(key=lambda item: item[1])  # ordena por numBlocks (== por numThreads tambem, ja que o total e' fixo)
    return found


def split_records(text):
    """Quebra o texto em um registro por execucao, usando a linha
    '===== t=...' como marcador de inicio. Robusto independente de
    haver ou nao uma linha de tracos entre execucoes."""
    starts = [m.start() for m in RECORD_START_RE.finditer(text)]
    if not starts:
        return []
    starts.append(len(text))
    return [text[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]


def parse_record(record, fallback_numBlocks=None, fallback_numThreads=None):
    """Extrai os campos de um unico registro (uma execucao). Retorna
    None se os campos minimos (distribuicao de threads + tempo) nao
    forem encontrados -- protege contra um registro truncado no fim
    do arquivo (ex.: execucao interrompida no meio da escrita)."""
    m_threads_dim = FIELD_RES["threadsDim"].search(record)
    m_sim_time = FIELD_RES["simTime"].search(record)
    if not m_threads_dim or not m_sim_time:
        return None

    nx, ny, nz = (int(v) for v in m_threads_dim.groups())

    m_t = FIELD_RES["t"].search(record)
    m_nt = FIELD_RES["numThreads"].search(record)
    m_nb = FIELD_RES["numBlocks"].search(record)
    m_grid = FIELD_RES["totalGrid"].search(record)
    m_total = FIELD_RES["totalThreads"].search(record)
    m_cubes = FIELD_RES["cubesInfo"].search(record)
    m_gencubes = FIELD_RES["gencubesTime"].search(record)

    xThreads, yThreads, zThreads = (int(v) for v in m_grid.groups()) if m_grid else (None, None, None)
    total_cells = xThreads * yThreads * zThreads if m_grid else None

    # Se o registro nao trouxer numBlocks/numThreads explicitamente,
    # usa os valores extraidos do nome do arquivo.
    numThreads = int(m_nt.group(1)) if m_nt else fallback_numThreads
    numBlocks = int(m_nb.group(1)) if m_nb else fallback_numBlocks

    return {
        "nxThreads": nx, "nyThreads": ny, "nzThreads": nz,
        "numThreads": numThreads,
        "numBlocks": numBlocks,
        "totalCells": total_cells,
        "totalThreadsDeclared": int(m_total.group(1)) if m_total else None,
        "iter": int(m_t.group(2)) if m_t else None,
        "time": float(m_sim_time.group(1)),
        "gencubesTime": float(m_gencubes.group(1)) if m_gencubes else None,
        "numCubes": int(m_cubes.group(1)) if m_cubes else None,
        "occupiedVolumePct": float(m_cubes.group(2)) if m_cubes else None,
        "skippedWarpsPct": float(m_cubes.group(3)) if m_cubes else None,
    }


def parse_data(text, fallback_numBlocks=None, fallback_numThreads=None):
    entries = []
    for record in split_records(text):
        parsed = parse_record(record, fallback_numBlocks, fallback_numThreads)
        if parsed is not None:
            entries.append(parsed)
    return entries


def aggregate_by_distribution(entries):
    """Aglutina execucoes com a MESMA combinacao (numBlocks, numThreads,
    nxThreads, nyThreads, nzThreads) -- ou seja, as REPEAT repeticoes de
    uma mesma linha do sweep XDIV x YDIV x ZDIV, dentro de um mesmo
    arquivo, viram um unico grupo com media/desvio."""
    groups = {}
    for e in entries:
        key = (e["numBlocks"], e["numThreads"], e["nxThreads"], e["nyThreads"], e["nzThreads"])
        groups.setdefault(key, []).append(e)

    stats = []
    for (numBlocks, numThreads, nx, ny, nz), elist in groups.items():
        times = np.array([e["time"] for e in elist])

        efficiencies = np.array([
            (e["totalCells"] * e["iter"] / e["time"])
            for e in elist
            if e["totalCells"] is not None and e["iter"] is not None and e["time"] > 0
        ])

        skipped = np.array([
            e["skippedWarpsPct"] for e in elist if e["skippedWarpsPct"] is not None
        ])
        occupied = np.array([
            e["occupiedVolumePct"] for e in elist if e["occupiedVolumePct"] is not None
        ])
        gencubes = np.array([
            e["gencubesTime"] for e in elist if e["gencubesTime"] is not None
        ])

        stats.append({
            "numBlocks": numBlocks, "numThreads": numThreads,
            "nxThreads": nx, "nyThreads": ny, "nzThreads": nz,
            "n": len(elist),
            "meanTime": float(np.mean(times)),
            "stdTime": float(np.std(times)),
            "minTime": float(np.min(times)),
            "maxTime": float(np.max(times)),
            "meanEfficiency": float(np.mean(efficiencies)) if efficiencies.size else None,
            "stdEfficiency": float(np.std(efficiencies)) if efficiencies.size else None,
            "meanSkippedWarpsPct": float(np.mean(skipped)) if skipped.size else None,
            "meanOccupiedVolumePct": float(np.mean(occupied)) if occupied.size else None,
            "meanGencubesTime": float(np.mean(gencubes)) if gencubes.size else None,
            "stdGencubesTime": float(np.std(gencubes)) if gencubes.size else None,
        })

    # Ordena tudo pelo tempo de simulacao medio, do menor para o maior.
    stats.sort(key=lambda s: s["meanTime"])
    return stats


def print_table(stats):
    header = (f"{'numBlocks':>9} {'numThreads':>10} {'nx':>5} {'ny':>5} {'nz':>5} {'n':>3}   "
              f"{'tempo medio (s)':>16} {'desvio (s)':>12}   {'min (s)':>10} {'max (s)':>10}")
    print(header)
    print("-" * len(header))
    for s in stats:
        print(
            f"{s['numBlocks']:>9} {s['numThreads']:>10} "
            f"{s['nxThreads']:>5} {s['nyThreads']:>5} {s['nzThreads']:>5} {s['n']:>3}   "
            f"{s['meanTime']:>16.6f} {s['stdTime']:>12.6f}   "
            f"{s['minTime']:>10.6f} {s['maxTime']:>10.6f}"
        )


def save_csv(stats, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "numBlocks", "numThreads", "nxThreads", "nyThreads", "nzThreads", "n",
            "meanTime", "stdTime", "minTime", "maxTime",
            "meanEfficiency", "stdEfficiency",
            "meanSkippedWarpsPct", "meanOccupiedVolumePct",
            "meanGencubesTime", "stdGencubesTime",
        ])
        writer.writeheader()
        for s in stats:
            writer.writerow(s)
    print(f"  -> {outpath}")


# Paleta discreta, mas ORDENADA por valor: o menor valor da chave
# (numThreads, nxThreads, nyThreads ou nzThreads) sempre fica vermelho,
# o maior sempre fica azul, com laranja/amarelo/verde nos valores
# intermediarios -- cada barra ainda tem uma cor SOLIDA (sem degrade
# dentro da barra), so' a escolha da cor segue essa escala.
_ORDERED_CMAP = plt.colormaps["jet_r"]


def build_discrete_color_map(values):
    uniq = sorted(set(values))
    n = len(uniq)
    if n == 1:
        positions = [0.0]
    else:
        positions = [i / (n - 1) for i in range(n)]
    return {v: _ORDERED_CMAP(p) for v, p in zip(uniq, positions)}


def _bar_labels_and_ns(stats):
    labels = [f'{s["nxThreads"]}×{s["nyThreads"]}×{s["nzThreads"]} (nB={s["numBlocks"]}, nT={s["numThreads"]})'
              for s in stats]
    ns = [s["n"] for s in stats]
    return labels, ns


def _apply_desanchored_ylim(ax, means, stds):
    """Em vez de deixar o eixo Y comecar em 0 (padrao de ax.bar), ancora
    o eixo perto da faixa real dos dados -- assim as diferencas entre
    barras ficam mais visiveis."""
    lower = float(np.min(means - stds))
    upper = float(np.max(means + stds))
    span = upper - lower
    margin = span * 0.08 if span > 0 else upper * 0.08 if upper else 1.0
    ax.set_ylim(lower - margin, upper + margin)


def plot_by_distribution_categorical(stats, totalThreads, color_key, color_title,
                                      mean_key, std_key, metric_label, outpath):
    """Grafico de uma metrica de tempo (mean_key/std_key) por distribuicao
    de threads, com cada barra colorida (cor SOLIDA, discreta) de acordo
    com o campo `color_key` do proprio grupo (ex.: 'numThreads',
    'nxThreads', 'nyThreads' ou 'nzThreads'). As barras ficam sempre
    ordenadas pela propria metrica, do menor para o maior tempo -- a cor
    apenas identifica a categoria, sem reagrupar a ordem."""
    usable = [s for s in stats if s[mean_key] is not None]
    if not usable:
        print(f"  ({mean_key} nao encontrado nos dados -- grafico nao gerado)")
        return

    ordered = sorted(usable, key=lambda s: s[mean_key])

    labels, ns = _bar_labels_and_ns(ordered)
    means = np.array([s[mean_key] for s in ordered])
    stds = np.array([s[std_key] for s in ordered])
    color_values = [s[color_key] for s in ordered]

    color_map = build_discrete_color_map(color_values)
    bar_colors = [color_map[v] for v in color_values]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.4), 6))
    x = np.arange(len(ordered))

    ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, alpha=0.9,
           edgecolor="black", linewidth=0.4)
    _apply_desanchored_ylim(ax, means, stds)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Distribuicao de threads (nxThreads × nyThreads × nzThreads)")
    ax.set_ylabel(f"{metric_label} — media ± desvio padrao")
    ax.set_title(
        f"{metric_label} por distribuicao de threads — cor por {color_title}\n"
        f"totalThreads={totalThreads}  "
        f"(repeticoes por barra: min={min(ns)}, max={max(ns)}; ordenado por {metric_label.lower()})"
    )
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=color_map[v], label=f"{color_title}={v}")
        for v in sorted(color_map)
    ]
    ax.legend(handles=legend_handles, title=color_title,
              fontsize=8, title_fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  -> {outpath}")


def plot_by_distribution_skipped_warps(stats, totalThreads, mean_key, std_key, metric_label, outpath):
    """Mesmo grafico (para a metrica mean_key/std_key), mas cada barra e'
    colorida em DEGRADE (colormap continuo) de acordo com o
    skippedWarpsPct medio do grupo, com uma barra de cores (colorbar) ao
    lado. Ordenado pela propria metrica, do menor para o maior."""
    usable = [s for s in stats if s[mean_key] is not None and s["meanSkippedWarpsPct"] is not None]
    if not usable:
        print(f"  ({mean_key} ou skippedWarpsPct nao encontrado nos dados -- grafico nao gerado)")
        return

    ordered = sorted(usable, key=lambda s: s[mean_key])
    labels, ns = _bar_labels_and_ns(ordered)
    means = np.array([s[mean_key] for s in ordered])
    stds = np.array([s[std_key] for s in ordered])
    skipped_vals = np.array([s["meanSkippedWarpsPct"] for s in ordered])

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=skipped_vals.min(), vmax=skipped_vals.max())
    bar_colors = cmap(norm(skipped_vals))

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.4), 6))
    x = np.arange(len(ordered))

    ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, alpha=0.95,
           edgecolor="black", linewidth=0.4)
    _apply_desanchored_ylim(ax, means, stds)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Distribuicao de threads (nxThreads × nyThreads × nzThreads)")
    ax.set_ylabel(f"{metric_label} — media ± desvio padrao")
    ax.set_title(
        f"{metric_label} por distribuicao de threads — degrade por skippedWarps (%)\n"
        f"totalThreads={totalThreads}  "
        f"(repeticoes por barra: min={min(ns)}, max={max(ns)}; ordenado por {metric_label.lower()})"
    )
    ax.grid(True, axis="y", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("skippedWarps medio (%)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"  -> {outpath}")


def main():
    if len(sys.argv) != 2:
        print(
            f"Uso: py {os.path.basename(sys.argv[0])} <totalThreads>\n\n"
            f"Exemplo: py {os.path.basename(sys.argv[0])} 1048576\n"
            f"(procura todos os arquivos {DATA_DIR}/dataOpt_<totalThreads>_*_*.txt)",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        totalThreads = int(sys.argv[1])
    except ValueError:
        print("Erro: totalThreads deve ser um inteiro.", file=sys.stderr)
        sys.exit(1)

    files = resolve_data_paths(totalThreads)
    print(f"totalThreads={totalThreads}: {len(files)} arquivo(s) encontrado(s)")
    for path, numBlocks, numThreads in files:
        print(f"  - {os.path.basename(path)}  (numBlocks={numBlocks}, numThreads={numThreads})")

    all_entries = []
    for path, numBlocks, numThreads in files:
        with open(path, "r") as f:
            text = f.read()
        entries = parse_data(text, fallback_numBlocks=numBlocks, fallback_numThreads=numThreads)
        all_entries.extend(entries)

    print(f"\n{len(all_entries)} execucoes parseadas no total")

    if not all_entries:
        print("Nenhuma execucao valida encontrada nos arquivos.", file=sys.stderr)
        sys.exit(1)

    stats = aggregate_by_distribution(all_entries)

    # Descarta combinacoes com nxThreads < 8 (pouco relevantes para a
    # analise) -- afeta tabela, CSV e todos os graficos abaixo.
    before = len(stats)
    stats = [s for s in stats if s["nxThreads"] >= 8]
    removed = before - len(stats)
    if removed:
        print(f"Descartadas {removed} combinacao(oes) com nxThreads < 8")

    if not stats:
        print("Nenhuma combinacao restante apos o filtro de nxThreads >= 8.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(stats)} combinacoes distintas (numBlocks x numThreads x distribuicao)\n")

    print_table(stats)

    save_csv(stats, os.path.join(
        SCRIPT_DIR, f"data-thread-factorial/distribuicao_total{totalThreads}{OUTPUT_SUFFIX}.csv"))

    print()

    color_keys = [
        ("numThreads", "numThreads", "numThreads"),
        ("nxThreads", "nxThreads", "nxThreads"),
        ("nyThreads", "nyThreads", "nyThreads"),
        ("nzThreads", "nzThreads", "nzThreads"),
    ]

    # Para os graficos de generateCubes, descarta tambem combinacoes com
    # numThreads < 64 (o tempo de gerar os cubos so' e' relevante a
    # partir de blocos maiores) -- essa restricao vale SO' para o
    # generateCubes, os graficos de tempo de simulacao usam `stats` sem
    # esse corte extra.
    stats_gencubes = [s for s in stats if s["numThreads"] >= 64]
    removed_gencubes = len(stats) - len(stats_gencubes)
    if removed_gencubes:
        print(f"generateCubes: descartadas {removed_gencubes} combinacao(oes) com numThreads < 64")

    # Um conjunto de 5 graficos (4 categoricos + 1 degrade) para cada
    # metrica de tempo: simulacao e generateCubes.
    metrics = [
        (stats, "meanTime", "stdTime", "Tempo de simulacao (s)",
         os.path.join(SCRIPT_DIR, "data-thread-factorial",
                       f"tempo_sim_por_distribuicao_total{totalThreads}{OUTPUT_SUFFIX}")),
        (stats_gencubes, "meanGencubesTime", "stdGencubesTime", "Tempo de generateCubes (s)",
         os.path.join(SCRIPT_DIR, "data-thread-factorial",
                       f"tempo_gencubes_por_distribuicao_total{totalThreads}{OUTPUT_SUFFIX}")),
    ]

    for metric_stats, mean_key, std_key, metric_label, out_prefix in metrics:
        if not metric_stats:
            print(f"  ({metric_label}: nenhuma combinacao restante apos os filtros -- graficos nao gerados)")
            continue

        for color_key, color_title, suffix in color_keys:
            plot_by_distribution_categorical(
                metric_stats, totalThreads, color_key, color_title,
                mean_key, std_key, metric_label,
                f"{out_prefix}_cor_{suffix}.png",
            )

        plot_by_distribution_skipped_warps(
            metric_stats, totalThreads,
            mean_key, std_key, metric_label,
            f"{out_prefix}_cor_skippedWarps.png",
        )


if __name__ == "__main__":
    main()