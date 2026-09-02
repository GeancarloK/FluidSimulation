"""
analyze_thread_factorial.py

Analisa os arquivos gerados pelo alvo "thread-factorial" do Makefile
(data-thread-factorial/dataOpt_<totalThreads>_<numBlocks>_<numThreads>.txt).

Agora o script recebe apenas TOTALTHREADS (numBlocks * numThreads) e
busca TODOS os arquivos dataOpt_<totalThreads>_*_*.txt dentro de
data-thread-factorial/ -- ou seja, todas as combinacoes de
numBlocks/numThreads que resultam nesse mesmo total. Cada arquivo
acumula varias execucoes (uma por combinacao valida de
XDIV x YDIV x ZDIV cujo produto bate com numThreads, repetida REPEAT
vezes).

FORMATO DE ENTRADA
------------------
Cada execucao e' anexada ao dataFile com o cabecalho:

    ===== t=... s, iter=..., velFlux=... =====
    === Grid Configuration ===
    Domain (m): length=...  width=...  height=...
    numThreads=...  numBlocks=...

    Blocks: nxBlock=...  nyBlock=...  nzBlock=...
    Block size (m): dxBlock=...  dyBlock=...  dzBlock=...

    Threads per block: nxThreads=...  nyThreads=...  nzThreads=...
    Thread size (m): dxThreads=...  dyThreads=...  dzThreads=...

    Total threads: xThreads=...  yThreads=...  zThreads=...
    totalThreads=...

    ValidSimulation=...
    Cubes Info: numCubes=... occupiedVolume=...% skippedWarps=...%
    generateCubes time (s): ...

    Total simulation time (s): ...

O marcador de inicio de registro e' a linha "===== t=...". O campo
ValidSimulation e' lido de cada execucao; arquivos no formato antigo
(sem o campo) sao tratados como validos.

SIMULACOES INVALIDAS
--------------------
Um grupo (barra) e' considerado INVALIDO se qualquer uma de suas
repeticoes tiver ValidSimulation=0. Barras invalidas sao pintadas de
PRETO em todos os graficos (ignorando a escala de cor / o degrade),
recebem o sufixo "[invalido]" no rotulo do eixo X e a coluna `valid` no
CSV.

O agrupamento das barras do grafico e' pela distribuicao de threads em
x/y/z (nxThreads, nyThreads, nzThreads) dentro de cada arquivo. Varias
repeticoes (REPEAT) da MESMA distribuicao sao aglutinadas em media e
desvio padrao do tempo de simulacao. A barra de erro exibida no grafico
NAO e' o desvio padrao cru: e' 3 * desvio / sqrt(n) (3x o erro padrao
da media), onde n e' o numero de execucoes daquela configuracao exata.

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

# Cor usada nas barras cuja simulacao veio marcada como invalida
# (ValidSimulation=0).
INVALID_COLOR = "#000000"
# ======================================================================


# Cada execucao anexada ao arquivo comeca com a linha "===== t=..."
# (escrita pelo main.cu a cada chamada de run(), antes do cabecalho
# "=== Grid Configuration ==="). Usamos ela como delimitador de registro
# em vez de depender de uma linha de tracos entre execucoes.
RECORD_START_RE = re.compile(r"===== t=")

# Nome de arquivo: dataOpt_<totalThreads>_<numBlocks>_<numThreads>.txt
FILENAME_RE = re.compile(r"^dataOpt_(\d+)_(\d+)_(\d+)\.txt$")

# Numero em ponto flutuante tolerante a nan/inf/notacao cientifica.
NUM = r"[-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"

FIELD_RES = {
    "t":               re.compile(r"===== t=(" + NUM + r") s, iter=(\d+), velFlux=(" + NUM + r") ====="),
    "numThreads":      re.compile(r"numThreads=(\d+)"),
    "numBlocks":       re.compile(r"numBlocks=(\d+)"),
    "threadsDim":      re.compile(r"Threads per block:\s*nxThreads=(\d+)\s+nyThreads=(\d+)\s+nzThreads=(\d+)"),
    "totalGrid":       re.compile(r"Total threads:\s*xThreads=(\d+)\s+yThreads=(\d+)\s+zThreads=(\d+)"),
    "totalThreads":    re.compile(r"totalThreads=(\d+)"),
    "validSimulation": re.compile(r"ValidSimulation=(-?\d+)"),
    "cubesInfo":       re.compile(r"Cubes Info:\s*numCubes=(\d+)\s+occupiedVolume=(" + NUM + r")%\s+skippedWarps=(" + NUM + r")%"),
    "gencubesTime":    re.compile(r"generateCubes time \(s\):\s*(" + NUM + r")"),
    "simTime":         re.compile(r"Total simulation time \(s\):\s*(" + NUM + r")"),
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
    m_valid = FIELD_RES["validSimulation"].search(record)

    xThreads, yThreads, zThreads = (int(v) for v in m_grid.groups()) if m_grid else (None, None, None)
    total_cells = xThreads * yThreads * zThreads if m_grid else None

    # Se o registro nao trouxer numBlocks/numThreads explicitamente,
    # usa os valores extraidos do nome do arquivo.
    numThreads = int(m_nt.group(1)) if m_nt else fallback_numThreads
    numBlocks = int(m_nb.group(1)) if m_nb else fallback_numBlocks

    # ValidSimulation=0 -> simulacao invalida. Arquivos no formato
    # antigo (sem o campo) sao tratados como validos.
    validSimulation = (int(m_valid.group(1)) != 0) if m_valid else True

    return {
        "nxThreads": nx, "nyThreads": ny, "nzThreads": nz,
        "numThreads": numThreads,
        "numBlocks": numBlocks,
        "totalCells": total_cells,
        "totalThreadsDeclared": int(m_total.group(1)) if m_total else None,
        "validSimulation": validSimulation,
        "hasValidField": m_valid is not None,
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
    arquivo, viram um unico grupo com media/desvio.

    O grupo herda a flag de validade: basta UMA repeticao com
    ValidSimulation=0 para o grupo inteiro ser marcado como invalido."""
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

        n_invalid = sum(1 for e in elist if not e["validSimulation"])

        stats.append({
            "numBlocks": numBlocks, "numThreads": numThreads,
            "nxThreads": nx, "nyThreads": ny, "nzThreads": nz,
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
            "meanOccupiedVolumePct": float(np.mean(occupied)) if occupied.size else None,
            "meanGencubesTime": float(np.mean(gencubes)) if gencubes.size else None,
            "stdGencubesTime": float(np.std(gencubes)) if gencubes.size else None,
        })

    # Ordena tudo pelo tempo de simulacao medio, do menor para o maior.
    stats.sort(key=lambda s: s["meanTime"])
    return stats


def print_table(stats):
    header = (f"{'numBlocks':>9} {'numThreads':>10} {'nx':>5} {'ny':>5} {'nz':>5} {'n':>3} {'valid':>6}   "
              f"{'tempo medio (s)':>16} {'desvio (s)':>12}   {'min (s)':>10} {'max (s)':>10}")
    print(header)
    print("-" * len(header))
    for s in stats:
        flag = "sim" if s["valid"] else f"NAO({s['nInvalid']})"
        print(
            f"{s['numBlocks']:>9} {s['numThreads']:>10} "
            f"{s['nxThreads']:>5} {s['nyThreads']:>5} {s['nzThreads']:>5} {s['n']:>3} {flag:>6}   "
            f"{s['meanTime']:>16.6f} {s['stdTime']:>12.6f}   "
            f"{s['minTime']:>10.6f} {s['maxTime']:>10.6f}"
        )


def save_csv(stats, outpath):
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "numBlocks", "numThreads", "nxThreads", "nyThreads", "nzThreads", "n",
            "valid", "nInvalid",
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


# Barra de erro exibida nos graficos: 3 * desvio_padrao / sqrt(n),
# ou seja, 3x o erro padrao da media, onde n e' o numero de execucoes
# daquela configuracao EXATA (numBlocks, numThreads, nxThreads,
# nyThreads, nzThreads) -- o campo "n" do grupo agregado.
ERRORBAR_SIGMAS = 3.0


def _mean_error_bars(stats, std_key):
    """3 * desvio / sqrt(n) por grupo (erro padrao da media multiplicado
    por ERRORBAR_SIGMAS). n <= 0 -> barra de erro zero."""
    return np.array([
        (ERRORBAR_SIGMAS * s[std_key] / np.sqrt(s["n"])) if s.get("n") else 0.0
        for s in stats
    ])


def _bar_labels_and_ns(stats):
    labels = []
    for s in stats:
        label = (f'{s["nxThreads"]}×{s["nyThreads"]}×{s["nzThreads"]} '
                 f'(nB={s["numBlocks"]}, nT={s["numThreads"]})')
        if not s["valid"]:
            label += " [invalido]"
        labels.append(label)
    ns = [s["n"] for s in stats]
    return labels, ns


def _style_invalid_ticks(ax, stats):
    """Deixa os rotulos das barras invalidas em negrito/preto para casar
    com a barra preta."""
    for tick, s in zip(ax.get_xticklabels(), stats):
        if not s["valid"]:
            tick.set_color(INVALID_COLOR)
            tick.set_fontweight("bold")


def _invalid_legend_handle():
    return mpatches.Patch(facecolor=INVALID_COLOR, edgecolor="black",
                          label="ValidSimulation=0 (invalido)")


def plot_by_distribution_categorical(stats, totalThreads, color_key, color_title,
                                      mean_key, std_key, metric_label, outpath):
    """Grafico de uma metrica de tempo (mean_key/std_key) por distribuicao
    de threads, com cada barra colorida (cor SOLIDA, discreta) de acordo
    com o campo `color_key` do proprio grupo (ex.: 'numThreads',
    'nxThreads', 'nyThreads' ou 'nzThreads'). As barras ficam sempre
    ordenadas pela propria metrica, do menor para o maior tempo -- a cor
    apenas identifica a categoria, sem reagrupar a ordem.

    Grupos com ValidSimulation=0 sao pintados de preto, ignorando a
    escala de cor."""
    usable = [s for s in stats if s[mean_key] is not None]
    if not usable:
        print(f"  ({mean_key} nao encontrado nos dados -- grafico nao gerado)")
        return

    ordered = sorted(usable, key=lambda s: s[mean_key])

    labels, ns = _bar_labels_and_ns(ordered)
    means = np.array([s[mean_key] for s in ordered])
    errs = _mean_error_bars(ordered, std_key)
    color_values = [s[color_key] for s in ordered]

    color_map = build_discrete_color_map(color_values)
    bar_colors = [color_map[v] if s["valid"] else INVALID_COLOR
                  for v, s in zip(color_values, ordered)]
    n_invalid = sum(1 for s in ordered if not s["valid"])

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.4), 6))
    x = np.arange(len(ordered))

    ax.bar(x, means, yerr=errs, capsize=3, color=bar_colors, alpha=0.9,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    _style_invalid_ticks(ax, ordered)
    ax.set_xlabel("Distribuicao de threads (nxThreads × nyThreads × nzThreads)")
    ax.set_ylabel(f"{metric_label} — media ± 3·desvio/√n")
    title = (
        f"{metric_label} por distribuicao de threads — cor por {color_title}\n"
        f"totalThreads={totalThreads}  "
        f"(repeticoes por barra: min={min(ns)}, max={max(ns)}; ordenado por {metric_label.lower()})"
    )
    if n_invalid:
        title += f"\n{n_invalid} combinacao(oes) invalida(s) em preto"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    legend_handles = [
        mpatches.Patch(color=color_map[v], label=f"{color_title}={v}")
        for v in sorted(color_map)
    ]
    if n_invalid:
        legend_handles.append(_invalid_legend_handle())
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
    lado. Ordenado pela propria metrica, do menor para o maior.

    Grupos com ValidSimulation=0 sao pintados de preto e ficam fora da
    normalizacao do degrade."""
    usable = [s for s in stats if s[mean_key] is not None and s["meanSkippedWarpsPct"] is not None]
    if not usable:
        print(f"  ({mean_key} ou skippedWarpsPct nao encontrado nos dados -- grafico nao gerado)")
        return

    ordered = sorted(usable, key=lambda s: s[mean_key])
    labels, ns = _bar_labels_and_ns(ordered)
    means = np.array([s[mean_key] for s in ordered])
    errs = _mean_error_bars(ordered, std_key)
    skipped_vals = np.array([s["meanSkippedWarpsPct"] for s in ordered])

    valid_mask = np.array([s["valid"] for s in ordered])
    n_invalid = int((~valid_mask).sum())

    # A escala do degrade considera apenas os grupos validos, para que
    # uma simulacao invalida nao distorca as cores das demais.
    scale_vals = skipped_vals[valid_mask] if valid_mask.any() else skipped_vals
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=float(scale_vals.min()), vmax=float(scale_vals.max()))
    bar_colors = [cmap(norm(v)) if ok else INVALID_COLOR
                  for v, ok in zip(skipped_vals, valid_mask)]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.4), 6))
    x = np.arange(len(ordered))

    ax.bar(x, means, yerr=errs, capsize=3, color=bar_colors, alpha=0.95,
           edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    _style_invalid_ticks(ax, ordered)
    ax.set_xlabel("Distribuicao de threads (nxThreads × nyThreads × nzThreads)")
    ax.set_ylabel(f"{metric_label} — media ± 3·desvio/√n")
    title = (
        f"{metric_label} por distribuicao de threads — degrade por skippedWarps (%)\n"
        f"totalThreads={totalThreads}  "
        f"(repeticoes por barra: min={min(ns)}, max={max(ns)}; ordenado por {metric_label.lower()})"
    )
    if n_invalid:
        title += f"\n{n_invalid} combinacao(oes) invalida(s) em preto"
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("skippedWarps medio (%)")

    if n_invalid:
        ax.legend(handles=[_invalid_legend_handle()], fontsize=8, loc="upper right")

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

    n_invalid_runs = sum(1 for e in all_entries if not e["validSimulation"])
    n_without_field = sum(1 for e in all_entries if not e["hasValidField"])
    print(f"  {n_invalid_runs} execucao(oes) com ValidSimulation=0 (serao plotadas em preto)")
    if n_without_field:
        print(f"  {n_without_field} execucao(oes) sem o campo ValidSimulation "
              f"(formato antigo) -- tratadas como validas")

    stats = aggregate_by_distribution(all_entries)
    n_invalid_groups = sum(1 for s in stats if not s["valid"])
    print(f"{len(stats)} combinacoes distintas (numBlocks x numThreads x distribuicao)"
          f" -- {n_invalid_groups} invalida(s)\n")

    print_table(stats)

    save_csv(stats, os.path.join(SCRIPT_DIR, f"data-thread-factorial/distribuicao_total{totalThreads}.csv"))

    print()

    color_keys = [
        ("numThreads", "numThreads", "numThreads"),
        ("nxThreads", "nxThreads", "nxThreads"),
        ("nyThreads", "nyThreads", "nyThreads"),
        ("nzThreads", "nzThreads", "nzThreads"),
    ]

    # Um conjunto de 5 graficos (4 categoricos + 1 degrade) para cada
    # metrica de tempo: simulacao e generateCubes.
    metrics = [
        ("meanTime", "stdTime", "Tempo de simulacao (s)",
         os.path.join(SCRIPT_DIR, "data-thread-factorial", f"tempo_sim_por_distribuicao_total{totalThreads}")),
        ("meanGencubesTime", "stdGencubesTime", "Tempo de generateCubes (s)",
         os.path.join(SCRIPT_DIR, "data-thread-factorial", f"tempo_gencubes_por_distribuicao_total{totalThreads}")),
    ]

    for mean_key, std_key, metric_label, out_prefix in metrics:
        for color_key, color_title, suffix in color_keys:
            plot_by_distribution_categorical(
                stats, totalThreads, color_key, color_title,
                mean_key, std_key, metric_label,
                f"{out_prefix}_cor_{suffix}.png",
            )

        plot_by_distribution_skipped_warps(
            stats, totalThreads,
            mean_key, std_key, metric_label,
            f"{out_prefix}_cor_skippedWarps.png",
        )


if __name__ == "__main__":
    main()
