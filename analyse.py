import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict

# ── Parse ──────────────────────────────────────────────────────────────────────

def parse_data(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    blocks = text.split('------------------------------------------------------------------')
    entries = []

    for block in blocks:
        m_nt = re.search(r'numThreads=(\d+)', block)
        m_nb = re.search(r'numBlocks=(\d+)', block)
        m_xt = re.search(r'xThreads=(\d+)', block)
        m_yt = re.search(r'yThreads=(\d+)', block)
        m_zt = re.search(r'zThreads=(\d+)', block)
        m_time = re.search(r'Total simulation time \(s\):\s+([\d.]+)', block)

        if all([m_nt, m_nb, m_xt, m_yt, m_zt, m_time]):
            nt = int(m_nt.group(1))
            nb = int(m_nb.group(1))
            xt = int(m_xt.group(1))
            yt = int(m_yt.group(1))
            zt = int(m_zt.group(1))
            total_cells = xt * yt * zt
            time = float(m_time.group(1))
            iterations = 100000
            efficiency = (nt * nb * iterations) / time if time > 0 else 0
            entries.append({
                'numThreads': nt,
                'numBlocks': nb,
                'totalCells': total_cells,
                'time': time,
                'efficiency': efficiency,
            })

    return entries


# ── Aggregate ──────────────────────────────────────────────────────────────────

def aggregate(entries):
    """Group by (numBlocks, numThreads) -> lists of measurements."""
    groups = defaultdict(list)
    for e in entries:
        key = (e['numBlocks'], e['numThreads'])
        groups[key].append(e)
    return groups


# ── Plot 1: Efficiency vs numThreads per numBlocks (mean ± std) ────────────────

def plot_efficiency_vs_threads(groups, outpath):
    fig, ax = plt.subplots(figsize=(14, 7))

    all_nb = sorted(set(k[0] for k in groups))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(all_nb) - 1, 1)) for i in range(len(all_nb))]

    for idx, nb in enumerate(all_nb):
        threads = []
        means = []
        stds = []
        for (gnb, gnt), elist in sorted(groups.items()):
            if gnb != nb:
                continue
            effs = [e['efficiency'] for e in elist]
            threads.append(gnt)
            means.append(np.mean(effs))
            stds.append(np.std(effs))

        threads = np.array(threads)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(threads, means, 'o-', color=colors[idx], label=f'{nb} blocks', markersize=4, linewidth=1.2)
        ax.fill_between(threads, means - stds, means + stds, alpha=0.15, color=colors[idx])

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Threads per Block')
    ax.set_ylabel('Efficiency (cells/s)')
    ax.set_title('Processing Efficiency vs Threads per Block (mean ± std)')
    ax.legend(title='Blocks', fontsize=7, title_fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f'  -> {outpath}')


# ── Plot 2: Time vs numThreads per numBlocks (mean ± std) ─────────────────────

def plot_time_vs_threads(groups, outpath):
    fig, ax = plt.subplots(figsize=(14, 7))

    all_nb = sorted(set(k[0] for k in groups))
    cmap = plt.cm.inferno
    colors = [cmap(i / max(len(all_nb) - 1, 1)) for i in range(len(all_nb))]

    for idx, nb in enumerate(all_nb):
        threads = []
        means = []
        stds = []
        for (gnb, gnt), elist in sorted(groups.items()):
            if gnb != nb:
                continue
            times = [e['time'] for e in elist]
            threads.append(gnt)
            means.append(np.mean(times))
            stds.append(np.std(times))

        threads = np.array(threads)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(threads, means, 's-', color=colors[idx], label=f'{nb} blocks', markersize=4, linewidth=1.2)
        ax.fill_between(threads, means - stds, means + stds, alpha=0.15, color=colors[idx])

    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel('Threads per Block')
    ax.set_ylabel('Simulation Time (s)')
    ax.set_title('Simulation Time vs Threads per Block (mean ± std)')
    ax.legend(title='Blocks', fontsize=7, title_fontsize=8, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f'  -> {outpath}')


# ── Plot 3: Heatmap de eficiência (numBlocks × numThreads) ────────────────────

def plot_heatmap(groups, outpath):
    all_nb = sorted(set(k[0] for k in groups))
    all_nt = sorted(set(k[1] for k in groups))

    matrix = np.full((len(all_nb), len(all_nt)), np.nan)
    for i, nb in enumerate(all_nb):
        for j, nt in enumerate(all_nt):
            key = (nb, nt)
            if key in groups:
                effs = [e['efficiency'] for e in groups[key]]
                matrix[i, j] = np.mean(effs)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(np.log10(matrix), aspect='auto', cmap='magma', origin='lower')

    ax.set_xticks(range(len(all_nt)))
    ax.set_xticklabels([str(x) for x in all_nt], rotation=45, ha='right')
    ax.set_yticks(range(len(all_nb)))
    ax.set_yticklabels([str(x) for x in all_nb])

    ax.set_xlabel('Threads per Block')
    ax.set_ylabel('Number of Blocks')
    ax.set_title('Efficiency Heatmap — log₁₀(cells/s)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('log₁₀(cells/s)')

    # annotate cells with actual values
    for i in range(len(all_nb)):
        for j in range(len(all_nt)):
            val = matrix[i, j]
            if not np.isnan(val):
                txt = f'{val:.0f}' if val < 1000 else f'{val:.0e}'
                fontsize = 6 if len(all_nt) > 8 else 7
                color = 'white' if np.log10(val) < (np.nanmin(np.log10(matrix)) + np.nanmax(np.log10(matrix))) / 2 else 'black'
                ax.text(j, i, txt, ha='center', va='center', fontsize=fontsize, color=color)

    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f'  -> {outpath}')


# ── Plot 4: Efficiency vs Total Cells (iso-cell analysis) ─────────────────────

def plot_efficiency_vs_total_cells(groups, outpath):
    """For combinations where numBlocks*numThreads is the same product,
    compare efficiency across different block/thread splits.
    E.g. 1b×1024t, 2b×512t, 4b×256t, ..., 1024b×1t all grouped together."""
    # group by product = numBlocks * numThreads
    by_product = defaultdict(list)
    for (nb, nt), elist in groups.items():
        product = nb * nt
        mean_eff = np.mean([e['efficiency'] for e in elist])
        std_eff = np.std([e['efficiency'] for e in elist])
        by_product[product].append({
            'nb': nb, 'nt': nt,
            'mean_eff': mean_eff, 'std_eff': std_eff,
        })

    # only groups with multiple configurations
    multi = {k: v for k, v in by_product.items() if len(v) > 1}
    if not multi:
        print('  (no iso-product configurations found, skipping)')
        return

    products = sorted(multi.keys())

    # one subplot per product for clarity
    n_plots = len(products)
    cols = min(n_plots, 4)
    rows = (n_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows), squeeze=False)

    # create output directory for individual plots
    outdir = os.path.join(os.path.dirname(outpath), 'iso_individual')
    os.makedirs(outdir, exist_ok=True)

    cmap = plt.cm.viridis

    def _draw_bars(ax, product):
        configs = sorted(multi[product], key=lambda c: c['nb'])
        labels = [f'B{c["nb"]}×T{c["nt"]}' for c in configs]
        means = [c['mean_eff'] for c in configs]
        stds = [c['std_eff'] for c in configs]

        x = np.arange(len(configs))
        colors = [cmap(i / max(len(configs) - 1, 1)) for i in range(len(configs))]
        ax.bar(x, means, yerr=stds, capsize=3, color=colors,
               alpha=0.85, edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'B×T = {product}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Efficiency (cells/s)', fontsize=8)
        ax.grid(True, alpha=0.2, axis='y')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    for idx, product in enumerate(products):
        # draw on combined figure
        ax = axes[idx // cols][idx % cols]
        _draw_bars(ax, product)

        # generate individual figure
        fig_i, ax_i = plt.subplots(figsize=(8, 5))
        _draw_bars(ax_i, product)
        fig_i.tight_layout()
        individual_path = os.path.join(outdir, f'iso_BxT_{product}.png')
        fig_i.savefig(individual_path, dpi=200, bbox_inches='tight')
        plt.close(fig_i)

    print(f'  -> {outdir}/ ({len(products)} individual plots)')

    # hide unused subplots
    for idx in range(n_plots, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('Efficiency for Same numBlocks×numThreads Product — Different Splits',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> {outpath}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'dataOpt.txt'
    print(f'Parsing {filepath}...')
    entries = parse_data(filepath)
    print(f'  {len(entries)} entries parsed')

    groups = aggregate(entries)
    print(f'  {len(groups)} unique (blocks, threads) configurations')

    # count iterations per config
    counts = [len(v) for v in groups.values()]
    print(f'  iterations per config: min={min(counts)}, max={max(counts)}, median={int(np.median(counts))}')

    print('\nGenerating plots...')
    plot_efficiency_vs_threads(groups, 'efficiency_vs_threads.png')
    plot_time_vs_threads(groups, 'time_vs_threads.png')
    plot_heatmap(groups, 'heatmap_efficiency.png')
    plot_efficiency_vs_total_cells(groups, 'iso_cell_comparison.png')

    print('\nDone.')


if __name__ == '__main__':
    main()