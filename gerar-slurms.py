#!/usr/bin/env python3
"""Gerador de scripts .slurm para os experimentos do FluidSimulation no PCAD.

Estrutura assumida no cluster:

    ~/FluidSimulation/
        chunk/  early-memory/  float/  ...      <- uma pasta por experimento,
        main/   mul-add/  restrict/  final/        cada uma com seu Makefile
        experimentos/                            <- saidas (criada pelos jobs)
            main-tupi/  main-cidia/  ...
        jobs/                                    <- .slurm gerados por este script
        logs/                                    <- .out/.err dos jobs

Cada job (um por par experimento x maquina):
  1. recompila na propria maquina (arquitetura detectada da GPU do no; o binario
     fica rotulado por maquina, ex.: build/fluidsim-tupi-sm_89);
  2. roda ncu e nsys nas 3 configuracoes de interesse (curto e limitado);
  3. roda o thread-factorial completo (longo).

O profiling vem ANTES da varredura de proposito: ele e' curto e limitado por
--launch-count, enquanto a varredura pode consumir todo o wall clock. Nessa
ordem, um job que estoura o --time ainda entrega os relatorios.

ISOLAMENTO DO NO (o motivo desta versao)
----------------------------------------
Por padrao o Slurm do PCAD escalona por recurso, nao por no: dois jobs cabem
no mesmo no se sobrarem cores. Sem nenhuma diretiva de exclusividade, varios
jobs desta bateria caiam juntos na mesma maquina (foi o que aconteceu na
cidia), disputando cores, cache, banda de memoria e -- pior -- a MESMA GPU,
porque sem --gres o Slurm nao define CUDA_VISIBLE_DEVICES e todo binario abre
o device 0. Resultado: tempos inflados e sem sentido, e relatorios de ncu/nsys
contaminados (o ncu serializa os kernels do processo que ele mede, mas nao ve
nem controla o kernel do job vizinho na mesma GPU).

As tres diretivas que resolvem isso, geradas agora em TODO job:

  --exclusive     o no inteiro fica alocado a este job; nenhum outro job
                  (seu ou de terceiros) entra nele enquanto ele roda.
  --mem=0         toda a memoria do no. Com --exclusive isso costuma ser
                  implicito, mas se a particao tiver DefMemPerCPU configurado
                  o job herdaria um teto de memoria pequeno sem esta linha.
  --gres=gpu:N    reserva GPU pelo escalonador e faz o Slurm definir
                  CUDA_VISIBLE_DEVICES, fixando qual GPU o binario enxerga.
                  Importante na cidia, que tem 2x RTX 2080 Ti por no.

Se a particao nao tiver GRES configurado, --gres=gpu:1 faz o sbatch recusar o
job ("Requested node configuration is not available"): nesse caso gere com
--sem-gres. O script do job pinna CUDA_VISIBLE_DEVICES sozinho quando o Slurm
nao o define, entao a GPU continua fixa nos dois cenarios.

Consequencia esperada: com --exclusive os jobs deixam de rodar em paralelo no
mesmo no e passam a enfileirar. A bateria demora mais em wall clock -- e' o
preco de medicao limpa.

Uso:
    python3 gerar_slurm.py                    # gera tudo com os defaults
    python3 gerar_slurm.py --experimentos main float --maquinas tupi
    python3 gerar_slurm.py --repeat 10 --totalthreads 4194304
    python3 gerar_slurm.py --sem-gres         # se a particao nao tiver GRES
    python3 gerar_slurm.py --bind             # fixa afinidade OpenMP
"""

import argparse
import stat
import sys
from pathlib import Path

# ---------------------------------------------------------------- configuracao

EXPERIMENTOS = [
    "chunk",
    "early-memory",
    "float",
    "inv-volume",
    "main",
    "mul-add",
    "restrict",
    "final",
]

# tempo = limite de wall clock pedido ao Slurm. E' um TETO: o no e' liberado
# assim que o script termina, entao pedir mais nao "gasta" a maquina -- so
# pesa no escalonamento (jobs curtos costumam entrar antes).
# O PCAD permite ate 24h nessas particoes.
#
# gpus     = quantas GPUs o job reserva (--gres=gpu:N). O FluidSimulation e'
#            single-GPU, entao 1 nos dois casos; na cidia isso deixa a segunda
#            2080 Ti ociosa de proposito, para o job nao depender de qual
#            device o binario resolveu abrir.
# exclude  = nos da particao que ficam fora da comparacao. A particao tupi e'
#            heterogenea: tupi1-2 sao Xeon E5-2620 v4 (16 cores) e tupi3-6 sao
#            i9-14900KF (24 cores), todos com RTX 4090. Comparar experimentos
#            medidos em CPUs diferentes distorce tudo que nao for kernel puro,
#            entao a bateria fica restrita aos i9.
#
#            (--exclude, e nao --nodelist: --nodelist EXIGE todos os nos
#            listados no allocation, e com --nodes=1 o sbatch recusaria o job
#            com "Node count specification invalid". Excluir o complemento e'
#            a forma correta de dizer "qualquer no deste subconjunto".)
MAQUINAS = {
    "tupi": {
        "particao": "tupi",
        "tempo": "07:00:00",
        "gpus": 1,                    # 1x RTX 4090 (sm_89)
        "exclude": "tupi1,tupi2",     # Xeon antigo -- fora da comparacao
    },
    "cidia": {
        "particao": "cidia",
        "tempo": "15:00:00",
        "gpus": 1,                    # o no tem 2x RTX 2080 Ti (sm_75); usamos 1
        "exclude": "",
    },
}

# Configuracoes de threadsDim perfiladas com ncu/nsys em cada job.
DIMS_PROFILING = ["512 1 1", "16 16 1", "4 8 1"]

# Caminho ABSOLUTO: as diretivas #SBATCH nao expandem variaveis de shell
# ($HOME, ~), entao --output/--error precisam do caminho literal. O default e'
# resolvido na hora da geracao a partir do home REAL de quem roda o gerador --
# no PCAD o home e' /home/users/<user>, nao /home/<user>, e chutar isso faz o
# job morrer antes de executar qualquer linha (ExitCode 0:53, sem log).
BASE_PADRAO = str(Path.home() / "FluidSimulation")
TOTALTHREADS_PADRAO = 1048576
REPEAT_PADRAO = 30

# ---------------------------------------------------------------- template

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={exp}-{maq}
#SBATCH --partition={particao}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time={tempo}
#SBATCH --output={base}/logs/%x_%j.out
#SBATCH --error={base}/logs/%x_%j.err
# --- isolamento: o no inteiro e' deste job, sem vizinhos disputando cores,
# --- banda de memoria ou GPU. Sem isto a medicao nao vale nada.
#SBATCH --exclusive
#SBATCH --mem=0
{gres}{exclude}
# Gerado por gerar_slurm.py -- nao edite a mao, edite o gerador.

set -u

BASE="{base}"
EXP="{exp}"
MAQ="{maq}"
SRC="$BASE/$EXP"
OUT="$BASE/experimentos/$EXP-$MAQ"

# O ambiente do job nao herda de forma confiavel o shell interativo, e o
# .bashrc pode nem ser lido: o CUDA e' exportado aqui explicitamente.
# ${{VAR:-}} porque o job roda com "set -u" e o ambiente limpo do Slurm
# frequentemente NAO tem LD_LIBRARY_PATH definido -- sem o :- o script morre
# na primeira linha com "unbound variable".
export PATH="/usr/local/cuda/bin:${{PATH:-}}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${{LD_LIBRARY_PATH:-}}"

# Fixa a GPU. Com --gres o Slurm ja define CUDA_VISIBLE_DEVICES e o valor dele
# manda (o indice e' relativo ao que foi alocado). Sem GRES na particao, ou com
# --sem-gres, a variavel chega vazia e o binario abriria o device 0 "por sorte":
# aqui ela e' pinada explicitamente, para o job ser reproduzivel nos dois casos.
export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-{gpu_default}}}"

# Redundante num no exclusivo (o default do OpenMP ja seria todos os cores),
# mas amarra o numero de threads ao que o Slurm alocou: se algum dia este job
# rodar sem --exclusive, ele nao vai mais abrir 1 thread por core do no
# enquanto divide a maquina com outro job.
export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-${{SLURM_CPUS_ON_NODE:-1}}}}"
{bind}
mkdir -p "$BASE/logs" "$OUT"

if [ ! -d "$SRC" ]; then
    echo "ERRO: pasta do experimento nao existe: $SRC"
    exit 1
fi
cd "$SRC" || exit 1

# --- checagem de exclusividade: se por qualquer motivo (particao configurada
# --- com OverSubscribe=FORCE, diretiva removida, etc.) o no estiver sendo
# --- compartilhado, e' melhor descobrir agora, no topo do log, do que ao
# --- analisar tempos estranhos daqui a duas semanas.
outros=$(squeue -h -w "$(hostname -s)" -o "%A" 2>/dev/null \\
         | grep -vx "${{SLURM_JOB_ID:-__none__}}" | wc -l)
if [ "${{outros:-0}}" -gt 0 ]; then
    echo "AVISO: ha $outros outro(s) job(s) neste no. O no NAO esta exclusivo;"
    echo "       os tempos medidos vao estar contaminados."
    squeue -w "$(hostname -s)" 2>/dev/null
fi

# --- proveniencia: sem isso, daqui a um mes ninguem sabe qual codigo gerou
# --- estes numeros, em qual GPU, com qual toolkit.
{{
    echo "job          : ${{SLURM_JOB_ID:-?}} (${{SLURM_JOB_NAME:-?}})"
    echo "no           : $(hostname)"
    echo "inicio       : $(date -Is)"
    echo "origem       : $SRC"
    echo "git branch   : $(git branch --show-current 2>/dev/null || echo n/a)"
    echo "git commit   : $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
    echo "nvcc         : $(nvcc --version 2>/dev/null | tail -1 || echo ausente)"
    echo "exclusivo    : ${{SLURM_JOB_NODES:-?}} no(s); outros jobs no no: ${{outros:-?}}"
    echo "cpus no no   : ${{SLURM_CPUS_ON_NODE:-?}}"
    echo "omp threads  : $OMP_NUM_THREADS"
    echo "gres pedido  : {gres_desc}"
    echo "cuda devices : $CUDA_VISIBLE_DEVICES"
    echo -n "gpu          : "
    nvidia-smi --query-gpu=index,name,compute_cap,driver_version \\
        --format=csv,noheader 2>/dev/null || echo n/a
    echo "binario      : $(make -s arch MACHINE="$MAQ" 2>/dev/null | sed -n 's/^BIN *= *//p')"
    echo "totalthreads : {tt}"
    echo "repeat       : {repeat}"
}} > "$OUT/execucao_${{SLURM_JOB_ID:-manual}}.info"

echo "=== build ==="
# MACHINE=$MAQ rotula o binario pela maquina do experimento (build/fluidsim-tupi-sm_89),
# em vez de depender do hostname do no sorteado pelo Slurm (tupi3, tupi5, ...).
make arch MACHINE="$MAQ"
if ! make MACHINE="$MAQ"; then
    echo "ERRO: build falhou em $SRC"
    exit 1
fi

echo
echo "=== profiling (ncu + nsys) ==="
# Primeiro porque e' curto e limitado; a varredura abaixo e' que arrisca
# bater no wall clock. Falha de profiling nao aborta o job.
for dim in {dims}; do
    echo "--- ncu  threadsDim=$dim ---"
    make ncu  MACHINE="$MAQ" TOTALTHREADS={tt} THREADSDIM="$dim" FOLDER="$OUT" \\
        || echo "AVISO: ncu falhou em '$dim' (segue o job)"
    echo "--- nsys threadsDim=$dim ---"
    make nsys MACHINE="$MAQ" TOTALTHREADS={tt} THREADSDIM="$dim" FOLDER="$OUT" \\
        || echo "AVISO: nsys falhou em '$dim' (segue o job)"
done

echo
echo "=== thread-factorial (varredura completa) ==="
# FORCE_REBUILD=1 (default do Makefile): recompila do zero antes de medir,
# garantindo que o binario da medicao e' o desta maquina e deste codigo.
make thread-factorial MACHINE="$MAQ" FORCE_REBUILD=1 \\
    TOTALTHREADS={tt} REPEAT={repeat} FOLDER="$OUT"
rc=$?

echo
echo "fim: $(date -Is)  (thread-factorial rc=$rc)"
echo "saidas em: $OUT"
exit $rc
"""

BIND_BLOCK = """
# Afinidade fixa: cada thread OpenMP presa a um core, sem migracao entre
# medicoes. Reduz variancia, mas MUDA os numeros em relacao a rodadas sem
# binding -- nao misture as duas na mesma tabela.
export OMP_PROC_BIND=close
export OMP_PLACES=cores
"""

SUBMIT_TEMPLATE = """#!/bin/bash
# Submete todos os jobs gerados. Gerado por gerar_slurm.py.
cd "$(dirname "$0")" || exit 1

BASE="{base}"

# --- verificacao que tem que rodar AQUI, no cluster ------------------------
# O Slurm abre o arquivo de --output ANTES de executar a primeira linha do
# script. Se o caminho nao existir, o job morre com ExitCode 0:53 em menos de
# um segundo, SEM deixar log -- e, com varios jobs na fila, cada um morre e
# libera o no para o proximo morrer, evaporando a bateria inteira em segundos.
#
# O modo classico de cair nisso e' gerar os .slurm numa maquina cujo $HOME e'
# diferente do $HOME do cluster: o caminho fica absoluto (e o gerador nao
# reclama) mas aponta para um home que so existe no notebook. Por isso a
# checagem e' aqui, onde os jobs de fato vao rodar, e nao na geracao.
if [ ! -d "$BASE" ]; then
    echo "ERRO: a raiz nao existe nesta maquina:"
    echo "         $BASE"
    echo "      \\$HOME aqui e': $HOME"
    echo "      Os .slurm foram gerados com o home de outra maquina. Corrija com:"
    echo "         sed -i \\"s|$BASE|\\$HOME/FluidSimulation|g\\" *.slurm"
    echo "      ou regenere no cluster: python3 gerar_slurm.py --base \\"\\$HOME/FluidSimulation\\""
    exit 1
fi

mkdir -p "$BASE/logs" "$BASE/experimentos" || exit 1
if [ ! -w "$BASE/logs" ]; then
    echo "ERRO: sem permissao de escrita em $BASE/logs -- os jobs morreriam com 0:53."
    exit 1
fi

# Os .slurm carregam o caminho literal nas diretivas #SBATCH (que nao expandem
# \\$HOME). Confere que nenhum ficou com o caminho de outra maquina.
if grep -l -- "--output=" *.slurm | xargs grep -h -- "--output=" \\
   | grep -v "^#SBATCH --output=$BASE/" | grep -q .; then
    echo "ERRO: ha .slurm apontando --output para fora de $BASE:"
    grep -h -- "--output=" *.slurm | sort -u
    exit 1
fi

# Todos os jobs pedem o no inteiro (--exclusive), entao eles NAO rodam em
# paralelo no mesmo no: a fila serializa por maquina. Submeter todos de uma vez
# continua certo -- o escalonador cuida da ordem -- mas espere wall clock longo.
for f in {arquivos}; do
    echo -n "$f -> "
    sbatch "$f"
done

echo
echo "acompanhe com: squeue -u $USER -o '%.10i %.18j %.9P %.2t %.10M %R'"
"""


def validar_base(base, forcar):
    """Recusa gerar com uma raiz que nao existe NESTA maquina.

    E' a unica defesa possivel contra o erro mais caro deste script: as
    diretivas #SBATCH nao expandem $HOME nem ~, entao o caminho de
    --output/--error e' gravado literalmente dentro do .slurm. Gerar no
    notebook assa o home do NOTEBOOK nos arquivos; no cluster o Slurm nao
    consegue abrir o arquivo de saida, e o job morre com ExitCode 0:53 em
    menos de um segundo, SEM deixar log -- e cada job que morre libera o no
    para o proximo morrer, evaporando a bateria inteira em segundos.

    Ser absoluto nao basta como teste: '/home/geancarlok/FluidSimulation' e'
    absoluto e valido em sintaxe. So existir de fato distingue os dois casos.
    """
    if not base.startswith("/"):
        print(f"ERRO: --base '{base}' nao e' absoluto.", file=sys.stderr)
        print("      As diretivas #SBATCH nao expandem $HOME nem ~.", file=sys.stderr)
        return False

    if Path(base).is_dir():
        return True

    print(f"ERRO: '{base}' nao existe nesta maquina.", file=sys.stderr)
    print(f"      $HOME aqui e': {Path.home()}", file=sys.stderr)
    print("", file=sys.stderr)
    print("      Gere no proprio cluster -- e' o unico lugar onde o default", file=sys.stderr)
    print("      resolve para o caminho certo:", file=sys.stderr)
    print("          ssh <cluster>", file=sys.stderr)
    print("          cd ~/FluidSimulation && python3 gerar_slurm.py ...", file=sys.stderr)
    print("", file=sys.stderr)
    print("      No PCAD o home e' /home/users/<user>, NAO /home/<user>.", file=sys.stderr)
    print("      Se voce sabe o caminho de la e quer gerar daqui mesmo:", file=sys.stderr)
    print(f"          python3 {Path(sys.argv[0]).name} --base /home/users/<user>/FluidSimulation --forcar",
          file=sys.stderr)
    if forcar:
        print("", file=sys.stderr)
        print("      --forcar informado: gerando assim mesmo. O submeter_todos.sh",
              file=sys.stderr)
        print("      vai recusar a submissao se o caminho nao bater no cluster.",
              file=sys.stderr)
        return True
    return False


def gerar(base, experimentos, maquinas, dims, tt, repeat, sem_gres, bind, saida):
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)

    # Se a raiz existe, ja cria as pastas que o Slurm precisa abrir. Uma
    # bateria inteira ja foi perdida porque logs/ nao existia no momento em
    # que o primeiro job entrou.
    if Path(base).is_dir():
        for sub in ("logs", "experimentos"):
            (Path(base) / sub).mkdir(exist_ok=True)
    dims_bash = " ".join(f'"{d}"' for d in dims)
    gerados = []

    for exp in experimentos:
        for maq in maquinas:
            cfg = MAQUINAS[maq]
            n_gpus = cfg["gpus"]

            if sem_gres:
                gres = ""
                gres_desc = "nenhum (--sem-gres); CUDA_VISIBLE_DEVICES pinado no script"
            else:
                gres = f"#SBATCH --gres=gpu:{n_gpus}\n"
                gres_desc = f"gpu:{n_gpus}"

            exclude = (
                f"#SBATCH --exclude={cfg['exclude']}\n" if cfg["exclude"] else ""
            )

            texto = TEMPLATE.format(
                exp=exp,
                maq=maq,
                particao=cfg["particao"],
                tempo=cfg["tempo"],
                base=base,
                dims=dims_bash,
                tt=tt,
                repeat=repeat,
                gres=gres,
                gres_desc=gres_desc,
                exclude=exclude,
                gpu_default=",".join(str(i) for i in range(n_gpus)),
                bind=BIND_BLOCK if bind else "",
            )
            caminho = saida / f"{exp}-{maq}.slurm"
            caminho.write_text(texto, encoding="utf-8")
            caminho.chmod(caminho.stat().st_mode | stat.S_IXUSR)
            gerados.append(caminho.name)

    submit = saida / "submeter_todos.sh"
    submit.write_text(
        SUBMIT_TEMPLATE.format(base=base, arquivos=" ".join(sorted(gerados))),
        encoding="utf-8",
    )
    submit.chmod(submit.stat().st_mode | stat.S_IXUSR)

    return gerados, submit


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=BASE_PADRAO,
                   help=f"raiz no cluster, ABSOLUTA (default: {BASE_PADRAO})")
    p.add_argument("--experimentos", nargs="+", default=EXPERIMENTOS,
                   help="pastas de experimento (default: todas as 8)")
    p.add_argument("--maquinas", nargs="+", default=list(MAQUINAS),
                   choices=list(MAQUINAS), help="maquinas alvo")
    p.add_argument("--dims", nargs="+", default=DIMS_PROFILING,
                   help='threadsDim perfilados, ex: --dims "512 1 1" "16 16 1"')
    p.add_argument("--totalthreads", type=int, default=TOTALTHREADS_PADRAO)
    p.add_argument("--repeat", type=int, default=REPEAT_PADRAO)
    p.add_argument("--sem-gres", action="store_true",
                   help="omite #SBATCH --gres (use se a particao nao tiver GRES "
                        "configurado e o sbatch recusar o job)")
    p.add_argument("--bind", action="store_true",
                   help="fixa OMP_PROC_BIND/OMP_PLACES (muda os numeros medidos)")
    p.add_argument("--saida", default="jobs", help="pasta dos .slurm (default: jobs)")
    p.add_argument("--forcar", action="store_true",
                   help="gera mesmo que --base nao exista nesta maquina "
                        "(so use se souber o caminho exato do cluster)")
    a = p.parse_args()

    if not validar_base(a.base, a.forcar):
        return 1

    gerados, submit = gerar(a.base, a.experimentos, a.maquinas, a.dims,
                            a.totalthreads, a.repeat, a.sem_gres, a.bind, a.saida)

    print(f"{len(gerados)} scripts gerados em {a.saida}/:")
    for g in sorted(gerados):
        print(f"  {g}")

    print("\ncada job pede o no INTEIRO (--exclusive --mem=0"
          + ("" if a.sem_gres else " --gres=gpu:N") + ").")
    for maq in a.maquinas:
        cfg = MAQUINAS[maq]
        if cfg["exclude"]:
            print(f"  {maq}: nos excluidos -> {cfg['exclude']}")
    print("  jobs do mesmo no enfileiram em vez de rodar juntos: wall clock maior,")
    print("  medicao limpa.")

    # O unico campo que nao pode estar errado sem consequencia: impresso
    # sempre, para conferencia visual antes de submeter.
    print(f"\ncaminho gravado nos .slurm (nao expande $HOME):")
    print(f"  #SBATCH --output={a.base}/logs/%x_%j.out")

    prefixo = "" if str(submit).startswith("/") else "./"
    print(f"\nsubmeter todos: {prefixo}{submit}")
    print(f"submeter um:    sbatch {a.saida}/main-tupi.slurm")
    return 0


if __name__ == "__main__":
    sys.exit(main())