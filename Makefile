NVCC   ?= nvcc
STD    ?= c++17
TARGET ?= fluidsim
BUILD  ?= build

# --- Arquitetura da GPU -----------------------------------------------------
# Por padrao detecta a compute capability da GPU 0 da maquina que compila
# (ex.: RTX 4090 -> "8.9" -> sm_89) e usa isso tanto no -arch quanto no nome
# do binario, que fica build/fluidsim-sm_89. Assim binarios de arquiteturas
# diferentes convivem sem se sobrescrever e da' pra ver, pelo nome, para o
# que cada um foi compilado.
#   make                 - detecta sozinho
#   make ARCH=sm_86      - forca uma arquitetura (nome vira fluidsim-sm_86)
#   make ARCH=native     - equivalente ao default (tambem e' resolvido p/ sm_XX)
# Se a deteccao falhar (sem nvidia-smi, GPU ausente, Windows sem tr/sed),
# cai no FALLBACK_ARCH.
FALLBACK_ARCH  ?= sm_75
DETECTED_ARCH  := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' .' | sed -e 's/^/sm_/' -e 's/^sm_$$//')

ARCH ?= $(DETECTED_ARCH)
ifeq ($(strip $(ARCH)),)
    ARCH := $(FALLBACK_ARCH)
endif
ifeq ($(strip $(ARCH)),native)
    override ARCH := $(if $(DETECTED_ARCH),$(DETECTED_ARCH),$(FALLBACK_ARCH))
endif

NCU         ?= ncu
NSYS        ?= nsys

# setInsideVertices roda 1x (setup, O(totalThreads x nTriangulos) - pesado)
# e é perfilado à parte, em escala reduzida (ver alvo ncu-setup).
# fluidMovement/recalculateVelocities rodam por timestep, são baratos e
# podem ser perfilados com --set full na escala real do problema.
NCU_LOOP_KERNELS  ?= fluidMovement|recalculateVelocities
NCU_SETUP_KERNELS ?= setInsideVertices
NCU_KERNELS       ?= $(NCU_LOOP_KERNELS)

NCU_SET     ?= full
NCU_LAUNCHES ?= 6            # nº de invocações de cada kernel a perfilar (0 = todas, cuidado!)
NCU_SKIP     ?= 0            # nº de invocações iniciais a pular (útil p/ ignorar warm-up)

# ARGS e obrigatorio. O binario (main.cu) aceita flags nomeadas, em
# qualquer ordem e quantidade -- CHECK_ARGS so valida que algo foi
# passado; formato e validacao de valores sao responsabilidade do
# parser em main.cu.
CHECK_ARGS = $(if $(strip $(ARGS)),,\
    $(error ARGS nao informado. Exemplo: ARGS="--numThreads 1024 --numBlocks 64 --blocksDim 4 2 8"))

# Pasta onde o alvo "factorial" joga as saidas/dados do teste em lote.
DATA_DIR ?= data
DATA_FACT_DIR ?= data-factorial
# Pasta de saida dos alvos fatoriais. FOLDER, quando informado, vale para o
# alvo que estiver rodando; sem ele cada alvo usa seu default proprio:
#   make thread-factorial ... FOLDER="cidia-main"
#   make chunks-factorial ... FOLDER="cidia-chunks"
FOLDER      ?=
DATA_TF_DIR ?= $(if $(strip $(FOLDER)),$(FOLDER),data-thread-factorial)
DATA_CF_DIR ?= $(if $(strip $(FOLDER)),$(FOLDER),data-chunks-factorial)

# Listas testadas no fatorial -- todas as combinacoes de PROBLEM x THREADS
# serao executadas. Sobrescreva na linha de comando se quiser, ex:
#   make factorial PROBLEM="1000 5000" THREADS="128 256"
PROBLEM ?= 1048576 4194304 16777216
THREADS ?= 32 64 128 256 512 1024

XDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
YDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
ZDIV ?= 1 2 4 8 16 32 64

# Listas varridas pelo alvo "chunks-factorial" (particao do grid de BLOCOS
# em chunks). Toda combinacao com x*y*z <= MAXCHUNKS e' testada.
CXDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
CYDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
CZDIV ?= 1 2 4 8 16 32 64 128 256 512 1024

# Teto do produto x*y*z no chunks-factorial.
MAXCHUNKS ?= 64
# Distribuicao FIXA de threads por bloco usada no chunks-factorial
# (3 valores; numThreads = tx*ty*tz).
THREADSDIM ?= 8 8 16

SRCS := main.cu kernels.cu utils.cu  mesh.cu
HDRS := defines.h kernels.h utils.h  mesh.h

ifeq ($(OS),Windows_NT)
    EXE       := .exe
    MKDIR      = if not exist "$(subst /,\,$(OBJDIR))" mkdir "$(subst /,\,$(OBJDIR))"
    RMDIR      = if exist "$(subst /,\,$(BUILD))" rmdir /S /Q "$(subst /,\,$(BUILD))"
    RMDIR_PATH = if exist "$(subst /,\,$1)" rmdir /S /Q "$(subst /,\,$1)"
    MKDIR_PATH = if not exist "$(subst /,\,$1)" mkdir "$(subst /,\,$1)"
    FIX        = $(subst /,\,$1)
    HOSTFLAGS :=
else
    EXE       :=
    MKDIR      = mkdir -p $(OBJDIR)
    RMDIR      = rm -rf $(BUILD)
    RMDIR_PATH = rm -rf $1
    MKDIR_PATH = mkdir -p $1
    FIX        = $1
    HOSTFLAGS := -Xcompiler -fpermissive
endif

# --- Identificacao do binario -----------------------------------------------
# O binario e os objetos sao nomeados por MAQUINA + ARQUITETURA, para que
# builds de maquinas diferentes convivam na mesma pasta (util quando varias
# maquinas compartilham o home via NFS, como no PCAD) e para que dê pra ver,
# pelo nome do arquivo, onde aquele binario foi gerado:
#
#   build/fluidsim-tupi-sm_89        build/obj-tupi-sm_89/
#   build/fluidsim-cidia-sm_75       build/obj-cidia-sm_75/
#
# MACHINE default = hostname sem os digitos finais (tupi3 -> tupi). Nos jobs
# do Slurm vale passar explicito (MACHINE=tupi), que e' mais confiavel do que
# depender do nome do no sorteado.
#   make MACHINE=cidia          - forca o rotulo da maquina
#   make MACHINE=               - volta ao nome so' por arquitetura
#   make VARIANT=tupi           - controla o sufixo inteiro de uma vez
MACHINE ?= $(shell hostname -s 2>/dev/null | sed -e 's/[0-9]*$$//')
VARIANT ?= $(if $(strip $(MACHINE)),$(strip $(MACHINE))-,)$(ARCH)$(if $(filter 1,$(DEBUG)),-debug)

OBJDIR  := $(BUILD)/obj-$(VARIANT)
BIN     := $(BUILD)/$(TARGET)-$(VARIANT)$(EXE)
OBJS    := $(patsubst %.cu,$(OBJDIR)/%.o,$(SRCS))

# Os alvos de experimento recompilam do zero antes de medir: assim o binario
# usado e' necessariamente o da maquina e do codigo atuais, e nao um objeto
# velho de outra arquitetura ou de uma edicao anterior. O custo (dezenas de
# segundos) e' irrelevante diante de varreduras de horas.
# FORCE_REBUILD=0 desliga, se voce acabou de compilar e quer economizar.
FORCE_REBUILD ?= 1
EXP_DEP := $(if $(filter 1,$(FORCE_REBUILD)),rebuild,$(BIN))

ifeq ($(DEBUG),1)
    NVCCFLAGS ?= -G -g -std=$(STD) -arch=$(ARCH) $(HOSTFLAGS)
else
    NVCCFLAGS ?= -O3   -std=$(STD) -arch=$(ARCH) $(HOSTFLAGS)
endif

.PHONY: all run rebuild factorial thread-factorial thread-factorial-experiment chunks-factorial clean clean-all help arch ncu ncu-setup ncu-quick ncu-full nsys
.DEFAULT_GOAL := all

all: $(BIN)
	@echo "Binario: $(BIN)  (arch=$(ARCH)$(if $(DETECTED_ARCH),, -- deteccao falhou, usando FALLBACK_ARCH))"

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(OBJDIR)/%.o: %.cu $(HDRS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	$(MKDIR)

# Recompila do zero a variante atual (maquina + arquitetura). E' o que os
# alvos de experimento usam como pre-requisito quando FORCE_REBUILD=1.
rebuild:
	@echo "=== rebuild ($(VARIANT)) ==="
	@$(call RMDIR_PATH,$(OBJDIR))
	@$(call RMDIR_PATH,$(BIN))
	@$(MAKE) --no-print-directory $(BIN)
	@echo "Binario recompilado: $(BIN)"

# Mostra o que foi detectado sem compilar nada.
.PHONY: arch
arch:
	@echo "DETECTED_ARCH = $(if $(DETECTED_ARCH),$(DETECTED_ARCH),(nao detectada))"
	@echo "ARCH          = $(ARCH)"
	@echo "MACHINE       = $(if $(strip $(MACHINE)),$(MACHINE),(vazio))"
	@echo "VARIANT       = $(VARIANT)"
	@echo "BIN           = $(BIN)"

run: $(BIN)
	@$(CHECK_ARGS)
	$(call FIX,$(BIN)) $(ARGS)

# Numero de vezes que cada combinacao PROBLEM x THREADS e repetida no
# alvo "factorial". Deve ser um inteiro positivo.
#   make factorial REPEAT=5
REPEAT ?= 1
OBJECT ?= cargo

CHECK_REPEAT = $(if $(shell test "$(REPEAT)" -gt 0 2>/dev/null && echo ok),,\
    $(error REPEAT invalido: "$(REPEAT)" -- precisa ser um inteiro positivo. Ex: REPEAT=5))

factorial: $(EXP_DEP)
	@$(CHECK_REPEAT)
	@$(call RMDIR_PATH,$(DATA_FACT_DIR))
	@$(call MKDIR_PATH,$(DATA_FACT_DIR))
	@echo Fatorial: PROBLEM={$(PROBLEM)} x THREADS={$(THREADS)} x REPEAT=$(REPEAT) = $(words $(PROBLEM)) x $(words $(THREADS)) x $(REPEAT) = $$(( $(words $(PROBLEM)) * $(words $(THREADS)) * $(REPEAT) )) execucoes
	@$(foreach r,$(shell seq 1 $(REPEAT)),$(foreach p,$(PROBLEM),$(foreach t,$(THREADS),echo -- rep=$(r) problemSize=$(p) numThreads=$(t) -- && $(call FIX,$(BIN)) --problemSize $(p) --numThreads $(t) --folder $(DATA_FACT_DIR) --write 0 --time 0 && )))echo Fatorial concluido.

CHECK_TF_ARGS = if [ -z "$(TOTALTHREADS)" ]; then \
	    echo "Erro: TOTALTHREADS nao informado. Exemplo: make thread-factorial TOTALTHREADS=1048576 REPEAT=3 FOLDER=cidia-main"; \
	    exit 1; \
	fi

# Recebe TOTALTHREADS e varre a lista THREADS: para cada numThreads t que
# divida TOTALTHREADS exatamente, usa numBlocks = TOTALTHREADS / t e roda
# todas as combinacoes XDIV x YDIV x ZDIV cujo produto x*y*z bate com t:
#   BIN --numBlocks (TOTALTHREADS/t) --threadsDim x y z --write 0 --time 0
# repetido REPEAT vezes. Combinacoes cujo produto != t sao puladas, assim
# como os t que nao dividem TOTALTHREADS (o filtro roda em shell, ja que
# make nao compara aritmetica nativamente).
# A pasta de saida e' definida por FOLDER (default: data-thread-factorial).

TIMERUN ?= 0.025

thread-factorial: $(EXP_DEP)
	@$(CHECK_TF_ARGS)
	@$(CHECK_REPEAT)
	@$(call MKDIR_PATH,$(DATA_TF_DIR))
	@echo "Thread-factorial: TOTALTHREADS=$(TOTALTHREADS) REPEAT=$(REPEAT) FOLDER=$(DATA_TF_DIR) -- numBlocks=TOTALTHREADS/numThreads"
	@echo "  THREADS={$(THREADS)} -- filtrando XDIV={$(XDIV)} x YDIV={$(YDIV)} x ZDIV={$(ZDIV)} por x*y*z=numThreads"
	@ran=0; \
	for r in $$(seq 1 $(REPEAT)); do \
	    for t in $(THREADS); do \
	        if [ $$(( $(TOTALTHREADS) % t )) -ne 0 ]; then \
	            if [ $$r -eq 1 ]; then echo "-- pulando numThreads=$$t (nao divide TOTALTHREADS=$(TOTALTHREADS))"; fi; \
	            continue; \
	        fi; \
	        nb=$$(( $(TOTALTHREADS) / t )); \
	        for x in $(XDIV); do \
	            for y in $(YDIV); do \
	                for z in $(ZDIV); do \
	                    if [ $$(( x * y * z )) -eq $$t ]; then \
	                        echo "-- rep=$$r numBlocks=$$nb numThreads=$$t threadsDim=$$x $$y $$z --"; \
	                        $(call FIX,$(BIN)) --numBlocks $$nb --threadsDim $$x $$y $$z --folder $(DATA_TF_DIR) --write 0 --time $(TIMERUN) --object $(OBJECT) || exit 1; \
	                        ran=$$(( ran + 1 )); \
	                    fi; \
	                done; \
	            done; \
	        done; \
	    done; \
	done; \
	if [ $$ran -eq 0 ]; then \
	    echo "Erro: nenhuma combinacao valida para TOTALTHREADS=$(TOTALTHREADS)."; \
	    exit 1; \
	fi; \
	echo "Thread-factorial concluido: $$ran execucoes em $(DATA_TF_DIR)."

# ----------------------------------------------------------------------
# thread-factorial-experiment
#
# Versao dirigida do thread-factorial: em vez de varrer XDIV x YDIV x ZDIV,
# roda apenas uma lista fixa de distribuicoes de threads, repetida EXP_REPEAT
# vezes, com o mesmo TOTALTHREADS.
#
#   make thread-factorial-experiment FOLDER="early-memory-tupi"
#
# Defaults: TOTALTHREADS=1048576, EXP_REPEAT=30, tres configuracoes
# (16x16x1, 512x1x1, 4x8x1). Cada uma e' escrita como x,y,z (sem espacos),
# para que o make trate o trio como uma palavra so'.
#
# As repeticoes sao o laco EXTERNO e as configuracoes o interno: assim as
# tres se intercalam ao longo do tempo e nenhuma fica concentrada num
# periodo de GPU fria ou quente, o que enviesaria a comparacao.
# ----------------------------------------------------------------------

EXP_TOTALTHREADS ?= 1048576
EXP_REPEAT       ?= 30
EXP_DIMS         ?= 16,16,1 512,1,1 4,8,1

thread-factorial-experiment: $(EXP_DEP)
	@$(call MKDIR_PATH,$(DATA_TF_DIR))
	@tt=$(if $(strip $(TOTALTHREADS)),$(TOTALTHREADS),$(EXP_TOTALTHREADS)); \
	if [ "$(EXP_REPEAT)" -gt 0 ] 2>/dev/null; then :; else \
	    echo "Erro: EXP_REPEAT invalido: '$(EXP_REPEAT)'."; exit 1; \
	fi; \
	echo "Thread-factorial-experiment: TOTALTHREADS=$$tt REPEAT=$(EXP_REPEAT) FOLDER=$(DATA_TF_DIR)"; \
	echo "  configuracoes: $(EXP_DIMS)"; \
	for d in $(EXP_DIMS); do \
	    set -- $$(echo "$$d" | tr ',' ' '); \
	    if [ $$# -ne 3 ]; then echo "Erro: configuracao '$$d' nao tem 3 valores (use x,y,z)."; exit 1; fi; \
	    nt=$$(( $$1 * $$2 * $$3 )); \
	    if [ $$(( tt % nt )) -ne 0 ]; then \
	        echo "Erro: numThreads=$$nt (=$$1*$$2*$$3) nao divide TOTALTHREADS=$$tt."; exit 1; \
	    fi; \
	done; \
	ran=0; \
	for r in $$(seq 1 $(EXP_REPEAT)); do \
	    for d in $(EXP_DIMS); do \
	        set -- $$(echo "$$d" | tr ',' ' '); \
	        nt=$$(( $$1 * $$2 * $$3 )); \
	        nb=$$(( tt / nt )); \
	        echo "-- rep=$$r/$(EXP_REPEAT) threadsDim=$$1 $$2 $$3 (numThreads=$$nt) numBlocks=$$nb --"; \
	        $(call FIX,$(BIN)) --numBlocks $$nb --threadsDim $$1 $$2 $$3 --folder $(DATA_TF_DIR) --write 0 --time $(TIMERUN) --object $(OBJECT) || exit 1; \
	        ran=$$(( ran + 1 )); \
	    done; \
	done; \
	echo "Thread-factorial-experiment concluido: $$ran execucoes em $(DATA_TF_DIR)."

# ----------------------------------------------------------------------
# chunks-factorial
#
# Fixa a distribuicao de threads (THREADSDIM = tx ty tz) e o tamanho do
# problema (TOTALTHREADS), e varre a particao do grid de BLOCOS em chunks:
# toda combinacao x y z de CXDIV x CYDIV x CZDIV com x*y*z <= MAXCHUNKS.
#
#   numThreads = tx*ty*tz
#   numBlocks  = TOTALTHREADS / numThreads
#   BIN --numBlocks nb --threadsDim tx ty tz --chunksDim x y z
#
# Combinacoes em que a particao de chunks nao divide o grid de blocos sao
# rejeitadas pelo proprio binario; aqui elas sao CONTADAS e PULADAS em vez
# de abortar a varredura, ja que nxBlock/nyBlock/nzBlock so' sao conhecidos
# em tempo de execucao.
# Saida em DATA_CF_DIR (default: data-chunks-factorial; use FOLDER p/ mudar).
# ----------------------------------------------------------------------

CHECK_CF_ARGS = if [ -z "$(TOTALTHREADS)" ]; then \
	    echo "Erro: TOTALTHREADS nao informado. Exemplo: make chunks-factorial TOTALTHREADS=4194304 THREADSDIM=\"8 8 16\" MAXCHUNKS=64 REPEAT=3 FOLDER=cidia-chunks"; \
	    exit 1; \
	fi; \
	if [ $(words $(THREADSDIM)) -ne 3 ]; then \
	    echo "Erro: THREADSDIM precisa de exatamente 3 valores (recebido: '$(THREADSDIM)')."; \
	    exit 1; \
	fi

chunks-factorial: $(EXP_DEP)
	@$(CHECK_CF_ARGS)
	@$(CHECK_REPEAT)
	@$(call MKDIR_PATH,$(DATA_CF_DIR))
	@tx=$(word 1,$(THREADSDIM)); ty=$(word 2,$(THREADSDIM)); tz=$(word 3,$(THREADSDIM)); \
	nt=$$(( tx * ty * tz )); \
	if [ $$(( $(TOTALTHREADS) % nt )) -ne 0 ]; then \
	    echo "Erro: numThreads=$$nt (=$$tx*$$ty*$$tz) nao divide TOTALTHREADS=$(TOTALTHREADS)."; \
	    exit 1; \
	fi; \
	nb=$$(( $(TOTALTHREADS) / nt )); \
	echo "Chunks-factorial: TOTALTHREADS=$(TOTALTHREADS) threadsDim=$$tx $$ty $$tz (numThreads=$$nt) numBlocks=$$nb FOLDER=$(DATA_CF_DIR)"; \
	echo "  varrendo CXDIV x CYDIV x CZDIV com x*y*z <= MAXCHUNKS=$(MAXCHUNKS), REPEAT=$(REPEAT)"; \
	ran=0; skipped=0; \
	for r in $$(seq 1 $(REPEAT)); do \
	    for x in $(CXDIV); do \
	        for y in $(CYDIV); do \
	            for z in $(CZDIV); do \
	                p=$$(( x * y * z )); \
	                if [ $$p -gt $(MAXCHUNKS) ]; then continue; fi; \
	                if [ $$p -gt $$nb ]; then continue; fi; \
	                echo "-- rep=$$r chunksDim=$$x $$y $$z (numChunks=$$p) numBlocks=$$nb threadsDim=$$tx $$ty $$tz --"; \
	                if $(call FIX,$(BIN)) --numBlocks $$nb --threadsDim $$tx $$ty $$tz --chunksDim $$x $$y $$z --folder $(DATA_CF_DIR) --write 0 --time $(TIMERUN) --object $(OBJECT); then \
	                    ran=$$(( ran + 1 )); \
	                else \
	                    echo "   (pulado: chunksDim incompativel com a particao de blocos)"; \
	                    skipped=$$(( skipped + 1 )); \
	                fi; \
	            done; \
	        done; \
	    done; \
	done; \
	if [ $$ran -eq 0 ]; then \
	    echo "Erro: nenhuma combinacao valida executou. Confira THREADSDIM/MAXCHUNKS."; \
	    exit 1; \
	fi; \
	echo "Chunks-factorial concluido: $$ran execucoes em $(DATA_CF_DIR), $$skipped puladas."

# ----------------------------------------------------------------------
# Perfilamento (ncu / nsys)
#
# Os alvos recebem a configuracao de lancamento em variaveis, montam a
# linha de execucao sozinhos e nomeiam o relatorio pela distribuicao de
# threads:
#
#   make ncu  TOTALTHREADS=1048576 THREADSDIM="16 16 1"
#   make nsys NUMBLOCKS=4096       THREADSDIM="8 8 16"
#
#   -> $(PROF_DIR)/ncu_report-16-16-1.ncu-rep
#      $(PROF_DIR)/nsys_report-8-8-16.nsys-rep
#
# THREADSDIM sao 3 numeros (numThreads = tx*ty*tz). O total de blocos vem
# de NUMBLOCKS, se informado; senao e' derivado de TOTALTHREADS
# (numBlocks = TOTALTHREADS / numThreads, que precisa dividir exato).
# PROF_DIR default e' a pasta de dados (respeita FOLDER), nao build/.
# ARGS, se informado, e' anexado ao final (ex.: ARGS="--vel 2.5").
# ----------------------------------------------------------------------

NUMBLOCKS ?=
PROF_DIR  ?= $(DATA_TF_DIR)

# "16 16 1" -> "16-16-1"
empty  :=
space  := $(empty) $(empty)
DIMTAG  = $(subst $(space),-,$(strip $(THREADSDIM)))

CHECK_PROF = if [ $(words $(THREADSDIM)) -ne 3 ]; then \
	    echo "Erro: THREADSDIM precisa de 3 valores (recebido: '$(THREADSDIM)'). Ex: THREADSDIM=\"16 16 1\""; \
	    exit 1; \
	fi; \
	if [ -z "$(strip $(NUMBLOCKS))$(strip $(TOTALTHREADS))" ]; then \
	    echo "Erro: informe TOTALTHREADS ou NUMBLOCKS. Ex: make $@ TOTALTHREADS=1048576 THREADSDIM=\"16 16 1\""; \
	    exit 1; \
	fi

# Calcula nt (threads/bloco) e nb (blocos) no shell da receita.
PROF_CALC = tx=$(word 1,$(THREADSDIM)); ty=$(word 2,$(THREADSDIM)); tz=$(word 3,$(THREADSDIM)); \
	nt=$$(( tx * ty * tz )); \
	if [ -n "$(strip $(NUMBLOCKS))" ]; then \
	    nb=$(NUMBLOCKS); \
	else \
	    if [ $$(( $(TOTALTHREADS) % nt )) -ne 0 ]; then \
	        echo "Erro: numThreads=$$nt (=$$tx*$$ty*$$tz) nao divide TOTALTHREADS=$(TOTALTHREADS)."; \
	        exit 1; \
	    fi; \
	    nb=$$(( $(TOTALTHREADS) / nt )); \
	fi; \
	echo "  numBlocks=$$nb  threadsDim=$(THREADSDIM) (numThreads=$$nt)"

PROF_RUNARGS = --numBlocks $$nb --threadsDim $(THREADSDIM) --write 0 --time $(TIMERUN) --object $(OBJECT) $(ARGS)

# Perfila os kernels do LOOP (fluidMovement/recalculateVelocities), que são
# baratos por invocação — pode (e deve) rodar na escala real do problema.
# Limitado a NCU_LAUNCHES invocações pra não gerar relatórios gigantes que
# travam o ncu-ui ao abrir.
ncu: $(BIN)
	@$(CHECK_PROF)
	@$(call MKDIR_PATH,$(PROF_DIR))
	@echo "ncu -> $(PROF_DIR)/ncu_report-$(DIMTAG).ncu-rep"
	@$(PROF_CALC); \
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(PROF_DIR)/ncu_report-$(DIMTAG) -f $(call FIX,$(BIN)) $(PROF_RUNARGS)

# Perfila SÓ o setInsideVertices, separado do loop. Ele roda 1x só, mas faz
# O(totalThreads x nTriangulos do .obj) trabalho, então --set full nele em
# escala real explode o tempo por passe do kernel replay e derruba o
# profiler (LaunchFailed). Use escala reduzida aqui (ex.: TOTALTHREADS=4096)
# — as métricas por-thread continuam representativas.
ncu-setup: $(BIN)
	@$(CHECK_PROF)
	@$(call MKDIR_PATH,$(PROF_DIR))
	@echo "ncu-setup -> $(PROF_DIR)/ncu_report_setup-$(DIMTAG).ncu-rep"
	@$(PROF_CALC); \
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_SETUP_KERNELS)" \
	    -o $(PROF_DIR)/ncu_report_setup-$(DIMTAG) -f $(call FIX,$(BIN)) $(PROF_RUNARGS)

# Sanidade rápida: set leve (basic), sem coleta pesada, útil para conferir
# se o binário/kernels estão sendo capturados antes de rodar o full.
# Inclui todos os kernels (setup + loop) pois o overhead do basic é baixo.
ncu-quick: $(BIN)
	@$(CHECK_PROF)
	@$(call MKDIR_PATH,$(PROF_DIR))
	@echo "ncu-quick -> $(PROF_DIR)/ncu_report_quick-$(DIMTAG).ncu-rep"
	@$(PROF_CALC); \
	$(NCU) --set basic -k "regex:$(NCU_SETUP_KERNELS)|$(NCU_LOOP_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(PROF_DIR)/ncu_report_quick-$(DIMTAG) -f $(call FIX,$(BIN)) $(PROF_RUNARGS)

# Full "sem rede de proteção": perfila TODAS as invocações dos kernels do
# loop (NCU_KERNELS). Só use se souber que roda poucas vezes; senão o
# relatório fica enorme. Não inclui setInsideVertices (use ncu-setup).
ncu-full: $(BIN)
	@$(CHECK_PROF)
	@$(call MKDIR_PATH,$(PROF_DIR))
	@echo "ncu-full -> $(PROF_DIR)/ncu_report_full-$(DIMTAG).ncu-rep"
	@$(PROF_CALC); \
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    -o $(PROF_DIR)/ncu_report_full-$(DIMTAG) -f $(call FIX,$(BIN)) $(PROF_RUNARGS)

nsys: $(BIN)
	@$(CHECK_PROF)
	@$(call MKDIR_PATH,$(PROF_DIR))
	@echo "nsys -> $(PROF_DIR)/nsys_report-$(DIMTAG).nsys-rep"
	@$(PROF_CALC); \
	$(NSYS) profile -o $(PROF_DIR)/nsys_report-$(DIMTAG) -f true --trace=cuda,nvtx,osrt \
	    $(call FIX,$(BIN)) $(PROF_RUNARGS)

# Remove so' a variante atual (ARCH/DEBUG correntes); os binarios das outras
# arquiteturas continuam em build/.
clean:
	@$(call RMDIR_PATH,$(OBJDIR))
	@$(call RMDIR_PATH,$(BIN))

# Remove a pasta build inteira (todas as arquiteturas).
clean-all:
	$(RMDIR)

help:
	@echo "make                                - compila p/ a GPU detectada; gera build/$(TARGET)-<arch>"
	@echo ""
	@echo "ARGS obrigatorio p/ run/ncu*/nsys, flags aceitas (qualquer ordem/quantidade):"
	@echo "  --blocksDim <x> <y> <z>           - fixa dimensoes exatas do grid de blocos (numBlocks = x*y*z)"
	@echo "  --threadsDim <x> <y> <z>          - fixa dimensoes exatas do bloco de threads (numThreads = x*y*z)"
	@echo "  --numBlocks <n>                   - total de blocos"
	@echo "  --numThreads <n>                  - total de threads por bloco"
	@echo "  --chunksDim <x> <y> <z>           - fixa a particao do grid de blocos em chunks (numChunks = x*y*z)"
	@echo "  --numChunks <n>                   - total de chunks (particao escolhida por bestPartition)"
	@echo "  --iter <n>                        - numero de iteracoes (define maxTime = n * deltaTime)"
	@echo "  --folder <dir>                    - pasta de saida dos dataOpt_*.txt"
	@echo "  --problemSize <n>                 - total de threads desejado; numBlocks e' recalculado (= n / numThreads)"
	@echo "  --vel <float>                     - velocidade do fluxo"
	@echo "  --time <float>                    - tempo maximo de simulacao (>= minTime)"
	@echo "  --scale <float>                   - fator de escala"
	@echo "  --deltaTime <float>               - passo de tempo"
	@echo "  --write <bool>                    - escreve saida (true/false ou 1/0)"
	@echo "  --object <nome>                   - nome do arquivo .obj (sem extensao)"
	@echo "  --deviceProperties                - mostra propriedades da GPU e sai"
	@echo "  -h, --help                        - mostra o help do binario e sai"
	@echo ""
	@echo "make run ARGS=\"--numBlocks 64 --numThreads 1024\""
	@echo ""
	@echo "Perfilamento -- recebe THREADSDIM (3 numeros) + TOTALTHREADS ou NUMBLOCKS;"
	@echo "o relatorio vai para $(PROF_DIR)/ nomeado pela distribuicao (ex.: ncu_report-16-16-1.ncu-rep):"
	@echo "  make ncu TOTALTHREADS=1048576 THREADSDIM=\"16 16 1\"     - fluidMovement/recalculateVelocities, limitado a NCU_LAUNCHES invocacoes"
	@echo "  make ncu NUMBLOCKS=4096 THREADSDIM=\"8 8 16\"            - idem, informando os blocos direto"
	@echo "  make ncu-setup TOTALTHREADS=4096 THREADSDIM=\"8 8 1\"    - setInsideVertices ISOLADO; use escala reduzida (e' O(totalThreads x nTriangulos))"
	@echo "  make ncu-quick TOTALTHREADS=1048576 THREADSDIM=\"16 16 1\" - todos os kernels com --set basic, rapido, p/ checagem inicial"
	@echo "  make ncu-full TOTALTHREADS=1048576 THREADSDIM=\"16 16 1\"  - TODAS as invocacoes do loop com --set full (relatorio pode ficar enorme)"
	@echo "  make nsys TOTALTHREADS=1048576 THREADSDIM=\"16 16 1\"    - execucao inteira com Nsight Systems"
	@echo "  ... NCU_LAUNCHES=20 NCU_SKIP=100                       - ajusta quantas invocacoes e a partir de qual pular"
	@echo "  ... PROF_DIR=relatorios                                - muda a pasta de saida dos relatorios"
	@echo "  ... ARGS=\"--vel 2.5\"                                   - flags extras anexadas a execucao"
	@echo "make factorial PROBLEM=\"100000 500000\" THREADS=\"128 256\" - limpa DATA_DIR e roda todas as combinacoes de PROBLEM x THREADS"
	@echo "make thread-factorial TOTALTHREADS=1048576 REPEAT=3      - p/ cada numThreads de THREADS que divida TOTALTHREADS, usa numBlocks=TOTALTHREADS/numThreads e varre XDIV x YDIV x ZDIV com x*y*z=numThreads"
	@echo "make thread-factorial TOTALTHREADS=1048576 FOLDER=\"cidia-main\" - mesma coisa, salvando os dataOpt_*.txt em cidia-main/ (default: data-thread-factorial)"
	@echo "make thread-factorial-experiment FOLDER=\"early-memory-tupi\"  - roda so as 3 configuracoes de EXP_DIMS (16x16x1, 512x1x1, 4x8x1), EXP_REPEAT=30 vezes, TOTALTHREADS=1048576"
	@echo "make thread-factorial-experiment EXP_DIMS=\"8,8,16 32,1,1\" EXP_REPEAT=10 - muda as configuracoes e o numero de repeticoes"
	@echo "make chunks-factorial TOTALTHREADS=4194304 THREADSDIM=\"8 8 16\" MAXCHUNKS=64 REPEAT=3"
	@echo "                                                          - fixa threadsDim e varre CXDIV x CYDIV x CZDIV com x*y*z <= MAXCHUNKS"
	@echo "                                                            (saidas em $(DATA_CF_DIR)/; use FOLDER=... para mudar)"
	@echo ""
	@echo "make arch                           - mostra arquitetura, maquina e nome do binario, sem compilar"
	@echo "make rebuild                        - recompila do zero a variante atual ($(VARIANT))"
	@echo "make MACHINE=tupi                   - rotula o binario pela maquina: build/$(TARGET)-tupi-<arch>"
	@echo "make thread-factorial ... FORCE_REBUILD=0 - nao recompila antes do experimento (default: recompila)"
	@echo "make DEBUG=1                        - build com debug de device (-G -g); binario vira $(TARGET)-<arch>-debug"
	@echo "make ARCH=sm_89                     - forca a arch (default: detectada via nvidia-smi; 'native' tambem e' resolvido)"
	@echo "make clean                          - remove so' a variante atual ($(VARIANT))"
	@echo "make clean-all                      - remove a pasta build inteira (todas as arquiteturas)"