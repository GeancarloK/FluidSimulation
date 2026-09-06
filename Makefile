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
# Pasta onde o alvo "thread-factorial" joga os dataOpt_*.txt.
# Pode ser sobrescrita com FOLDER="cidia-main".
FOLDER      ?= data-thread-factorial
DATA_TF_DIR ?= $(FOLDER)

# Listas testadas no fatorial -- todas as combinacoes de PROBLEM x THREADS
# serao executadas. Sobrescreva na linha de comando se quiser, ex:
#   make factorial PROBLEM="1000 5000" THREADS="128 256"
PROBLEM ?= 1048576 4194304 16777216
THREADS ?= 32 64 128 256 512 1024

XDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
YDIV ?= 1 2 4 8 16 32 64 128 256 512 1024
ZDIV ?= 1 2 4 8 16 32 64

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

# Binarios por arquitetura em build/, objetos em subpasta propria por
# arquitetura (+ sufixo -debug), para que trocar ARCH ou DEBUG nao reaproveite
# objetos compilados para outra configuracao.
VARIANT := $(ARCH)$(if $(filter 1,$(DEBUG)),-debug)
OBJDIR  := $(BUILD)/obj-$(VARIANT)
BIN     := $(BUILD)/$(TARGET)-$(VARIANT)$(EXE)
OBJS    := $(patsubst %.cu,$(OBJDIR)/%.o,$(SRCS))

ifeq ($(DEBUG),1)
    NVCCFLAGS ?= -G -g -std=$(STD) -arch=$(ARCH) $(HOSTFLAGS)
else
    NVCCFLAGS ?= -O3   -std=$(STD) -arch=$(ARCH) $(HOSTFLAGS)
endif

.PHONY: all run factorial thread-factorial clean clean-all help arch ncu ncu-setup ncu-quick ncu-full nsys
.DEFAULT_GOAL := all

all: $(BIN)
	@echo "Binario: $(BIN)  (arch=$(ARCH)$(if $(DETECTED_ARCH),, -- deteccao falhou, usando FALLBACK_ARCH))"

$(BIN): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

$(OBJDIR)/%.o: %.cu $(HDRS) | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJDIR):
	$(MKDIR)

# Mostra o que foi detectado sem compilar nada.
.PHONY: arch
arch:
	@echo "DETECTED_ARCH = $(if $(DETECTED_ARCH),$(DETECTED_ARCH),(nao detectada))"
	@echo "ARCH          = $(ARCH)"
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

factorial: $(BIN)
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

TIMERUN ?= 0.03

thread-factorial: $(BIN)
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

# Perfila os kernels do LOOP (fluidMovement/recalculateVelocities), que são
# baratos por invocação — pode (e deve) rodar na escala real do problema
# (ex.: ARGS="--numBlocks 1024 --numThreads 1024"). Limitado a NCU_LAUNCHES
# invocações pra não gerar relatórios gigantes que travam o ncu-ui ao abrir.
ncu: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(BUILD)/ncu_report -f $(call FIX,$(BIN)) $(ARGS)

# Perfila SÓ o setInsideVertices, separado do loop. Ele roda 1x só, mas faz
# O(totalThreads x nTriangulos do .obj) trabalho, então --set full nele em
# escala real explode o tempo por passe do kernel replay e derruba o
# profiler (LaunchFailed). Use ARGS reduzido aqui (ex.: "--numBlocks 64
# --numThreads 64") — as métricas por-thread continuam representativas.
ncu-setup: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_SETUP_KERNELS)" \
	    -o $(BUILD)/ncu_report_setup -f $(call FIX,$(BIN)) $(ARGS)

# Sanidade rápida: set leve (basic), sem coleta pesada, útil para conferir
# se o binário/kernels estão sendo capturados antes de rodar o full.
# Inclui todos os kernels (setup + loop) pois o overhead do basic é baixo.
ncu-quick: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set basic -k "regex:$(NCU_SETUP_KERNELS)|$(NCU_LOOP_KERNELS)" \
	    --launch-count $(NCU_LAUNCHES) --launch-skip $(NCU_SKIP) \
	    -o $(BUILD)/ncu_report_quick -f $(call FIX,$(BIN)) $(ARGS)

# Full "sem rede de proteção": perfila TODAS as invocações dos kernels do
# loop (NCU_KERNELS). Só use se souber que roda poucas vezes; senão o
# relatório fica enorme. Não inclui setInsideVertices (use ncu-setup).
ncu-full: $(BIN)
	@$(CHECK_ARGS)
	$(NCU) --set $(NCU_SET) -k "regex:$(NCU_KERNELS)" \
	    -o $(BUILD)/ncu_report_full -f $(call FIX,$(BIN)) $(ARGS)

nsys: $(BIN)
	@$(CHECK_ARGS)
	$(NSYS) profile -o $(BUILD)/nsys_report -f true --trace=cuda,nvtx,osrt $(call FIX,$(BIN)) $(ARGS)

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
	@echo "make ncu ARGS=\"--numBlocks 1024 --numThreads 1024\"       - profila fluidMovement/recalculateVelocities (escala real), limitado a NCU_LAUNCHES invocacoes"
	@echo "make ncu-setup ARGS=\"--numBlocks 64 --numThreads 64\"     - profila setInsideVertices ISOLADO; use escala reduzida (ele e' O(totalThreads x nTriangulos), full na escala real trava o profiler)"
	@echo "make ncu-quick ARGS=\"--numBlocks 64 --numThreads 1024\"   - profila todos os kernels com --set basic, rapido, para checagem inicial"
	@echo "make ncu-full ARGS=\"--numBlocks 64 --numThreads 1024\"    - profila TODAS as invocacoes do loop com --set full (relatorio pode ficar enorme)"
	@echo "make ncu ARGS=\"--numBlocks 1024 --numThreads 1024\" NCU_LAUNCHES=20 NCU_SKIP=100 - ajusta quantas invocacoes e a partir de qual pular"
	@echo "make nsys ARGS=\"--numBlocks 64 --numThreads 1024\"        - profila a execucao inteira com Nsight Systems"
	@echo "make factorial PROBLEM=\"100000 500000\" THREADS=\"128 256\" - limpa DATA_DIR e roda todas as combinacoes de PROBLEM x THREADS"
	@echo "make thread-factorial TOTALTHREADS=1048576 REPEAT=3      - p/ cada numThreads de THREADS que divida TOTALTHREADS, usa numBlocks=TOTALTHREADS/numThreads e varre XDIV x YDIV x ZDIV com x*y*z=numThreads"
	@echo "make thread-factorial TOTALTHREADS=1048576 FOLDER=\"cidia-main\" - mesma coisa, salvando os dataOpt_*.txt em cidia-main/ (default: data-thread-factorial)"
	@echo "make arch                           - mostra a arquitetura detectada e o nome do binario, sem compilar"
	@echo "make DEBUG=1                        - build com debug de device (-G -g); binario vira $(TARGET)-<arch>-debug"
	@echo "make ARCH=sm_89                     - forca a arch (default: detectada via nvidia-smi; 'native' tambem e' resolvido)"
	@echo "make clean                          - remove so' a variante atual ($(VARIANT))"
	@echo "make clean-all                      - remove a pasta build inteira (todas as arquiteturas)"