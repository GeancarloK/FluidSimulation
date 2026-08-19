@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM Compila o projeto e roda o executavel para varias combinacoes
REM de (numBlocks x numThreads) mantendo o trabalho TOTAL fixo.
REM
REM Cada valor em BASE_BLOCKS e o numero de blocos no ponto de
REM partida, quando threads = 1024. A partir dai:
REM     threads: 1024, 512, 256, ..., 2, 1
REM     blocks : base, base*2, base*4, ..., base*1024
REM
REM Exemplo com base = 4096:
REM     4096 blocos x 1024 threads
REM     8192 blocos x  512 threads
REM    16384 blocos x  256 threads
REM     ...
REM  4194304 blocos x    1 thread
REM ============================================================
REM Arquitetura da GPU: RTX 3070 (Ampere) = sm_86
set ARCH=sm_86
set OUTPUT=FluidSimulation.exe
set OPT_FLAGS=-O3
REM Numero de blocos no ponto de partida (com 1024 threads por bloco)
set BASE_BLOCKS=4096
REM Caminho explicito do nvcc (v13.2 - v13.3 tem bug de crash no cudafe++
REM com o toolset do VS 2026)
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"
REM ============================================================
REM ETAPA 1: Compilacao
REM ============================================================
echo ============================================
echo Verificando se o nvcc (v13.2) esta disponivel...
echo ============================================
if not exist %NVCC% (
    echo.
    echo ERRO: nvcc nao encontrado em %NVCC%
    echo Confira se o CUDA Toolkit v13.2 esta instalado nesse caminho,
    echo ou ajuste a variavel NVCC neste script.
    echo.
    pause
    exit /b 1
)
%NVCC% --version
echo.
echo ============================================
echo Compilando %OUTPUT% para arquitetura %ARCH%...
echo ============================================
%NVCC% -arch=%ARCH% %OPT_FLAGS% -std=c++17 -allow-unsupported-compiler main.cu kernels.cu utils.cu mesh.cu -o %OUTPUT%
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERRO na compilacao. Veja as mensagens acima.
    pause
    exit /b %ERRORLEVEL%
)
echo.
echo Compilacao concluida com sucesso: %OUTPUT%
echo.
REM ============================================================
REM ETAPA 2: Bateria de execucoes
REM ============================================================
echo ============================================
echo Iniciando bateria de execucoes...
echo ============================================
echo.
for %%B in (%BASE_BLOCKS%) do (
    call :RunSweep %%B
)
echo ============================================
echo Bateria de execucoes concluida.
echo ============================================
pause
exit /b 0
REM ============================================================
REM Subrotina: recebe o numero de blocos base (para 1024 threads)
REM e percorre threads 1024 -> 1, dobrando os blocos a cada passo.
REM O produto blocks*threads permanece constante.
REM ============================================================
:RunSweep
set /a blocks=%~1
set /a threads=1024
set /a total=blocks * threads
echo --- Trabalho total fixo: %total% threads (base: %~1 blocos x 1024) ---
:RunSweep_loop
if %threads% LSS 1 goto :RunSweep_end
echo Executando: %OUTPUT% %blocks% %threads%   (total=%total%)
%OUTPUT% %blocks% %threads%
set /a threads=threads / 2
set /a blocks=blocks * 2
goto :RunSweep_loop
:RunSweep_end
echo.
goto :eof