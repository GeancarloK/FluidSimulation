@echo off
REM ============================================================
REM Compila o projeto e roda a bateria de perfilagem gerando
REM arquivos do Nsight Systems e do Nsight Compute (um perfil
REM separado para cada kernel) para cada combinacao blocos x threads.
REM
REM Estrutura de pastas gerada:
REM   profiles\nsys\                       nsys_<total>_<blocks>_<threads>.nsys-rep
REM   profiles\ncu_<kernelA>\              ncu_<total>_<blocks>_<threads>.ncu-rep
REM   profiles\ncu_<kernelB>\              ncu_<total>_<blocks>_<threads>.ncu-rep
REM
REM NOTA TECNICA: este script NAO usa "setlocal EnableDelayedExpansion"
REM e chama nsys/ncu com "call", porque os wrappers dessas ferramentas
REM executam endlocal / transferem controle no processo pai.
REM
REM IMPORTANTE: rode em um terminal aberto como ADMINISTRADOR,
REM caso contrario o ncu falha com ERR_NVGPUCTRPERM.
REM ============================================================

set ARCH=sm_86
set OUTPUT=FluidSimulation.exe
set OPT_FLAGS=-O3

REM Numero de blocos no ponto de partida (com 1024 threads por bloco)
set BASE_LIST=1024 2048 4096

set PROFILE_DIR=profiles
set DATA_DIR=data

REM Numero de iteracoes da simulacao (primeiro argumento do executavel)
set MAX_ITER=5

REM Kernels a perfilar com o Nsight Compute (um perfil por kernel)
set KERNEL_A=recalculateVelocities
set KERNEL_B=fluidMovement

REM Quantas invocacoes de cada kernel o ncu deve capturar
set NCU_COUNT=3

REM Conjunto de metricas: "basic" (rapido) ou "full" (40 passes)
set NCU_SET=basic

REM Caminho explicito do nvcc (v13.2 - v13.3 tem bug de crash no cudafe++
REM com o toolset do VS 2026)
set NVCC="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe"

REM Subpastas de saida
set NSYS_DIR=%PROFILE_DIR%\nsys
set NCU_DIR_A=%PROFILE_DIR%\ncu_%KERNEL_A%
set NCU_DIR_B=%PROFILE_DIR%\ncu_%KERNEL_B%

REM ============================================================
REM ETAPA 1: Verificacao de ferramentas
REM ============================================================
echo ============================================
echo Verificando ferramentas...
echo ============================================

if not exist %NVCC% (
    echo ERRO: nvcc nao encontrado em %NVCC%
    echo Confira se o CUDA Toolkit v13.2 esta instalado nesse caminho,
    echo ou ajuste a variavel NVCC neste script.
    pause
    exit /b 1
)

set HAS_NSYS=1
where nsys >nul 2>nul
if errorlevel 1 set HAS_NSYS=0
if "%HAS_NSYS%"=="0" echo AVISO: nsys nao encontrado. Nsight Systems sera pulado.

set HAS_NCU=1
where ncu >nul 2>nul
if errorlevel 1 set HAS_NCU=0
if "%HAS_NCU%"=="0" echo AVISO: ncu nao encontrado. Nsight Compute sera pulado.

echo.
%NVCC% --version
echo.

REM ============================================================
REM ETAPA 2: Compilacao
REM ============================================================
echo ============================================
echo Compilando %OUTPUT% para arquitetura %ARCH%...
echo ============================================
%NVCC% -arch=%ARCH% %OPT_FLAGS% -std=c++17 -allow-unsupported-compiler main.cu kernels.cu utils.cu mesh.cu -o %OUTPUT%

if errorlevel 1 (
    echo.
    echo ERRO na compilacao. Veja as mensagens acima.
    pause
    exit /b 1
)

echo.
echo Compilacao concluida com sucesso: %OUTPUT%
echo.

REM ============================================================
REM Limpa resultados anteriores e recria a arvore de pastas
REM (rmdir /s remove tambem as subpastas, ao contrario de del /q)
REM ============================================================
echo Limpando pastas de resultados anteriores...

if exist "%PROFILE_DIR%" rmdir /s /q "%PROFILE_DIR%"
if exist "%DATA_DIR%" rmdir /s /q "%DATA_DIR%"

mkdir "%PROFILE_DIR%"
mkdir "%NSYS_DIR%"
mkdir "%NCU_DIR_A%"
mkdir "%NCU_DIR_B%"
mkdir "%DATA_DIR%"

echo Pastas recriadas:
echo   %NSYS_DIR%
echo   %NCU_DIR_A%
echo   %NCU_DIR_B%
echo   %DATA_DIR%
echo.

echo ============================================
echo Iniciando bateria de perfilagem...
echo ============================================
echo.

REM ============================================================
REM ETAPA 3: Laco externo - percorre a lista de bases
REM ============================================================
:BaseLoop
if "%BASE_LIST%"=="" goto :AllDone

REM Retira o primeiro item da lista e guarda o resto
for /f "tokens=1,*" %%a in ("%BASE_LIST%") do (
    set CURBASE=%%a
    set BASE_LIST=%%b
)

set /a blocks=%CURBASE%
set /a threads=1024
set /a total=%blocks% * %threads%

echo --- Trabalho total fixo: %total% threads (base: %CURBASE% blocos x 1024) ---
echo.

REM ============================================================
REM Laco interno - threads 1024 -> 1, blocos dobrando
REM ============================================================
:SweepLoop
if %threads% LSS 1 goto :BaseLoop

set TAG=%total%_%blocks%_%threads%
echo [%TAG%] maxIter=%MAX_ITER% blocos=%blocks% threads=%threads%

if "%HAS_NSYS%"=="1" (
    echo   - Nsight Systems...
    call nsys profile --force-overwrite true -o "%NSYS_DIR%\nsys_%TAG%" %OUTPUT% %MAX_ITER% %blocks% %threads%
)
@echo off

if "%HAS_NCU%"=="1" (
    echo   - Nsight Compute [%KERNEL_A%]...
    call ncu --set %NCU_SET% -k %KERNEL_A% -c %NCU_COUNT% --force-overwrite -o "%NCU_DIR_A%\ncu_%TAG%" %OUTPUT% %MAX_ITER% %blocks% %threads%
)
@echo off

if "%HAS_NCU%"=="1" (
    echo   - Nsight Compute [%KERNEL_B%]...
    call ncu --set %NCU_SET% -k %KERNEL_B% -c %NCU_COUNT% --force-overwrite -o "%NCU_DIR_B%\ncu_%TAG%" %OUTPUT% %MAX_ITER% %blocks% %threads%
)
@echo off
echo.

set /a threads=%threads% / 2
set /a blocks=%blocks% * 2
goto :SweepLoop

:AllDone
echo ============================================
echo Bateria concluida.
echo ============================================
echo.
echo --- %NSYS_DIR% ---
dir /b "%NSYS_DIR%"
echo.
echo --- %NCU_DIR_A% ---
dir /b "%NCU_DIR_A%"
echo.
echo --- %NCU_DIR_B% ---
dir /b "%NCU_DIR_B%"
pause
exit /b 0