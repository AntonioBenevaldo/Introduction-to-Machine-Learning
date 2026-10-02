@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
rem Usa o Python 3 mais novo instalado (launcher "py"); se nao houver, tenta "python".
rem Funciona com qualquer Python 3.9 ou mais novo, inclusive o 3.14.
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 goto usar_py
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if not errorlevel 1 goto usar_python
echo Nao foi encontrado Python 3.9 ou mais novo.
echo Instale o Python em https://www.python.org/downloads/ e marque "Add python.exe to PATH".
pause
exit /b 1
:usar_py
py -3 preparar_ambiente.py
goto fim
:usar_python
python preparar_ambiente.py
:fim
if errorlevel 1 goto falha
pause
exit /b 0
:falha
echo A execucao nao foi concluida. Veja a mensagem acima e o arquivo logs\instalacao_e_execucao.txt.
pause
exit /b 1
