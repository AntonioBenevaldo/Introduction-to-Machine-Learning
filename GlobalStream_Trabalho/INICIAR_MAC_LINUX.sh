#!/bin/sh
# macOS / Linux: prepara o ambiente, executa a analise e os testes (Python 3.9 ou mais novo).
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
exec "$PY" preparar_ambiente.py
