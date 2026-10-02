"""Prepara o ambiente e executa o projeto do zero.

Funciona com qualquer Python 3.9 ou mais novo (inclusive o 3.14), no Windows, macOS ou Linux.
No Windows, e chamado por INICIAR_WINDOWS.bat; nos demais sistemas: python3 preparar_ambiente.py

Etapas: cria um ambiente virtual proprio (.venv + versao do Python, por exemplo .venv314),
instala as bibliotecas do requirements.txt, executa a analise e os testes, e abre o painel.
Nenhuma pasta e apagada; o registro completo fica em logs/instalacao_e_execucao.txt.
"""
import datetime
import json
import os
import platform
import subprocess
import sys
import venv
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
MINIMO = (3, 9)


def python_do_ambiente(envdir):
    return envdir / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def main():
    if sys.version_info < MINIMO:
        print(f'Este projeto precisa de Python {MINIMO[0]}.{MINIMO[1]} ou mais novo. '
              f'Encontrado: {platform.python_version()}.')
        return 2
    versao = f'{sys.version_info[0]}{sys.version_info[1]}'
    envdir = BASE / f'.venv{versao}'          # um ambiente por versao do Python
    python = python_do_ambiente(envdir)
    logs = BASE / 'logs'
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    try:
        if not python.is_file():
            if envdir.exists():
                raise RuntimeError(f'A pasta {envdir.name} esta incompleta. Renomeie-a e execute novamente.')
            print(f'Criando o ambiente {envdir.name} com Python {platform.python_version()}...', flush=True)
            venv.EnvBuilder(with_pip=True).create(envdir)
        if not python.is_file():
            raise RuntimeError(f'Nao foi possivel criar o ambiente {envdir.name}.')

        etapas = [
            # (titulo, argumentos, obrigatoria?)
            ('Atualizando o pip', ['-m', 'pip', 'install', '--upgrade', 'pip'], False),
            ('Instalando as bibliotecas (a primeira vez exige internet)',
             ['-m', 'pip', 'install', '--prefer-binary', '-r', str(BASE / 'requirements.txt')], True),
            ('Conferindo as bibliotecas', ['-m', 'pip', 'check'], True),
            ('Executando a analise', ['-u', str(BASE / 'globalstream.py')], True),
            ('Executando os testes', ['-u', str(BASE / 'testes.py')], True),
        ]
        with (logs / 'instalacao_e_execucao.txt').open('w', encoding='utf-8') as log:
            log.write('Execucao local: ' + datetime.datetime.now().astimezone().isoformat() + '\n')
            log.write(f'Python do inicializador: {platform.python_version()} ({platform.platform()})\n')
            for titulo, args, obrigatoria in etapas:
                print('\n' + titulo, flush=True)
                log.write('\n' + titulo + '\n')
                log.flush()
                proc = subprocess.Popen([str(python), *args], cwd=BASE, env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if proc.wait() != 0:
                    if obrigatoria:
                        raise RuntimeError(titulo + ' falhou. Consulte logs/instalacao_e_execucao.txt.')
                    print('(etapa opcional nao concluida; continuando)', flush=True)

        (BASE / 'resultados' / 'ambiente_local.json').write_text(json.dumps({
            'python': platform.python_version(), 'sistema': platform.platform(), 'arquitetura': platform.machine(),
            'ambiente': envdir.name, 'resultado': 'Analise e testes concluidos'}, ensure_ascii=False, indent=2),
            encoding='utf-8')
        webbrowser.open((BASE / 'resultados' / 'painel_resultados.html').as_uri())
        print('\nConcluido. Painel aberto no navegador; registro em logs/instalacao_e_execucao.txt.')
        return 0
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print('\nERRO:', exc)
        print('Envie a mensagem de erro ou o arquivo logs/instalacao_e_execucao.txt para diagnostico.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
