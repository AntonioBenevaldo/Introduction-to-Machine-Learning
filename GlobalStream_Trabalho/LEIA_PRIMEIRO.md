# GlobalStream — Parte Prática

Segmentação de clientes e predição de churn com a base oficial do enunciado (`globalstream_data.xlsx`, 1.000 clientes).
O projeto funciona com **qualquer Python 3.9 ou mais novo, inclusive o 3.14**, no Windows, no macOS, no Linux e no Google Colab.

## Arquivos

| Arquivo | Para que serve |
|---|---|
| `globalstream.py` | O projeto: limpeza, análise exploratória, segmentação, classificação, avaliação e análise crítica, organizado nas seções do enunciado |
| `dados/globalstream_data.xlsx` | Base oficial, sem alterações |
| `resultados/` | Gráficos, tabelas, painel e a saída completa da execução |
| `requirements.txt` | Bibliotecas necessárias (versões mínimas) |
| `testes.py` | 12 testes automáticos |
| `INICIAR_WINDOWS.bat` / `INICIAR_MAC_LINUX.sh` | Instalam as bibliotecas e executam tudo |
| `ABRIR_RESULTADOS.bat` | Abre o painel sem instalar nada |
| `COBERTURA_REQUISITOS.md` | Onde cada item do enunciado está no código |
| `DICIONARIO_DADOS.md` | Descrição das colunas da base |
| `VERSOES_TESTADAS.md` | Versões do Python e das bibliotecas testadas |

## Ver os resultados sem instalar nada

Extraia o zip e dê dois cliques em `ABRIR_RESULTADOS.bat`, ou abra `resultados/painel_resultados.html` no navegador.
A saída completa, com todas as tabelas e a análise crítica, está em `resultados/saida_execucao.txt`.

## Executar

**Windows:** extraia o zip e dê dois cliques em `INICIAR_WINDOWS.bat`. Na primeira vez é preciso estar com internet para instalar as bibliotecas.
O inicializador usa o Python mais novo instalado, cria um ambiente próprio na pasta (por exemplo, `.venv314` para o Python 3.14), executa a análise e os testes e abre o painel. Nenhuma pasta é apagada; o registro fica em `logs/instalacao_e_execucao.txt`.

**Pelo terminal (qualquer sistema),** dentro da pasta extraída:

```
python -m pip install -r requirements.txt
python globalstream.py --abrir
python testes.py
```

No Windows, se `python` não for reconhecido, use `py` no lugar. No macOS e no Linux, use `python3` ou `./INICIAR_MAC_LINUX.sh`.

**Google Colab:** envie `globalstream.py`, `requirements.txt` e `globalstream_data.xlsx` e execute:

```
!pip install -q -r requirements.txt
!python globalstream.py --arquivo globalstream_data.xlsx
```

## Principais resultados

- **Base:** 1.000 clientes e 54,2% de churn; sem ausentes, duplicatas ou outliers. 220 clientes têm mais de 36 meses de contrato, o que precisa ser confirmado com a empresa.
- **Exploração:** o churn cresce com os tickets de suporte (correlação de 0,76) e cai com a frequência de acesso (−0,31). Plano, gênero e dispositivo não diferem de forma significativa.
- **Segmentação (K-Means + PCA):** 2 grupos, com separação fraca (silhouette de 0,124). O grupo com mais tickets e menos acessos cancela 93,3%; o outro, 20,3%.
- **Predição:** seis classificadores comparados em validação cruzada (Random Forest, Gradient Boosting, SVM, k-NN, Regressão Logística e Dummy). A Regressão Logística teve a maior Average Precision (0,981).
- **Teste (200 clientes nunca vistos):** AUC de 0,984. Com o limiar ajustado ao custo dos erros (0,25), a precisão foi de 84,1% e o recall de 98,1%.
- **Análise crítica:** Seção 5 do código e `resultados/analise_critica.txt`.

## Entrega

- **Parte Prática:** `globalstream.py`, com a pasta `dados/`.
- **Parte Teórica:** relatório em PDF, entregue junto com esta pasta.
- **Vídeo pitch:** link no final do relatório em PDF.
