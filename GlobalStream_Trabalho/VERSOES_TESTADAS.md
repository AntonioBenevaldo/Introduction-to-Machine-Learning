# Versões testadas

Em cada versão do Python, foi criado um ambiente novo, as bibliotecas foram instaladas pelo `requirements.txt`, e a análise e os 12 testes foram executados. Em todas, os resultados foram idênticos: Regressão Logística escolhida, limiar de 0,25, AUC de 0,984 no teste, precisão de 84,1%, recall de 98,1% e 2 grupos na segmentação.

| Python | numpy | pandas | scikit-learn | matplotlib | Testes |
|---|---|---|---|---|---|
| 3.9.23 | 2.0.2 | 2.3.3 | 1.6.1 | 3.9.4 | 12 de 12 |
| 3.10.21 | 2.2.6 | 2.3.3 | 1.7.2 | 3.10.9 | 12 de 12 |
| 3.11.15 | 2.4.6 | 3.0.6 | 1.9.1 | 3.11.2 | 12 de 12 |
| 3.12.11 | 2.5.3 | 3.0.6 | 1.9.1 | 3.11.2 | 12 de 12 |
| 3.13.7 | 2.5.3 | 3.0.6 | 1.9.1 | 3.11.2 | 12 de 12 |
| 3.14.7 | 2.5.3 | 3.0.6 | 1.9.1 | 3.11.2 | 12 de 12 |
| 3.9.23 com as versões mínimas do `requirements.txt` | 1.23.0 | 1.5.0 | 1.4.0 | 3.6.0 | 12 de 12 |

Os resultados em `resultados/` foram gerados com o Python 3.14.7, pelo inicializador `preparar_ambiente.py`, num ambiente criado do zero.

**Windows:** para cada versão do Python, de 3.9 a 3.14, foi verificado que todas as bibliotecas instalam no Windows 64 bits como pacotes prontos, sem precisar compilar nada. Os testes acima foram executados em Linux.
