# Onde cada item do enunciado está

A Parte Prática está em `globalstream.py`, dividida nas mesmas seções do enunciado. Ao executar, cada seção imprime seus resultados na tela; a saída completa fica em `resultados/saida_execucao.txt`.

| Item do enunciado | Onde está no código | Resultado gerado |
|---|---|---|
| **1. Exploração e limpeza dos dados** | | |
| Tratamento de dados ausentes | `preparar_base()` transforma valores impossíveis em ausentes; `preprocessar()` imputa mediana (numéricas) e moda (categóricas), aprendidas só no treino | `auditoria_dados.json`; teste `test_imputacao_prevista_e_categoria_nova` |
| Encoding de variáveis categóricas | `preprocessar()`: One-Hot Encoding das 3 categóricas | — |
| Análise e tratamento de outliers | Classe `IQRClipper` (limites Q1 − 1,5×IQR e Q3 + 1,5×IQR aprendidos no treino); `explorar()` | `outliers_treino.csv`; figura `02_suporte_outliers.png` |
| Análise exploratória inicial | `explorar()`: distribuições, churn por variável e por categoria (qui-quadrado) e correlação | `estatisticas_treino.csv`; figuras `01`, `02`, `02b` e `02c` |
| **2. Segmentação de clientes** | | |
| Técnica de agrupamento | `segmentar()`: K-Means, k de 2 a 6 escolhido pela silhouette | `selecao_clusters.csv` |
| Visualização dos clusters (PCA) | `segmentar()`: projeção em 2 componentes | figura `03_clusters_pca.png` |
| Interpretação dos grupos | `segmentar()`: perfil médio, categorias predominantes, taxa de churn de cada grupo e ação sugerida | `perfis_clusters.csv` |
| **3. Predição de churn** | | |
| Classificador supervisionado | `comparar_classificadores()` | `comparacao_validacao_cruzada.csv` |
| Random Forest, SVM, k-NN ou Gradient Boosting | `candidatos()`: os quatro, mais Regressão Logística e Dummy como referências | — |
| Treinamento, validação e prevenção de overfitting | `dividir()` (600 treino / 200 validação / 200 teste); validação cruzada em 5 folds; hiperparâmetros que limitam a complexidade; comparação treino × validação (`gap_treino_cv`) | `particoes.csv` |
| **4. Avaliação do modelo** | | |
| Matriz de confusão | `avaliar()` | figura `04_avaliacao.png`; `relatorio_classificacao.txt` |
| Métricas de classificação | `metricas()` e `avaliar()`: acurácia, precisão, recall, F1, ROC AUC e Average Precision | `metricas_teste.csv` |
| Curva ROC e AUC | `avaliar()`, com intervalo de confiança por bootstrap | figura `04_avaliacao.png`; `resumo.json` |
| Discussão crítica sobre desempenho e utilidade | `analise_critica()`, item 5.1; limiar escolhido pelo custo dos erros em `escolher_limiar()` | `analise_critica.txt`; figura `05_comparacao_limiar.png` |
| **5. Análise crítica** | | |
| Comparação entre abordagens e resultados | `analise_critica()`, item 5.2: classificadores, limiar padrão × ajustado, segmentação × modelo e dependência dos tickets | `analise_critica.txt` |
| Propostas de melhoria futura | `analise_critica()`, item 5.4 | `analise_critica.txt` |
| Limitações do pipeline | `analise_critica()`, item 5.3 | `analise_critica.txt` |
