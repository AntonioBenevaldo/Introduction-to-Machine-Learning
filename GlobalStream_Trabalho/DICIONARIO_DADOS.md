# Dicionário da planilha fornecida

Fonte: `globalstream_data.xlsx`, aba `Sheet1`, A1:I1001, com 1.000 registros e nove colunas. A planilha original foi copiada sem alterações; seu SHA-256 está em `resultados/resumo.json`. Nenhum dado foi gerado artificialmente para treinar ou avaliar os modelos.

| Coluna | Papel e observações |
|---|---|
| ID_Cliente | Identificador; 1.000 valores distintos. Somente chave, nunca preditor. |
| Meses_Contrato | Inteiro de 1 a 47. Prazo contratual versus tempo de relacionamento não esclarecido. Não truncar em 24 ou 36 sem definição. |
| Valor_Mensal | Número de 30,04 a 59,88. Unidade monetária não especificada no arquivo. |
| Frequencia_Acesso_Semanal | Inteiro de 0 a 7. Medida de frequência; não há definição se dias ou acessos por semana. Não equivale a horas assistidas. |
| Tickets_Suporte | Inteiro de 0 a 9. Contagem de tickets, mas janela de acumulação não informada. |
| Plano | Basico (327), Standard (333), Premium (340). Categoria nominal. |
| Genero_Favorito | Documentarios (267), Desporto (251), Filmes (250), Series (232). Categoria nominal. |
| Dispositivo_Principal | Web (349), Mobile (329), Smart TV (322). Categoria nominal. |
| Churn | Alvo binário: 542 valores 1 e 458 valores 0. Convenção do estudo: 1=cancelamento e 0=permanência. Mês/horizonte do evento não documentado. |

## Ausências, duplicatas e regras

Não há ausências ou IDs duplicados no arquivo recebido. O programa remove cópias exatas caso existam, rejeita IDs conflitantes e rótulos ausentes/inválidos. Números negativos, não finitos ou texto não numérico viram NaN; inteiros fracionários em meses/frequência/tickets também viram NaN. Valores totalmente ausentes em um atributo causam erro explícito. As categorias são aparadas nas extremidades; vazios viram NaN. A fonte entregue não sofreu correções de valores por essas regras.

A mediana numérica e a moda categórica são aprendidas somente no treino de cada fold. Não houve valores a imputar na base desta execução. Para demonstrar o mecanismo, o teste unitário usa uma cópia em memória com ausências introduzidas: isso não modifica a base usada no modelo.

## Outliers e encoding

O IQRClipper aprende Q1 e Q3 no treino, depois da imputação. Limites: Q1−1,5×IQR e Q3+1,5×IQR. Se IQR=0, a variável não é limitada. O transform apenas aplica limites aprendidos. Nenhum valor do treino foi alterado neste estudo; os limites e contagens constam em `outliers_treino.csv`.

Numéricas recebem StandardScaler. As três categorias recebem OneHotEncoder sem ordem artificial; categorias não vistas no treino são ignoradas no encoding, preservando o número de colunas. Essa tolerância não elimina a necessidade de monitorar categorias novas.

## Informações temporais ausentes

Não há data de observação, data do cancelamento, mês de corte ou histórico individual de 24 linhas por cliente. Meses_Contrato não substitui esse calendário. Também não há tempo de tela. Por isso, a análise não cria série temporal, médias mensais ou rótulos do mês seguinte. A generalização medida é entre clientes da mesma base, sob a hipótese de atributos disponíveis antes do alvo, que precisa de confirmação externa.
