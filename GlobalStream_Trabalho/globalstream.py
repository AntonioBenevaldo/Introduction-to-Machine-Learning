"""
GlobalStream - Otimização de Retenção e Segmentação de Clientes
Introduction to Machine Learning - Parte Prática

COMO EXECUTAR (qualquer Python 3.9 ou mais novo, inclusive o 3.14)
    python -m pip install -r requirements.txt
    python globalstream.py             # roda tudo e grava a pasta resultados/
    python globalstream.py --abrir     # idem, e abre o painel no navegador
    No Windows, basta dar dois cliques em INICIAR_WINDOWS.bat.

ORGANIZAÇÃO DO ARQUIVO (segue os itens do enunciado)
    1. Exploração e limpeza dos dados ... preparar_base(), preprocessar(), explorar()
    2. Segmentação de clientes .......... segmentar()
    3. Predição de churn ................ comparar_classificadores()
    4. Avaliação do modelo .............. avaliar()
    5. Análise crítica .................. analise_critica()

BASE: dados/globalstream_data.xlsx (aba Sheet1), 1.000 clientes, uma linha por cliente.
A planilha não tem datas: a avaliação classifica o Churn registrado e não comprova,
sozinha, previsão para o próximo mês (ver Seção 5).
"""
import sys

if sys.version_info < (3, 9):
    raise SystemExit('Este projeto precisa de Python 3.9 ou mais novo. Versão encontrada: '
                     + sys.version.split()[0])

import argparse
import hashlib
import html
import json
import platform
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # gera as figuras direto em arquivo, sem abrir janelas
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------------
# Configuração geral
# ----------------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
SEED = 42                       # semente fixa: mesma divisão e mesmos resultados a cada execução
NUM = ['Meses_Contrato', 'Valor_Mensal', 'Frequencia_Acesso_Semanal', 'Tickets_Suporte']
CAT = ['Plano', 'Genero_Favorito', 'Dispositivo_Principal']
ALVO = 'Churn'                  # 1 = cancelou, 0 = permaneceu
CUSTO_FP = 20                   # custo ilustrativo de um desconto desnecessário (falso positivo)
CUSTO_FN = 100                  # custo ilustrativo de um cancelamento ignorado (falso negativo)
CORES = ['#127565', '#3274A1', '#E3A03B', '#9B5D91', '#73804A', '#BA6555']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': 'white'})


def pct(x, casas=1):
    """0.8412 -> '84,1%' (formato brasileiro)."""
    return f'{100 * x:.{casas}f}%'.replace('.', ',')


def num(x, casas=3):
    """0.9838 -> '0,984' (formato brasileiro)."""
    return f'{x:.{casas}f}'.replace('.', ',')


class Relatorio:
    """Mostra o texto na tela e guarda uma cópia em resultados/saida_execucao.txt."""

    def __init__(self):
        self.linhas = []

    def __call__(self, texto=''):
        print(texto, flush=True)
        self.linhas.append(str(texto))

    def titulo(self, texto):
        self('')
        self('=' * 78)
        self(texto)
        self('=' * 78)

    def tabela(self, df, casas=3):
        self(df.to_string(float_format=lambda v: f'{v:.{casas}f}'))

    def salvar(self, caminho):
        caminho.write_text('\n'.join(self.linhas) + '\n', encoding='utf-8')


def salvar_fig(fig, saida, nome):
    fig.tight_layout()
    fig.savefig(saida / nome, dpi=160, bbox_inches='tight')
    plt.close(fig)


def json_dump(obj, path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                               default=lambda x: x.item() if hasattr(x, 'item') else str(x)),
                    encoding='utf-8')


def onehot():
    """OneHotEncoder compatível com versões antigas e novas do scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # scikit-learn >= 1.2
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)         # versões anteriores


# ==================================================================================
# 1. EXPLORAÇÃO E LIMPEZA DOS DADOS
# ==================================================================================
class IQRClipper(BaseEstimator, TransformerMixin):
    """Tratamento de outliers: limita valores fora de [Q1 - 1,5*IQR ; Q3 + 1,5*IQR].

    Os limites são aprendidos SOMENTE no fit (dados de treino) e apenas aplicados no
    transform, evitando vazamento de informação do teste. Se IQR = 0, a variável não
    é limitada, para preservar contagens raras.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        q1, q3 = np.quantile(a, [.25, .75], axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.factor * iqr, -np.inf)
        self.upper_ = np.where(iqr > 0, q3 + self.factor * iqr, np.inf)
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


def preparar_base(dados):
    """Limpeza estrutural com regras fixas (não aprende médias, modas ou escalas).

    - confere as colunas obrigatórias;
    - remove duplicatas exatas e rejeita IDs repetidos com dados diferentes;
    - exige alvo 0/1 sem ausentes;
    - valores numéricos impossíveis (negativos, texto, fração em contagens) viram ausentes,
      que serão tratados depois pela imputação do pipeline;
    - categorias: remove espaços extras; texto vazio vira ausente.
    """
    d = dados.copy()
    obrig = {'ID_Cliente', *NUM, *CAT, ALVO}
    if not obrig.issubset(d.columns):
        raise ValueError(f'Colunas faltantes: {sorted(obrig - set(d.columns))}')
    audit = {'linhas_brutas': len(d), 'colunas': list(d.columns),
             'ausentes_brutos': d.isna().sum().to_dict()}

    d = d[['ID_Cliente', *NUM, *CAT, ALVO]].drop_duplicates()
    audit['duplicatas_exatas_removidas'] = len(dados) - len(d)
    if d.ID_Cliente.isna().any() or d.ID_Cliente.astype(str).str.strip().eq('').any():
        raise ValueError('ID ausente.')
    d.ID_Cliente = d.ID_Cliente.astype(str).str.strip()
    if d.ID_Cliente.duplicated().any():
        raise ValueError('ID duplicado com dados conflitantes.')
    if not d[ALVO].isin([0, 1]).all():
        raise ValueError('Churn deve ser 0 ou 1, sem ausentes.')
    if d[ALVO].nunique() != 2:
        raise ValueError('São necessárias as duas classes de Churn.')

    audit['invalidos_dominio'] = {}
    for c in NUM:
        a = pd.to_numeric(d[c], errors='coerce')
        invalido = (d[c].notna() & a.isna()) | (a < 0) | (a.notna() & ~np.isfinite(a))
        if c != 'Valor_Mensal':  # meses, frequência e tickets são contagens inteiras
            invalido |= a.notna() & (a % 1 != 0)
        audit['invalidos_dominio'][c] = int(invalido.sum())
        d[c] = a.mask(invalido)
        if d[c].isna().all():
            raise ValueError(f'Variável totalmente ausente/inválida: {c}')
    for c in CAT:
        d[c] = d[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan).replace('', np.nan)
        if d[c].isna().all():
            raise ValueError(f'Categoria totalmente ausente: {c}')

    audit.update(clientes=len(d), ausentes_apos_regras=d.isna().sum().to_dict(),
                 contratos_acima_36=int((d.Meses_Contrato > 36).sum()),
                 alvo_contagem=d[ALVO].value_counts().sort_index().to_dict(),
                 limitacao_temporal='Sem datas, mês de referência ou horizonte do alvo; avaliação transversal.')
    return d.set_index('ID_Cliente'), audit


def preprocessar():
    """Pipeline de pré-processamento, reajustado dentro de cada treino (sem vazamento).

    Numéricas:   imputação pela MEDIANA -> limitação de outliers (IQR) -> padronização.
    Categóricas: imputação pela MODA -> One-Hot Encoding (sem ordem artificial;
                 categorias novas em produção são ignoradas em vez de gerar erro).
    """
    numericas = Pipeline([('imputar', SimpleImputer(strategy='median')),
                          ('limitar', IQRClipper()),
                          ('escalar', StandardScaler())])
    categoricas = Pipeline([('imputar', SimpleImputer(strategy='most_frequent')),
                            ('encoding', onehot())])
    return ColumnTransformer([('numericas', numericas, NUM), ('categoricas', categoricas, CAT)])


def dividir(b):
    """Divisão estratificada por cliente: 60% treino, 20% validação, 20% teste."""
    treino_val, teste = train_test_split(b.index, test_size=.2, stratify=b[ALVO], random_state=SEED)
    treino, val = train_test_split(treino_val, test_size=.25, stratify=b.loc[treino_val, ALVO],
                                   random_state=SEED)
    return {'treino': treino, 'validacao': val, 'teste': teste}


def explorar(X, y, tr, audit, saida, rel):
    """Análise exploratória (somente no treino) e relatório de outliers."""
    rel.titulo('1. EXPLORAÇÃO E LIMPEZA DOS DADOS')
    rel(f"Linhas lidas: {audit['linhas_brutas']} | duplicatas removidas: {audit['duplicatas_exatas_removidas']}"
        f" | valores fora do domínio: {sum(audit['invalidos_dominio'].values())}"
        f" | ausentes: {sum(audit['ausentes_brutos'].values())}")
    rel('Ausentes: a base não tem nenhum, mas o pipeline imputa mediana (numéricas) e moda (categóricas)')
    rel('para estar preparado para dados novos. Encoding: One-Hot nas 3 variáveis categóricas.')
    rel(f"Clientes com Meses_Contrato > 36 (empresa tem 3 anos): {audit['contratos_acima_36']} "
        '- registros mantidos; o significado da coluna precisa ser confirmado.')
    X.loc[tr].describe(include='all').to_csv(saida / 'estatisticas_treino.csv')
    taxa = y.loc[tr].mean()
    rel(f'Treino: {len(tr)} clientes | taxa de churn {pct(taxa)} (classes quase equilibradas).')

    # Figura 01 - alvo, valor mensal e frequência por classe
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].bar(['Churn=0', 'Churn=1'], y.loc[tr].value_counts().reindex([0, 1]), color=CORES[:2])
    axs[0].set_title('Alvo no treino')
    axs[0].set_ylabel('Clientes')
    axs[1].hist(X.loc[tr, 'Valor_Mensal'], bins=18, color=CORES[0])
    axs[1].set_title('Valor mensal no treino')
    axs[1].set_xlabel('Unidade monetária não especificada')
    for lab, c in [(0, CORES[0]), (1, CORES[1])]:
        axs[2].hist(X.loc[tr][y.loc[tr] == lab]['Frequencia_Acesso_Semanal'], bins=np.arange(-.5, 8.5, 1),
                    alpha=.6, label=f'Churn={lab}', color=c)
    axs[2].set_title('Frequência semanal por classe')
    axs[2].set_xlabel('Valor informado na planilha')
    axs[2].legend()
    salvar_fig(fig, saida, '01_exploracao.png')

    # Figura 02 - tickets por classe e boxplots (análise de outliers)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for lab, c in [(0, CORES[0]), (1, CORES[1])]:
        axs[0].hist(X.loc[tr][y.loc[tr] == lab]['Tickets_Suporte'], bins=np.arange(-.5, 10.5, 1),
                    alpha=.6, label=f'Churn={lab}', color=c)
    axs[0].set_title('Tickets de suporte por classe')
    axs[0].set_xlabel('Contagem (janela não informada)')
    axs[0].legend()
    axs[1].boxplot([X.loc[tr, n].dropna() for n in NUM])
    axs[1].set_xticks([1, 2, 3, 4])
    axs[1].set_xticklabels(['Meses', 'Valor', 'Frequência', 'Tickets'])
    axs[1].set_title('Distribuições numéricas do treino')
    axs[1].set_ylabel('Cada atributo em sua unidade')
    salvar_fig(fig, saida, '02_suporte_outliers.png')

    # Outliers: limites IQR aprendidos no treino e quantos valores seriam limitados
    pre = preprocessar().fit(X.loc[tr])
    imp = pre.named_transformers_['numericas'].named_steps['imputar']
    clip = pre.named_transformers_['numericas'].named_steps['limitar']
    a = imp.transform(X.loc[tr, NUM])
    tratado = clip.transform(a)
    tab_out = pd.DataFrame({'variavel': NUM, 'limite_inferior': clip.lower_, 'limite_superior': clip.upper_,
                            'valores_limitados_treino': (a != tratado).sum(axis=0)})
    tab_out.to_csv(saida / 'outliers_treino.csv', index=False)
    rel('')
    rel('Outliers (regra IQR aprendida no treino):')
    rel.tabela(tab_out.set_index('variavel'), casas=2)
    rel('Nenhum valor ultrapassa os limites: todas as variáveis estão em faixas plausíveis.')

    # Figura 02b - taxa de churn por nível de cada variável numérica (treino)
    Xt, yt = X.loc[tr], y.loc[tr]
    faixas = {
        'Tickets de suporte': Xt['Tickets_Suporte'].astype(int),
        'Acessos por semana': Xt['Frequencia_Acesso_Semanal'].astype(int),
        'Meses de contrato': pd.cut(Xt['Meses_Contrato'], [0, 12, 24, 36, 48],
                                    labels=['1-12', '13-24', '25-36', '37-47']),
        'Quintil de valor mensal': pd.qcut(Xt['Valor_Mensal'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']),
    }
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    rel('')
    rel('Taxa de churn por nível das variáveis numéricas (treino):')
    for ax, (nome, f) in zip(axs, faixas.items()):
        t = yt.groupby(f, observed=True).mean()
        ax.bar(t.index.astype(str), t.values, color=CORES[1])
        ax.axhline(taxa, ls='--', color='gray', lw=1)
        ax.set_title(f'Churn por {nome.lower()}')
        ax.set_ylim(0, 1.05)
        rel(f'  {nome}: ' + ' | '.join(f'{i}: {pct(v, 0)}' for i, v in t.items()))
    axs[0].set_ylabel('Taxa de churn (treino)')
    salvar_fig(fig, saida, '02b_churn_por_variavel.png')

    # Figura 02c - churn por categoria (com teste qui-quadrado) e correlação
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={'width_ratios': [1, 1, 1, 1.25]})
    rel('')
    rel('Taxa de churn por categoria (treino) e teste qui-quadrado de independência:')
    for ax, c in zip(axs[:3], CAT):
        t = yt.groupby(Xt[c]).mean().sort_values(ascending=False)
        p = chi2_contingency(pd.crosstab(Xt[c], yt))[1]
        ax.bar(t.index, t.values, color=CORES[1])
        ax.axhline(taxa, ls='--', color='gray', lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{c.replace('_', ' ')} (p = {num(p, 2)})")
        ax.tick_params(axis='x', labelsize=8, labelrotation=20)
        rel(f'  {c}: ' + ' | '.join(f'{i}: {pct(v, 0)}' for i, v in t.items()) + f'  (p = {num(p, 3)})')
    corr = pd.concat([Xt[NUM], yt], axis=1).corr()
    im = axs[3].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    rotulos = ['Meses', 'Valor', 'Frequência', 'Tickets', 'Churn']
    axs[3].set_xticks(range(5))
    axs[3].set_xticklabels(rotulos, rotation=30, ha='right', fontsize=8)
    axs[3].set_yticks(range(5))
    axs[3].set_yticklabels(rotulos, fontsize=8)
    for i in range(5):
        for j in range(5):
            axs[3].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7.5,
                        color='white' if abs(corr.iloc[i, j]) > .6 else 'black')
    axs[3].set_title('Correlação (treino)')
    fig.colorbar(im, ax=axs[3], shrink=.8)
    salvar_fig(fig, saida, '02c_categorias_correlacao.png')
    rel('')
    rel('Correlação de cada variável numérica com o churn (treino): '
        + ' | '.join(f'{c}: {num(v, 2)}' for c, v in corr[ALVO].drop(ALVO).items()))
    rel('Leitura: tickets de suporte (forte, positiva) e frequência de acesso (moderada, negativa)')
    rel('concentram a relação com o churn; plano, gênero e dispositivo não diferem de forma significativa.')


# ==================================================================================
# 2. SEGMENTAÇÃO DE CLIENTES (aprendizado não supervisionado)
# ==================================================================================
def segmentar(X, y, tr, saida, rel):
    """K-Means nos atributos padronizados + PCA para visualização.

    O Churn NÃO participa do agrupamento. Ele só é consultado DEPOIS, para descrever
    quanto cada grupo cancela (interpretação das personas).
    """
    rel.titulo('2. SEGMENTAÇÃO DE CLIENTES (K-Means + PCA)')
    pre = preprocessar()
    ztr = pre.fit_transform(X.loc[tr])      # pré-processamento aprendido só no treino
    z = pre.transform(X)

    # Escolha do número de grupos: maior silhouette no treino entre k = 2 e 6
    ks, kms = [], {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(ztr)
        ks.append({'k': k, 'silhouette': silhouette_score(ztr, km.labels_), 'inercia': km.inertia_})
        kms[k] = km
    tabk = pd.DataFrame(ks)
    tabk.to_csv(saida / 'selecao_clusters.csv', index=False)
    k = int(tabk.sort_values('silhouette', ascending=False).iloc[0].k)
    km = kms[k]
    clusters = km.predict(z)
    pca = PCA(n_components=2, random_state=SEED).fit(ztr)
    coords = pca.transform(z)
    sil = float(tabk.loc[tabk.k == k, 'silhouette'].iloc[0])
    rel.tabela(tabk.set_index('k'))
    rel(f'k escolhido: {k} (silhouette {num(sil)}). PCA com 2 componentes retem '
        f'{pct(pca.explained_variance_ratio_.sum())} da variância.')

    # Figura 03 - silhouette por k e projeção PCA
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    axs[0].plot(tabk.k, tabk.silhouette, 'o-', color=CORES[0])
    axs[0].set_xlabel('Número de grupos')
    axs[0].set_ylabel('Silhouette do treino')
    axs[0].set_title(f'Escolha exploratória: k={k}')
    for i in range(k):
        mask = clusters == i
        axs[1].scatter(coords[mask, 0], coords[mask, 1], s=13, alpha=.5, label=f'Grupo {i}', color=CORES[i])
    axs[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axs[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axs[1].set_title('PCA: projeção de todos os clientes')
    axs[1].legend(fontsize=8)
    salvar_fig(fig, saida, '03_clusters_pca.png')

    # Perfis (personas): médias e categorias predominantes no treino
    seg = X.copy()
    seg['cluster'] = clusters
    perf = seg.loc[tr].groupby('cluster')[NUM].mean().round(3)
    perf['clientes_treino'] = seg.loc[tr].groupby('cluster').size()
    perf['clientes_total'] = seg.groupby('cluster').size()
    for c in CAT:
        perf[c + '_predominante'] = seg.loc[tr].groupby('cluster')[c].agg(lambda v: v.mode().iloc[0])
        perf[c + '_proporcao'] = seg.loc[tr].groupby('cluster')[c].agg(lambda v: v.value_counts(normalize=True).iloc[0])
    freq_media = X.loc[tr, 'Frequencia_Acesso_Semanal'].mean()
    tickets_mediana = X.loc[tr, 'Tickets_Suporte'].median()
    perf['persona'] = [('Acesso mais frequente' if row.Frequencia_Acesso_Semanal >= freq_media else 'Acesso menos frequente')
                       + ', ' + ('mais tickets' if row.Tickets_Suporte >= tickets_mediana else 'menos tickets')
                       for row in perf.itertuples()]
    perf['acao_proposta'] = [('Investigar experiência de suporte e testar recuperação do atendimento.'
                              if row.Tickets_Suporte >= tickets_mediana else
                              'Testar recomendações de conteúdo e adequação da comunicação ao uso.')
                             for row in perf.itertuples()]
    # Interpretação: taxa de churn de cada grupo, calculada DEPOIS do agrupamento
    perf['taxa_churn_treino'] = y.loc[tr].groupby(seg.loc[tr, 'cluster']).mean()
    perf['taxa_churn_total'] = y.groupby(seg['cluster']).mean()
    perf.to_csv(saida / 'perfis_clusters.csv')
    seg.reset_index().to_csv(saida / 'clientes_segmentados.csv', index=False)

    rel('')
    rel('Personas (médias no treino; churn calculado depois do agrupamento):')
    for i, row in perf.iterrows():
        rel(f'  Grupo {i} - {row.persona}: {int(row.clientes_total)} clientes | '
            f'{num(row.Frequencia_Acesso_Semanal, 2)} acessos/sem | {num(row.Tickets_Suporte, 2)} tickets | '
            f'{num(row.Meses_Contrato, 1)} meses | plano {row.Plano_predominante} | '
            f'gênero {row.Genero_Favorito_predominante} | CHURN {pct(row.taxa_churn_total)}')
        rel(f'      Ação: {row.acao_proposta}')
    rel('Interpretação: os grupos se separam por atrito com o suporte e frequência de uso. O grupo')
    rel('com mais tickets cancela muito mais. Silhouette baixo = fronteiras difusas: personas exploratórias.')
    return {'clusters': clusters, 'perfil': perf, 'k': k, 'silhouette': sil,
            'pca_variancia': float(pca.explained_variance_ratio_.sum())}


# ==================================================================================
# 3. PREDIÇÃO DE CHURN (aprendizado supervisionado)
# ==================================================================================
def metricas(y, p, limiar=.5):
    """Métricas de classificação para um limiar de decisão + custo ilustrativo."""
    pred = (p >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'acuracia': accuracy_score(y, pred), 'precisao': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
            'roc_auc': roc_auc_score(y, p), 'average_precision': average_precision_score(y, p),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'custo_ilustrativo': int(CUSTO_FP * fp + CUSTO_FN * fn), 'limiar': float(limiar)}


def escolher_limiar(y, p):
    """Testa limiares de 0 a 1 (passo 0,01) e 1,01 (ninguém contatado); escolhe o de menor custo.
    Desempate: maior recall, depois maior precisão, depois maior limiar."""
    tabela = pd.DataFrame([metricas(y, p, t) for t in np.r_[np.linspace(0, 1, 101), 1.01]])
    melhor = tabela.sort_values(['custo_ilustrativo', 'recall', 'precisao', 'limiar'],
                                ascending=[True, False, False, False]).iloc[0]
    return float(melhor.limiar), tabela


def candidatos():
    """Classificadores comparados. Hiperparâmetros fixados ANTES de olhar o teste,
    escolhidos para limitar a complexidade (prevenção de overfitting)."""
    return {
        'Dummy': DummyClassifier(strategy='prior'),                     # referência "sem modelo"
        'Regressao Logistica': LogisticRegression(C=1, max_iter=1500, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=8,
                                                max_features='sqrt', random_state=SEED, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=.04, max_depth=2,
                                                        min_samples_leaf=12, subsample=.8, random_state=SEED),
        'SVM (RBF)': SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=SEED),
        'k-NN': KNeighborsClassifier(n_neighbors=25),                   # k ~ raiz de 600 (regra prática)
    }


def comparar_classificadores(X, y, tr, saida, rel):
    """Validação cruzada estratificada (5 folds) no treino; escolha pela Average Precision."""
    rel.titulo('3. PREDIÇÃO DE CHURN - comparação de classificadores (5 folds no treino)')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res, pipes = [], {}
    for nome, clf in candidatos().items():
        pipe = Pipeline([('preparar', preprocessar()), ('modelo', clf)])
        scores = cross_validate(pipe, X.loc[tr], y.loc[tr], cv=cv,
                                scoring={'ap': 'average_precision', 'auc': 'roc_auc', 'recall': 'recall', 'f1': 'f1'},
                                return_train_score=True, n_jobs=1, error_score='raise')
        linha = {'modelo': nome}
        for m in ['ap', 'auc', 'recall', 'f1']:
            linha[f'{m}_media'] = float(scores[f'test_{m}'].mean())
            linha[f'{m}_dp'] = float(scores[f'test_{m}'].std(ddof=1))
        linha['ap_treino'] = float(scores['train_ap'].mean())
        res.append(linha)
        pipe.fit(X.loc[tr], y.loc[tr])
        pipes[nome] = pipe
    cvtab = pd.DataFrame(res)
    cvtab['gap_treino_cv'] = cvtab['ap_treino'] - cvtab['ap_media']   # sinal de overfitting
    cvtab.to_csv(saida / 'comparacao_validacao_cruzada.csv', index=False)
    escolhido = cvtab[cvtab.modelo != 'Dummy'].sort_values('ap_media', ascending=False).iloc[0].modelo
    rel.tabela(cvtab.set_index('modelo')[['ap_treino', 'ap_media', 'ap_dp', 'auc_media', 'f1_media', 'gap_treino_cv']])
    rel(f'Modelo escolhido (maior Average Precision média na validação cruzada): {escolhido}')
    rel('Prevenção de overfitting: pipeline reajustado em cada fold, hiperparâmetros que limitam a')
    rel('complexidade, validação e teste separados, e comparação do desempenho de treino com o da CV.')
    return pipes, cvtab, escolhido


# ==================================================================================
# 4. AVALIAÇÃO DO MODELO
# ==================================================================================
def avaliar(X, y, split, pipes, cvtab, escolhido, saida, rel):
    """Limiar escolhido na validação; teste usado uma única vez, com tudo já congelado."""
    rel.titulo('4. AVALIAÇÃO DO MODELO (teste com 200 clientes nunca vistos)')
    va, te = split['validacao'], split['teste']
    best = pipes[escolhido]
    pval = best.predict_proba(X.loc[va])[:, 1]
    limiar, tablim = escolher_limiar(y.loc[va], pval)
    tablim.to_csv(saida / 'limiares_validacao.csv', index=False)
    rel(f'Limiar de decisão escolhido na validação: {num(limiar, 2)} '
        f'(custo = {CUSTO_FP} x FP + {CUSTO_FN} x FN; ignorar um cancelamento custa 5x mais).')

    testrows, probs = [], {}
    for nome, pipe in pipes.items():
        p = pipe.predict_proba(X.loc[te])[:, 1]
        probs[nome] = p
        testrows.append({'modelo': nome, **metricas(y.loc[te], p)})
    ptest = probs[escolhido]
    final = metricas(y.loc[te], ptest, limiar)
    testrows.append({'modelo': escolhido + ' (limiar ajustado)', **final})
    testtab = pd.DataFrame(testrows)
    testtab.to_csv(saida / 'metricas_teste.csv', index=False)
    pred = (ptest >= limiar).astype(int)
    rel_class = classification_report(y.loc[te], pred, target_names=['Permanece', 'Cancela'], digits=4, zero_division=0)
    (saida / 'relatorio_classificacao.txt').write_text(rel_class, encoding='utf-8')
    pd.DataFrame({'ID_Cliente': te, 'real': y.loc[te].values, 'probabilidade': ptest, 'previsao': pred,
                  'limiar': limiar}).to_csv(saida / 'previsoes_teste.csv', index=False)
    rel.tabela(testtab.set_index('modelo')[['acuracia', 'precisao', 'recall', 'f1', 'roc_auc', 'tp', 'fp', 'fn', 'tn', 'custo_ilustrativo']])
    rel('')
    rel(f'Relatório de classificação - {escolhido}, limiar {num(limiar, 2)}:')
    rel(rel_class)

    # Figura 04 - matriz de confusão, curvas ROC e precisão-recall
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    cm = confusion_matrix(y.loc[te], pred, labels=[0, 1])
    axs[0].imshow(cm, cmap='Greens')
    for i in range(2):
        for j in range(2):
            axs[0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=19,
                        color='white' if cm[i, j] > cm.max() * .55 else '#153B35')
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(['Permanece', 'Cancela'])
    axs[0].set_yticks([0, 1])
    axs[0].set_yticklabels(['Permanece', 'Cancela'])
    axs[0].set_xlabel('Previsto')
    axs[0].set_ylabel('Real')
    axs[0].set_title(f'Matriz de confusão (t = {limiar:.2f})')
    for idx, (nome, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y.loc[te], p)
        axs[1].plot(fpr, tpr, label=f'{nome}: {roc_auc_score(y.loc[te], p):.3f}', color=CORES[idx])
    axs[1].plot([0, 1], [0, 1], '--', color='gray')
    axs[1].legend(fontsize=7)
    axs[1].set_xlabel('Taxa de falsos positivos')
    axs[1].set_ylabel('Recall')
    axs[1].set_title('ROC / AUC no teste')
    prec, rec, _ = precision_recall_curve(y.loc[te], ptest)
    axs[2].plot(rec, prec, color=CORES[0], label=f'AP = {average_precision_score(y.loc[te], ptest):.3f}')
    axs[2].axhline(y.loc[te].mean(), ls='--', color='gray', label='Prevalência')
    axs[2].set_xlabel('Recall')
    axs[2].set_ylabel('Precisão')
    axs[2].legend(fontsize=8)
    axs[2].set_title('Precisão-recall: modelo escolhido')
    salvar_fig(fig, saida, '04_avaliacao.png')

    # Figura 05 - custo por limiar (validação) e comparação dos modelos na CV
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].plot(tablim.limiar, tablim.custo_ilustrativo, color=CORES[0])
    axs[0].axvline(limiar, color=CORES[2], ls='--')
    axs[0].set_xlabel('Limiar')
    axs[0].set_ylabel(f'{CUSTO_FP} × FP + {CUSTO_FN} × FN (unidades)')
    axs[0].set_title('Custo ilustrativo na validação')
    axs[1].barh(cvtab.modelo, cvtab.ap_media, xerr=cvtab.ap_dp, color=CORES[:len(cvtab)])
    axs[1].set_xlabel('Average Precision (média ± desvio-padrão)')
    axs[1].set_title('Comparação em 5 folds do treino')
    salvar_fig(fig, saida, '05_comparacao_limiar.png')

    # Figura 06 - importância por permutação (validação)
    imp = permutation_importance(best, X.loc[va], y.loc[va], scoring='average_precision', n_repeats=10,
                                 random_state=SEED, n_jobs=1)
    itab = pd.DataFrame({'variavel': NUM + CAT, 'queda_ap_media': imp.importances_mean,
                         'dp': imp.importances_std}).sort_values('queda_ap_media', ascending=False)
    itab.to_csv(saida / 'importancia_permutacao_validacao.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    top = itab.head(8).iloc[::-1]
    ax.barh(top.variavel, top.queda_ap_media, xerr=top.dp, color=CORES[0])
    ax.set_xlabel('Queda de Average Precision após permutação')
    ax.set_title('Associação preditiva na validação (não causal)')
    salvar_fig(fig, saida, '06_importancia.png')
    rel('Importância por permutação (queda da AP ao embaralhar a variável): '
        + ' | '.join(f'{r.variavel}: {num(r.queda_ap_media)}' for r in itab.head(3).itertuples()))

    # Intervalo de confiança da AUC por bootstrap (modelo fixo, reamostrando o teste)
    rng = np.random.default_rng(SEED)
    vals = []
    yt = y.loc[te].to_numpy()
    for _ in range(1000):
        ix = rng.integers(0, len(te), len(te))
        if len(np.unique(yt[ix])) == 2:
            vals.append(roc_auc_score(yt[ix], ptest[ix]))
    ci = np.quantile(vals, [.025, .975]).tolist()
    rel(f"AUC no teste: {num(final['roc_auc'])} (IC 95% bootstrap: {num(ci[0])} a {num(ci[1])})")

    # Capacidade limitada: contatar só os 20% de maior risco
    topn = max(1, int(np.ceil(.2 * len(te))))
    idx = np.argsort(-ptest, kind='stable')[:topn]
    top_tp = int(yt[idx].sum())
    return {'limiar': limiar, 'final': final, 'testtab': testtab, 'ptest': ptest, 'pval': pval, 'ci': ci,
            'importancia': itab,
            'top20': {'contatos': topn, 'verdadeiros_positivos': top_tp, 'precisao': float(top_tp / topn),
                      'recall': float(top_tp / yt.sum())}}


# ==================================================================================
# 5. ANÁLISE CRÍTICA
# ==================================================================================
def analise_critica(X, y, split, seg, cvtab, escolhido, aval, saida, rel):
    """Discussão crítica: desempenho e utilidade prática, comparação entre abordagens,
    limitações do pipeline e propostas de melhoria. Os números são calculados aqui."""
    tr, te = split['treino'], split['teste']
    yt = y.loc[te].to_numpy()
    ptest, limiar, final = aval['ptest'], aval['limiar'], aval['final']
    t05 = aval['testtab'].set_index('modelo').loc[escolhido]
    cv = cvtab.set_index('modelo')

    # (a) Segmentação x modelo: com o MESMO número de contatos, quem acha mais cancelamentos?
    perf = seg['perfil']
    grupo_risco = int(perf['taxa_churn_treino'].idxmax())        # decidido com dados de treino
    no_grupo = seg['clusters'][X.index.get_indexer(te)] == grupo_risco
    n = int(no_grupo.sum())
    pelo_modelo = np.zeros(len(te), dtype=bool)
    pelo_modelo[np.argsort(-ptest, kind='stable')[:n]] = True

    def politica(sel):
        tp = int((sel & (yt == 1)).sum())
        fp = int((sel & (yt == 0)).sum())
        fn = int((~sel & (yt == 1)).sum())
        return {'contatos': int(sel.sum()), 'tp': tp, 'fp': fp, 'fn': fn,
                'precisao': tp / max(sel.sum(), 1), 'recall': tp / yt.sum(), 'custo': CUSTO_FP * fp + CUSTO_FN * fn}
    seg_pol, mod_pol = politica(no_grupo), politica(pelo_modelo)

    # (b) Dependência de Tickets_Suporte: mesmo modelo, sem essa variável (CV no treino)
    num_sem = [c for c in NUM if c != 'Tickets_Suporte']
    pre_sem = ColumnTransformer([
        ('numericas', Pipeline([('imputar', SimpleImputer(strategy='median')), ('limitar', IQRClipper()),
                                ('escalar', StandardScaler())]), num_sem),
        ('categoricas', Pipeline([('imputar', SimpleImputer(strategy='most_frequent')), ('encoding', onehot())]), CAT)])
    modelo_sem = Pipeline([('preparar', pre_sem), ('modelo', candidatos()[escolhido])])
    s = cross_validate(modelo_sem, X.loc[tr, num_sem + CAT], y.loc[tr],
                       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
                       scoring={'ap': 'average_precision', 'auc': 'roc_auc'})
    sem_tickets = {'ap_media': float(s['test_ap'].mean()), 'auc_media': float(s['test_auc'].mean())}

    custo_todos = int(CUSTO_FP * (yt == 0).sum())
    custo_ninguem = int(CUSTO_FN * (yt == 1).sum())
    listados = [m for m in ['Random Forest', 'Gradient Boosting', 'SVM (RBF)', 'k-NN'] if m in cv.index]
    melhor_listado = cv.loc[listados, 'ap_media'].idxmax()
    gaps = ', '.join(f'{m} {num(cv.loc[m, "gap_treino_cv"])}' for m in cv.index if m != 'Dummy')
    top = aval['top20']

    maior_gap = cv.drop('Dummy')['gap_treino_cv'].idxmax()
    menor_gap = cv.drop('Dummy')['gap_treino_cv'].idxmin()
    ranking = ' | '.join(f'{m} {num(cv.loc[m, "ap_media"])}' for m in cv.sort_values('ap_media', ascending=False).index)
    texto = (f"""
5.1 DISCUSSÃO CRÍTICA SOBRE DESEMPENHO E UTILIDADE PRÁTICA
- O modelo escolhido ({escolhido}) ordena muito bem quem cancela: AUC de {num(final['roc_auc'])} no
  teste (IC 95%: {num(aval['ci'][0])} a {num(aval['ci'][1])}), contra 0,500 do Dummy, que não usa os atributos.
- Precisão x recall: no limiar 0,50, o modelo teve precisão de {pct(t05['precisao'])} e recall de {pct(t05['recall'])}
  ({int(t05['fn'])} cancelamentos ignorados). Com o limiar de custo ({num(limiar, 2)}), o recall subiu para
  {pct(final['recall'])} ({final['fn']} ignorados) e a precisão caiu para {pct(final['precisao'])} ({final['fp']} descontos
  desnecessários). Como ignorar quem vai sair custa 5 vezes mais, o limiar deve favorecer o recall.
- Utilidade: no teste, contatar todos custa {custo_todos} unidades; não contatar ninguém, {custo_ninguem};
  o modelo no limiar 0,50, {int(t05['custo_ilustrativo'])}; e no limiar ajustado, {final['custo_ilustrativo']}, com
  {final['tp'] + final['fp']} contatos em vez de {len(te)}. Se a equipe só puder abordar 20% da base, os
  {top['contatos']} clientes de maior risco tiveram precisão de {pct(top['precisao'])} (recall de {pct(top['recall'])}).
- Cautela: o teste tem só {len(te)} clientes; os resultados não garantem o mesmo desempenho em
  outro período nem medem o efeito de uma campanha.

5.2 COMPARAÇÃO ENTRE ABORDAGENS E RESULTADOS
- Classificadores (Average Precision média na validação cruzada):
  {ranking}.
  As diferenças entre os modelos são pequenas diante da variação entre folds. Entre os
  algoritmos sugeridos no enunciado, o melhor foi o {melhor_listado}, com {num(cv.loc[melhor_listado, 'ap_media'])};
  a Regressão Logística ficou à frente por pouco e foi mantida por ser mais simples e
  interpretável.
- Overfitting (AP de treino menos AP da validação cruzada):
  {gaps}.
  A maior diferença aparece em {maior_gap}, que memoriza mais o treino; a menor, em {menor_gap}.
- Segmentação x modelo, com o mesmo número de contatos no teste ({n}):
    contatar o grupo de maior churn ("{perf.loc[grupo_risco, 'persona']}"): {seg_pol['tp']}
      cancelamentos captados, precisão {pct(seg_pol['precisao'])}, recall {pct(seg_pol['recall'])}, custo {seg_pol['custo']};
    contatar os {n} clientes de maior probabilidade segundo o modelo: {mod_pol['tp']} captados,
      precisão {pct(mod_pol['precisao'])}, recall {pct(mod_pol['recall'])}, custo {mod_pol['custo']}.
  O modelo escolhe melhor QUEM abordar; a persona ajuda a decidir COMO abordar. As duas
  abordagens são complementares.
- Dependência de uma variável: sem Tickets_Suporte, a AP da validação cruzada cai de
  {num(cv.loc[escolhido, 'ap_media'])} para {num(sem_tickets['ap_media'])} (AUC de {num(sem_tickets['auc_media'])}).
  Quase todo o poder preditivo vem dos tickets.

5.3 LIMITAÇÕES DO PIPELINE
- Sem datas, não há como confirmar que os tickets foram registrados ANTES do cancelamento. Se
  parte deles for aberta para cancelar, o modelo estaria "vendo a resposta" (vazamento) e o
  desempenho real seria bem menor (ver o resultado sem Tickets_Suporte acima).
- Sem histórico mensal nem tempo de tela: a validação é entre clientes, não entre meses, e a
  frequência semanal é só uma aproximação do consumo.
- {int((X['Meses_Contrato'] > 36).sum())} clientes têm mais de 36 meses de contrato numa empresa de 3 anos; o significado
  da coluna precisa ser confirmado.
- Segmentação fraca (silhouette de {num(seg['silhouette'])}, k = {seg['k']}): os grupos são exploratórios, não tipos rígidos.
- Os custos de 20 e 100 são hipóteses didáticas, não custos reais nem retorno financeiro.
- Base pequena (1.000 clientes; teste com {len(te)}): há incerteza amostral nas métricas.

5.4 PROPOSTAS DE MELHORIA FUTURA
- Obter datas de tickets e cancelamentos, o motivo de cada chamado e o tempo de tela real, e
  montar janelas de observação anteriores ao churn.
- Validação temporal: treinar com meses passados e testar nos meses seguintes.
- Calibrar as probabilidades e escolher o limiar com custos reais e a capacidade da equipe.
- Testar a segmentação só com variáveis de comportamento (ou K-Prototypes, para dados mistos)
  e avaliar a estabilidade dos grupos.
- Piloto controlado (teste A/B) para medir o efeito real das ofertas; modelos de uplift.
- Monitorar o desempenho, as mudanças nos dados e as diferenças de erro entre grupos de clientes.
""").strip('\n')
    rel.titulo('5. ANÁLISE CRÍTICA')
    for linha in texto.split('\n'):
        rel(linha)
    (saida / 'analise_critica.txt').write_text(texto + '\n', encoding='utf-8')
    return {'texto': texto, 'segmentacao_vs_modelo': {'grupo_alto_risco': grupo_risco,
                                                      'segmentacao': seg_pol, 'modelo': mod_pol},
            'sem_tickets_cv': sem_tickets}


# ==================================================================================
# Execução completa
# ==================================================================================
def executar(arquivo, saida, abrir=False):
    saida.mkdir(parents=True, exist_ok=True)
    rel = Relatorio()
    rel('GlobalStream - Retenção e segmentação de clientes')
    rel(f'Python {platform.python_version()} | pandas {pd.__version__} | scikit-learn {sklearn.__version__} | '
        f'numpy {np.__version__} | matplotlib {matplotlib.__version__}')

    original = pd.read_excel(arquivo, sheet_name='Sheet1', engine='openpyxl')
    b, audit = preparar_base(original)
    split = dividir(b)
    tr, va, te = split['treino'], split['validacao'], split['teste']
    X = b[NUM + CAT]
    y = b[ALVO].astype(int)
    rel(f'Planilha lida: {len(b)} clientes | partições: treino {len(tr)}, validação {len(va)}, teste {len(te)}')
    json_dump(audit, saida / 'auditoria_dados.json')
    b.reset_index().to_csv(saida / 'base_preparada.csv', index=False)
    pd.concat([pd.DataFrame({'ID_Cliente': v, 'particao': k}) for k, v in split.items()]).to_csv(
        saida / 'particoes.csv', index=False)

    explorar(X, y, tr, audit, saida, rel)                                     # Seção 1
    seg = segmentar(X, y, tr, saida, rel)                                     # Seção 2
    pipes, cvtab, escolhido = comparar_classificadores(X, y, tr, saida, rel)  # Seção 3
    aval = avaliar(X, y, split, pipes, cvtab, escolhido, saida, rel)          # Seção 4
    critica = analise_critica(X, y, split, seg, cvtab, escolhido, aval, saida, rel)  # Seção 5

    final = aval['final']
    resumo = {
        'origem': 'Planilha oficial globalstream_data.xlsx do enunciado; nenhum dado foi gerado ou alterado.',
        'arquivo_entrada': arquivo.name, 'aba': 'Sheet1',
        'sha256_entrada': hashlib.sha256(arquivo.read_bytes()).hexdigest(),
        'semente': SEED, 'clientes': len(b), 'prevalencia': float(y.mean()), 'prevalencia_teste': float(y.loc[te].mean()),
        'particoes': {k: len(v) for k, v in split.items()}, 'modelo_escolhido': escolhido, 'limiar': aval['limiar'],
        'metricas_teste': final, 'auc_ic95_bootstrap': aval['ci'],
        'validacao_limiar_padrao': metricas(y.loc[va], aval['pval'], .5),
        'validacao_limiar_ajustado': metricas(y.loc[va], aval['pval'], aval['limiar']),
        'clusters_k': seg['k'], 'silhouette_treino': seg['silhouette'], 'pca_variancia_total': seg['pca_variancia'],
        'taxa_churn_por_grupo': {int(i): float(v) for i, v in seg['perfil']['taxa_churn_total'].items()},
        'custo_teste_todos_contatados': int(CUSTO_FP * (y.loc[te] == 0).sum()),
        'custo_teste_ninguem_contatado': int(CUSTO_FN * (y.loc[te] == 1).sum()),
        'top20_teste': aval['top20'],
        'segmentacao_vs_modelo': critica['segmentacao_vs_modelo'],
        'sem_tickets_cv': critica['sem_tickets_cv'],
        'limitacoes': ['Sem datas para validação temporal ou identificação do mês do churn.',
                       'Sem definição da janela dos tickets e da disponibilidade dos atributos antes do churn.',
                       'Meses_Contrato pode indicar duração contratual ou tempo de relacionamento; significado não confirmado.',
                       'Ausência de metadados sobre a origem da planilha; não se afirma que são clientes reais.'],
        'ambiente': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
    }
    json_dump(resumo, saida / 'resumo.json')
    gerar_painel(resumo, aval['testtab'], seg['perfil'], critica['texto'], saida)
    rel('')
    rel(f"Concluído. Painel: {saida / 'painel_resultados.html'}")
    rel.salvar(saida / 'saida_execucao.txt')
    if abrir:
        webbrowser.open((saida / 'painel_resultados.html').resolve().as_uri())
    return resumo


def gerar_painel(s, testtab, perf, critica, saida):
    """Painel HTML local com os principais resultados e figuras."""
    m = s['metricas_teste']
    cards = ''.join(f'<article><small>{lab}</small><strong>{v}</strong></article>' for lab, v in [
        ('Clientes', s['clientes']), ('Modelo', s['modelo_escolhido']), ('AUC no teste', f"{m['roc_auc']:.3f}"),
        ('Precisão', f"{m['precisao']:.1%}"), ('Recall', f"{m['recall']:.1%}"), ('Limiar', f"{s['limiar']:.2f}")])
    tabs = testtab[['modelo', 'precisao', 'recall', 'f1', 'roc_auc', 'average_precision', 'custo_ilustrativo']].to_html(
        index=False, border=0, float_format=lambda v: f'{v:.3f}')
    ps = perf[['persona', 'clientes_total', 'Frequencia_Acesso_Semanal', 'Tickets_Suporte', 'Plano_predominante',
               'taxa_churn_total', 'acao_proposta']].to_html(border=0, float_format=lambda v: f'{v:.2f}')
    figs = ''.join(f'<section><img src="{f.name}" alt="{html.escape(f.stem)}"></section>' for f in sorted(saida.glob('*.png')))
    pagina = f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GlobalStream | Resultados</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#f1f6f4;color:#173c34;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:36px 22px}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:24px}}.nota{{background:#e1efe7;border-left:5px solid #127565;padding:16px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:14px;margin:24px 0}}article,section{{background:white;padding:22px;border-radius:12px;margin-bottom:18px}}small{{display:block}}strong{{display:block;font-size:24px;line-height:1.3;margin-top:8px}}img{{max-width:100%;height:auto}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dce8e1}}pre{{white-space:pre-wrap;font:14px/1.55 ui-monospace,Consolas,monospace}}a{{color:#127565}}@media print{{section{{break-inside:avoid}}}}
    </style></head><body><main><p>INTRODUCTION TO MACHINE LEARNING · PARTE PRÁTICA</p><h1>GlobalStream<br>Retenção e segmentação</h1>
    <p class="nota">Análise de <b>globalstream_data.xlsx</b> (aba Sheet1), base oficial do enunciado. A planilha tem uma linha por cliente e não tem datas: a avaliação classifica o Churn registrado e não comprova, sozinha, previsão para o próximo mês.</p>
    <div class="cards">{cards}</div><section><h2>Desenho do experimento</h2><p>Treino: 600; validação: 200; teste: 200. Seis classificadores comparados por Average Precision em 5 folds do treino. Limiar escolhido na validação. Imputação, limites de outliers e encoding aprendidos no treino. ID e Churn excluídos dos atributos.</p>
    <p>Custo ilustrativo: {CUSTO_FP} × FP + {CUSTO_FN} × FN (não representa ROI). No teste, contatar todos custa {s['custo_teste_todos_contatados']} unidades; não contatar ninguém custa {s['custo_teste_ninguem_contatado']}; a política ajustada custa {m['custo_ilustrativo']}.</p></section>
    <section><h2>Comparação no teste</h2><div class="scroll">{tabs}</div><p>Teste consultado após congelar modelo e limiar. AP = Average Precision. A AUC não muda com o limiar.</p></section>
    <section><h2>Personas</h2><div class="scroll">{ps}</div><p>Grupos formados sem usar o Churn; a taxa de churn de cada grupo foi calculada depois. Silhouette: {s['silhouette_treino']:.3f}; variância retida no PCA: {s['pca_variancia_total']:.1%}.</p></section>{figs}
    <section><h2>Análise crítica</h2><pre>{html.escape(critica)}</pre>
    <p><a href="metricas_teste.csv">Métricas CSV</a> · <a href="resumo.json">Resumo</a> · <a href="auditoria_dados.json">Auditoria</a> · <a href="saida_execucao.txt">Saída completa</a></p></section></main></body></html>'''
    (saida / 'painel_resultados.html').write_text(pagina, encoding='utf-8')


def main():
    try:  # evita erro de acentuação em terminais do Windows com codificação antiga
        sys.stdout.reconfigure(errors='replace')
    except Exception:
        pass
    p = argparse.ArgumentParser(description='GlobalStream: segmentação e predição de churn.')
    p.add_argument('--arquivo', type=Path, default=BASE / 'dados' / 'globalstream_data.xlsx',
                   help='planilha de entrada (padrão: dados/globalstream_data.xlsx)')
    p.add_argument('--saida', type=Path, default=BASE / 'resultados', help='pasta de resultados')
    p.add_argument('--abrir', action='store_true', help='abre o painel no navegador ao final')
    a = p.parse_args()
    executar(a.arquivo, a.saida, a.abrir)


if __name__ == '__main__':
    main()
