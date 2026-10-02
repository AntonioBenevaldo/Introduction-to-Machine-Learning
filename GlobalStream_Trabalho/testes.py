"""Testes das invariantes da análise; a planilha original nunca é alterada."""
import hashlib,json,unittest
import numpy as np
import pandas as pd
from globalstream import BASE,NUM,CAT,preparar_base,dividir,preprocessar,IQRClipper,metricas,escolher_limiar
class TestesGlobalStream(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.original=pd.read_excel(BASE/'dados/globalstream_data.xlsx')
  cls.b,cls.audit=preparar_base(cls.original);cls.part=dividir(cls.b)
 def test_dimensoes_e_auditoria(self):
  self.assertEqual(self.original.shape,(1000,9));self.assertEqual(len(self.b),1000)
  self.assertEqual(self.audit['duplicatas_exatas_removidas'],0)
  self.assertEqual(sum(self.audit['ausentes_brutos'].values()),0)
  self.assertEqual(int(self.b.Churn.sum()),542)
 def test_ids_nao_compartilhados(self):
  tr,va,te=map(set,self.part.values())
  self.assertFalse(tr & va or va & te or tr & te)
  self.assertEqual([len(tr),len(va),len(te)],[600,200,200]);self.assertEqual(len(tr|va|te),1000)
 def test_atributos_sem_id_ou_alvo(self):
  self.assertNotIn('Churn',NUM+CAT);self.assertNotIn('ID_Cliente',NUM+CAT)
  self.assertEqual(len(NUM+CAT),7)
 def test_rejeita_id_conflitante(self):
  d=self.original.copy();d.loc[0,'ID_Cliente']=d.loc[1,'ID_Cliente']
  with self.assertRaisesRegex(ValueError,'duplicado'):preparar_base(d)
 def test_rejeita_alvo_ausente(self):
  d=self.original.copy();d.loc[0,'Churn']=np.nan
  with self.assertRaisesRegex(ValueError,'Churn'):preparar_base(d)
 def test_rejeita_coluna_faltante(self):
  with self.assertRaisesRegex(ValueError,'faltantes'):preparar_base(self.original.drop(columns=['Plano']))
 def test_valor_invalido_vira_ausente(self):
  d=self.original.copy();d.loc[0,'Valor_Mensal']=-100
  b,a=preparar_base(d);self.assertTrue(pd.isna(b.iloc[0].Valor_Mensal));self.assertEqual(a['invalidos_dominio']['Valor_Mensal'],1)
 def test_imputacao_prevista_e_categoria_nova(self):
  tr=self.b.loc[self.part['treino'],NUM+CAT].copy()
  tr.iloc[0,tr.columns.get_loc('Valor_Mensal')]=np.nan
  tr.iloc[1,tr.columns.get_loc('Plano')]=np.nan
  pre=preprocessar().fit(tr)
  stats=pre.named_transformers_['numericas'].named_steps['imputar'].statistics_.copy()
  np.testing.assert_allclose(stats,tr[NUM].median())
  novo=tr.iloc[:2].copy();novo['Plano']='INEDITO';novo['Valor_Mensal']=1e9
  self.assertTrue(np.isfinite(pre.transform(novo)).all())
  np.testing.assert_array_equal(stats,pre.named_transformers_['numericas'].named_steps['imputar'].statistics_)
  self.assertNotIn('INEDITO',pre.named_transformers_['categoricas'].named_steps['encoding'].categories_[0])
 def test_clipping_aprende_apenas_fit(self):
  c=IQRClipper().fit(np.arange(100).reshape(-1,1));lim=c.upper_.copy()
  self.assertLess(c.transform([[1e9]])[0,0],150);np.testing.assert_array_equal(lim,c.upper_)
 def test_metricas_em_exemplo_conhecido(self):
  m=metricas(np.array([0,0,1,1]),np.array([.1,.8,.2,.9]))
  self.assertEqual([m[k] for k in ['tn','fp','fn','tp']],[1,1,1,1]);self.assertEqual(m['custo_ilustrativo'],120)
 def test_limiar_na_tabela_fornecida(self):
  t,tab=escolher_limiar(np.array([0,0,1,1]),np.array([.1,.8,.2,.9]))
  self.assertEqual(tab.loc[tab.limiar==t,'custo_ilustrativo'].iloc[0],tab.custo_ilustrativo.min())
 def test_evidencias_e_hash_da_planilha(self):
  s=json.loads((BASE/'resultados/resumo.json').read_text(encoding='utf-8'));d=pd.read_csv(BASE/'resultados/previsoes_teste.csv')
  self.assertEqual(s['sha256_entrada'],hashlib.sha256((BASE/'dados/globalstream_data.xlsx').read_bytes()).hexdigest())
  m=metricas(d.real,d.probabilidade,s['limiar'])
  for k,v in m.items():self.assertAlmostEqual(v,s['metricas_teste'][k])
  self.assertEqual(set(d.ID_Cliente),set(self.part['teste']))
if __name__=='__main__':unittest.main(verbosity=2)
