import numpy as np
import pandas as pd

def _label(rangeAUC, V, db):
	"""
	Calcula acurácia dos intervalos
	accuratedRange: {Cluster, Atributo, min_faixa, min_faixa, Accuracy}
	"""
	accuratedRange = (rangeAUC
		.assign(Precision = rangeAUC.apply(lambda x: _calc_accuracy_range(info = x, data = db, classe = x.Cluster), axis=1))
		.sort_values(by=['Cluster', 'Precision'], ascending=[True, False])
		.drop(['AUC'], axis=1))

	labels = (pd.DataFrame(columns=accuratedRange.columns)
		.astype({'min_faixa': 'float64', 'min_faixa': 'float64', 'Precision': 'float64'}))

	results = (pd.DataFrame(columns=['Cluster', 'Accuracy'])
		.astype({'Accuracy': 'float64'}))

	rotulation_process = pd.DataFrame(columns=['Cluster', 'iteracao', 'acuracias'])

	for i in db['classe'].unique():
		"""
		Seleciona todos os pares atributo intervalo candidatos ao rótulo do grupo
		"""
		rotulo_ = accuratedRange[(accuratedRange['Cluster']==i)]
		rc = pd.DataFrame(columns=rotulo_.columns)
		iteracao = 0
		"""
		Adiciona atributos ao rótulo enquanto o acerto em outros grupos for maior que V
		"""
		repit = True
		while repit:
			"""
			seleciona os elementos de maior acurácia
			"""
			pro_attr = rotulo_[(rotulo_['Precision'] == rotulo_.max()['Precision'])]
			rotulo_.drop(pro_attr.index, axis=0, inplace=True)
			while not pro_attr.empty:
				"""
				calcula o acerto em todos os grupos para os elementos de maior acuracia
				"""
				acc = pro_attr.apply(lambda x: _hit_label(rc.append(x, ignore_index=True), db, i), axis=1)
				min_ = min([acc[i][1] for i in acc.keys()])
				add = [i for i in acc.keys() if acc[i][1]==min_][0]

				rc = rc.append(pro_attr.loc[add], ignore_index=True)
				pro_attr.drop(add, inplace=True)
				rotulation_process.loc[rotulation_process.shape[0],['Cluster', 'iteracao']] = [i,iteracao]
				rotulation_process.loc[rotulation_process.shape[0]-1,['acuracias']] = [acc[add][0]]
				iteracao += 1

				"""
				verifica a restrição
				"""
				if min_ <= V or rotulo_.empty:
					repit = False
					labels = pd.concat([labels, rc], sort=False)
					results.loc[results.shape[0],:] = [i, acc[add][2]]
					break
	return accuratedRange, results, labels, rotulation_process

def _calc_accuracy_range(info, data, classe):
	acertos = data[(data['classe'] == classe) & (data[info['Atributo']] >= info['min_faixa']) &  (data[info['Atributo']] <= info['max_faixa'])].shape[0]
	return round(acertos / data[(data['classe'] == classe)].shape[0],2)

def _hit_label(rotulo, data, i):
	accuracy = {}
	for clt in data['classe'].unique():
		data_ = data[(data['classe'] == clt)]
		idx = rotulo.groupby(['Atributo']).apply(lambda x:
			list(data_[(data_[x.Atributo.values[0]] >= x.min_faixa.values[0]) & (data_[x.Atributo.values[0]] <= x.max_faixa.values[0])].index))
		intersec = list(set.intersection(*map(set, [idx[i] for i in list(idx.keys())])))
		accuracy[clt] = np.round(len(intersec) / data_.shape[0],2)
	cluster_acc = accuracy[i]
	max_other_acc = max([accuracy[x] for x in list(accuracy) if x != i])
	return (accuracy, max_other_acc, cluster_acc)
