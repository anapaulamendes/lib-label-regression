import gc
import warnings
import os.path
import numpy as np
import pandas as pd
import saving_results as save
import ploting_functions as pltFunc

from sklearn.preprocessing import minmax_scale
from regression_model import _TrainingModels
from rotulator_model import _range_delimitation
from rotulate import _label

warnings.filterwarnings("ignore")

def _poly_apro(results):
	polynomials = {}
	for attr, values in results.groupby(['Atributo']):
		d = {}
		for clt, data in values.groupby(['Cluster']):
			if data.shape[0] > 1:
				d[clt] = list(np.polyfit(data['Actual'].to_numpy().astype(float), data['ErroMedio'].to_numpy().astype(float), 2))
				polynomials[attr] = d

	return polynomials

def _import_dataset(path):
	"""
	Carrega a base de dados e separa os atributos(X) do grupo(Y).
	"""
	dataset = pd.read_csv(path, sep=',', parse_dates=True)
	Y = dataset.loc[:,'classe']
	X = dataset.drop('classe', axis=1)

	"""
	Normaliza os atributos
	"""
	XNormal = pd.DataFrame(X.apply(minmax_scale).values, columns=X.columns)
	XNormal = XNormal.astype('float64')

	"""
	Retorna a base de dados original, os atributos(X), os grupos(Y),
	os atributos normalizados (XNormal) e a lista de atributos (attr_names)
	"""
	return dataset, X, Y, XNormal

def _call_predictions(db, X, Y, XNormal, kernel, c, gamma):
	"""
	Constrói os modelos de regressão e retorna um dataframe com as predições
	predisctions: {'index', 'Atributo', 'predict'}
	"""
	models = _TrainingModels(XNormal, 10, kernel, c, gamma)
	predictions = models.predictions

	"""
	Estrutura de dados para armazenar o erro das predições
	"""
	yy = pd.DataFrame(columns=['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
	for attr in XNormal.columns:
		"""
		Seleciona as predições para o atributo attr
		"""
		y_ = pd.DataFrame(columns=['Atributo', 'Actual', 'Normalizado', 'Predicted', 'Cluster', 'Erro'])
		y_['Actual'] = X[attr].values
		y_['Normalizado'] = XNormal[attr].values
		y_['Predicted'] = predictions[(predictions['Atributo']==attr)].sort_values(by='index')['predict'].values
		y_['Cluster'] = Y.values
		y_ = y_.assign(Erro=lambda x: abs(x.Normalizado-x.Predicted))
		y_ = y_.assign(Atributo=attr)

		yy = pd.concat([yy, y_])

	""" models._erros, models._metrics """
	return yy

def _save_info_dataset(title, yy, errorByValue, polynomials, attrRangeByGroup):
	save._save_table(title, errorByValue, 'errorByValue.csv')
	save._save_table(title, attrRangeByGroup, 'attrRangeByGroup.csv')
	save._save_json(title, polynomials, 'polynomials')

def _save_info_label(title, ranged_attr, relevanteRanges, results, labels, rotulation_process, errorByValue, polynomials):
	save._save_table(title, ranged_attr, 'atributos_ordenados_por_acerto.csv')
	save._save_table(title, results, 'acuracia.csv')
	save._save_table(title, labels, 'rotulos.csv')
	save._save_table(title, rotulation_process, 'rotulos_por_iteracao.csv')

	pltFunc._plot_auc(title, errorByValue, polynomials, relevanteRanges)
	### BUG ###
	#pltFunc._render_results_table(title, results, header_columns=0, col_width=2.0)
	#pltFunc._render_labels_table(title, labels, header_columns=0, col_width=2.0)

def _main(dataset_path, curves_diff, acceptable_error, kernel, c, gamma):

	datasets = [dataset_path]

	for dataset in datasets:
		title = dataset.split('/')[2].split('.')[0]
		db, X, Y, XNormal = _import_dataset(dataset)
		is_missing_info = False

		if not os.path.isfile('Teste/' + title + '/predictions.csv'):
			yy = _call_predictions(db, X, Y, XNormal, kernel, c, gamma)
			save._save_table(title, yy, 'predictions.csv')
			is_missing_info = True
		else:
			yy = pd.read_csv('Teste/' + title + '/predictions.csv')

		if not os.path.isfile('Teste/' + title + '/errorByValue.csv'):
			errorByValue = (yy.groupby(['Atributo', 'Cluster', 'Actual'])['Erro'].agg({'ErroMedio': np.average})
				.reset_index()
				.astype({'Actual': 'float64', 'ErroMedio': 'float64'}))
			is_missing_info = True
		else:
			errorByValue = pd.read_csv('Teste/' + title + '/errorByValue.csv').astype({'Actual': 'float64', 'ErroMedio': 'float64'})

		if not os.path.isfile('Teste/' + title + '/attrRangeByGroup.csv'):
			attrRangeByGroup = (yy.groupby(['Atributo', 'Cluster'])['Actual'].agg({'minValue': np.min, 'maxValue': np.max})
				.reset_index()
				.astype({'minValue': 'float64', 'maxValue': 'float64'}))
			is_missing_info = True
		else:
			attrRangeByGroup = pd.read_csv('Teste/' + title + '/attrRangeByGroup.csv').astype({'minValue': 'float64', 'maxValue': 'float64'})

		if not os.path.isfile('Teste/' + title + '/polynomials'):
			polynomials = _poly_apro(errorByValue)
			is_missing_info = True
		else:
			polynomials = save._get_json(title, 'polynomials')

		if is_missing_info: _save_info_dataset(title, yy, errorByValue, polynomials, attrRangeByGroup)
		out = pd.DataFrame(columns =['d', 'accuracys', 'n_elemForLabel'])

		if not os.path.isfile('Teste/'+title+str(curves_diff)+'/range.csv'):
			relevanteRanges = _range_delimitation(attrRangeByGroup, polynomials, curves_diff)
			save._save_table(title, relevanteRanges, 'range.csv')
		else:
			relevanteRanges = pd.read_csv('Teste/'+title+str(curves_diff)+'/range.csv')

		ranged_attr, results, labels, rotulation_process = _label(relevanteRanges, acceptable_error, db)

		output_return = labels

		_save_info_label(title + str(curves_diff), ranged_attr, relevanteRanges, results, labels, rotulation_process, errorByValue, polynomials)

		out = out.append(pd.Series({'d': np.round(curves_diff, 2),
		 'n_elemForLabel': labels.groupby(['Cluster', 'Atributo']).size().values,
		 'accuracys': results['Accuracy'].values}), ignore_index=True)

		del relevanteRanges, ranged_attr, results, labels, rotulation_process
		gc.collect()

		out.to_csv('./Teste/results_'+ title + '.csv', index=False)
		return output_return
