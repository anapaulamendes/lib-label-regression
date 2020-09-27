import warnings
import os.path
import numpy as np
import pandas as pd

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

def _import_dataset(X, Y):
	"""
	Gera o DataFrame a partir dos atributos(X) do grupo(Y).
	"""
	dataset = pd.DataFrame(data=X)
	dataset['classe'] = Y

	"""
	Refaz o x e y para serem reconhecido pelos métodos do pandas para df
	"""

	y = dataset.loc[:,'classe']
	x = dataset.drop('classe', axis=1)

	"""
	Normaliza os atributos
	"""
	XNormal = pd.DataFrame(x.apply(minmax_scale).values, columns=x.columns)
	XNormal = XNormal.astype('float64')

	"""
	Retorna a base de dados original e os atributos normalizados (XNormal)
	"""
	return dataset, x, y, XNormal

def _call_predictions(X, Y, XNormal, kernel, c, gamma):
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

def _main(x, y, curves_diff, acceptable_error, kernel, c, gamma):

	db, X, Y, XNormal = _import_dataset(x, y)
	yy = _call_predictions(X, Y, XNormal, kernel, c, gamma)
	errorByValue = (yy.groupby(['Atributo', 'Cluster', 'Actual'])['Erro'].agg({'ErroMedio': np.average})
		.reset_index()
		.astype({'Actual': 'float64', 'ErroMedio': 'float64'}))
	attrRangeByGroup = (yy.groupby(['Atributo', 'Cluster'])['Actual'].agg({'minValue': np.min, 'maxValue': np.max})
		.reset_index()
		.astype({'minValue': 'float64', 'maxValue': 'float64'}))
	polynomials = _poly_apro(errorByValue)
	relevanteRanges = _range_delimitation(attrRangeByGroup, polynomials, curves_diff)
	ranged_attr, results, labels, rotulation_process = _label(relevanteRanges, acceptable_error, db)

	return (labels, results)
