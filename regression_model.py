import gc
import statistics
import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

class _TrainingModels:

	def __init__(self, normalBD, folds, kernel, c, gamma):
		"""
		predisctions: {'index', 'Atributo', 'predict'}
		"""
		self.predictions, erros = self._training(normalBD, folds, kernel, c, gamma)

	def _training(self, normalBD, folds, kernel, c, gamma):
		"""
		Cria DataFrames de treino e teste da bd normalizada
		y: atributo de saída
		"""
		predict = pd.DataFrame(columns=['index', 'Atributo', 'predict'])
		predict = predict.astype({'predict': 'float64'})

		erro_metrics = []
		n_attr = 1
		for attr in normalBD.columns:
			"""
			model = SVR(kernel='linear', C=100, gamma='auto')
			"""
			model = SVR(kernel=kernel, C=c, gamma=gamma)

			Y = normalBD[attr]
			X = normalBD.drop(attr, axis=1)

			model.fit(X, Y)
			attr_predicted = model.predict(X)

			for i in X.index:
				predict.loc[predict.shape[0],:] = [i, attr, attr_predicted[i]]

			r2 = model.score(X, Y)
			erro = mean_squared_error(Y, attr_predicted)
			erro_metrics.append((attr, erro, r2))

			del model
			gc.collect()
			n_attr += 1
		return predict , erro_metrics
