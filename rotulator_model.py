import numpy as np
import pandas as pd
import scipy.integrate as integrate

from itertools import combinations
from scipy.optimize import fsolve

def _range_delimitation(attrRangeByGroup, polynomials, d):
	inter_points, inter_dict = _intersections(attrRangeByGroup, polynomials)

	"""
	Calcula o erro estimado para a função de cada grupo selecionado
	"""
	erroFaixa = pd.DataFrame(columns=['Cluster', 'Atributo', 'min_faixa', 'max_faixa', 'AUC'])
	erroFaixa = erroFaixa.astype({'min_faixa': 'float64', 'max_faixa': 'float64' ,'AUC': 'float64'})

	for attr, data in attrRangeByGroup.groupby(['Atributo']):
		"""
		attr = 'tqwt_stdValue_dec_9'
		data = attrRangeByGroup[attrRangeByGroup['Atributo'] == 'tqwt_stdValue_dec_9']
		seleciona o conjunto de limite do atributo
		"""
		limites = sorted(set(list(data['minValue'].values) + list(data['maxValue'].values)))
		L = sorted(set(inter_points[attr] + limites))

		"""calcula os ranges iniciais"""
		for i in range(len(L)-1):
			inicio = L[i]
			fim = L[i+1]
			xzinho = _get_groups(inicio, fim, data, d, polynomials[attr], erroFaixa, attr, 'front', False)

		"""calcula as extensões dos ranges"""
		start_points = erroFaixa[(erroFaixa['Atributo'] == attr)]['min_faixa'].values
		end_points = erroFaixa[(erroFaixa['Atributo'] == attr)]['max_faixa'].values
		points = sorted(set(list(start_points) + list(end_points)))

		for point in points:
			"""pega os grupos cujas funções passam pelo ponto"""
			clusters =list(data[(data['minValue'] < point)
			          & (data['maxValue'] > point)]['Cluster'].values)
			if clusters:
				"""pega o início da função mais distante do ponto"""
				inicio_ = data[data['Cluster'].isin(clusters)]['minValue'].min()
				fim_ = point
				I = sorted(set(np.round(np.linspace(inicio_, fim_, 100),2)))
				"""para cada pequeno pedaço, atribui aos clusters"""
				for i in range(len(I), 1, -1):
					if not _get_groups(I[i-2], I[i-1], data, d, polynomials[attr], erroFaixa, attr, 'back', True):
						break

				"""pega o fim da função mais distante do ponto"""
				inicio_ = point
				fim_ = data[data['Cluster'].isin(clusters)]['maxValue'].max()
				I = sorted(set(np.round(np.linspace(inicio_, fim_, 100),2)))

				"""para cada pequeno pedaço, atribui aos clusters"""
				for i in range(len(I)-1):
					if not _get_groups(I[i], I[i+1], data, d, polynomials[attr], erroFaixa, attr, 'front', True):
						break
			else:
				continue

	return erroFaixa

def _intersections( attrRangeByGroup, polynomials):
	intersections_points_by_attr = {}
	intersections_points_and_clusters_by_attr = {}

	for attr, values in attrRangeByGroup.groupby('Atributo'):
		d = {}
		inter = []

		clusters = list(combinations(attrRangeByGroup['Cluster'].unique(), 2))

		for c in clusters:
			x_max = values[(values['Cluster'] == c[0]) | (values['Cluster'] == c[1])].min()['maxValue']
			x_min = values[(values['Cluster'] == c[0]) | (values['Cluster'] == c[1])].max()['minValue']
			if x_min < x_max:
				"""divide o intervalo em 5 partes iguais"""
				xx = np.linspace(x_min, x_max, num=5)

				"""seleciona os polinômios dos grupos c[0] e c[1]"""
				poly1 = polynomials[attr][c[0]]
				poly2 = polynomials[attr][c[1]]

				"""calcula as interseções em cada intervalo"""
				for x0 in xx:
					r = fsolve(lambda x : np.polyval(poly1, x) - np.polyval(poly2, x),x0, full_output=True, factor=10)
					if (r[3] == 'The solution converged.' and r[0][0] >= x_min and r[0][0] <= x_max):
						d[round(r[0][0],2)] = c
						inter.append(round(r[0][0],2))

		intersections_points_by_attr[attr] = list(set(inter))
		intersections_points_and_clusters_by_attr[attr] = d

	return intersections_points_by_attr, intersections_points_and_clusters_by_attr

def _get_groups( a, b, data, d, polynomials, erroFaixa, attr, sentido, extendendo):
	continuar = False

	inicio = a
	fim = b

	"""seleciona os grupos com domínio da função do erro na faixa"""
	clusters = data[(data['minValue'] <= inicio) & (data['maxValue'] >= fim)]['Cluster'].values

	if clusters.shape[0] >= 1:
		"""calcula o erro para cada cluster"""
		errors = []
		for k in clusters:
			errors.append((k, integrate.quad(np.poly1d(polynomials[k]), a, b)[0]))
		errors.sort(key=lambda tup: tup[1], reverse=False)

		"""seleciona os grupos para os quais a faixa será atribuída com base no parâmetro d"""
		eminimo = errors[0][1]
		finalClusters = [i[0] for i in errors if i[1] <= (eminimo + (d * eminimo))]

		"""Verifica se é necessário concatenar"""
		if sentido == 'front':
			for (clt, auc) in [i for i in errors if i[0] in finalClusters]:
				if not erroFaixa[(erroFaixa['max_faixa'] == inicio) & (erroFaixa['Cluster'] == clt)].empty:
					if erroFaixa[(erroFaixa['min_faixa'] >= inicio ) & (erroFaixa['max_faixa'] <= fim ) & (erroFaixa['Cluster'] == clt)].empty:
						erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['max_faixa'] == inicio) & (erroFaixa['Cluster'] == clt)].index, 'AUC'] += auc
						erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['max_faixa'] == inicio) & (erroFaixa['Cluster'] == clt)].index, 'max_faixa'] = fim
						if extendendo:
							continuar = True
				else:
					if not extendendo:
						erroFaixa.loc[erroFaixa.shape[0],:] = [clt, attr, inicio, fim, auc]

		if sentido == 'back':
			for (clt, auc) in [i for i in errors if i[0] in finalClusters]:
				"""se existe um range na tabela que começa no fim do intervalo"""
				if not erroFaixa[(erroFaixa['min_faixa'] == fim) & (erroFaixa['Cluster'] == clt)].empty:
					"""e se o intervalo não está contido em outro range"""
					if erroFaixa[(erroFaixa['min_faixa'] >= inicio ) & (erroFaixa['max_faixa'] <= fim ) & (erroFaixa['Cluster'] == clt)].empty:
						"""extende o range"""
						erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['min_faixa'] == fim) & (erroFaixa['Cluster'] == clt)].index, 'AUC'] += auc
						erroFaixa.loc[erroFaixa[(erroFaixa['Atributo'] == attr) & (erroFaixa['min_faixa'] == fim) & (erroFaixa['Cluster'] == clt)].index, 'min_faixa'] = inicio
						continuar = True
	return continuar
