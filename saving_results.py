import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _save_table(dataset_name, table, file_name):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Teste/' + dataset_name + '/')

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	tmp = table.select_dtypes(include=[np.number])
	table.loc[:, tmp.columns] = np.round(tmp, 2)
	table.to_csv(results_dir + file_name, index=False)

def _save_fig(dataset_name, figName):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Teste/' + dataset_name + '/')

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)
	plt.savefig(results_dir + figName)

def _save_json(dataset_name, dict_, file_name):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Teste/' + dataset_name + '/')

	if not os.path.isdir(results_dir):
		os.makedirs(results_dir)

	json.dump(dict_, open(results_dir+file_name, 'w', encoding="utf8"))

def _get_json(dataset_name, file_name):
	script_dir = os.path.dirname(__file__)
	results_dir = os.path.join(script_dir, 'Teste/' + dataset_name + '/' + file_name)

	return json.load(open(results_dir))
