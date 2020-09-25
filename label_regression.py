from indices_test import _main

"""
Exemplo:

import label_regression

label_regression.call("./databases/iris.csv", 0.1, 0.2, 'linear', 100, 'auto')
"""

def call(dataset, d, t, kernel, c, gamma):
    output = _main(dataset, d, t, kernel, c, gamma)
    dataset_name = dataset.split("/")[2].split(".")[0].capitalize()
    print("\n------------ Rótulos dos grupos para base de dados {} ------------\n\n{}".format(dataset_name, output))
