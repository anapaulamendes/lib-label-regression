from main import _main


"""
Exemplo:

>>> from label_regression import LabelRegression
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> lr = LabelRegression(iris.data, iris.target, 0.1, 0.2, 'linear', 100, 'auto')
>>> lr.getLabels()
   Cluster  Atributo  min_faixa  max_faixa  Precision
0      0.0       2.0        1.0        1.9        1.0
0      1.0       2.0        3.0        5.1        1.0
1      1.0       3.0        1.0        1.8        1.0
0      2.0       3.0        1.8        2.5        0.9
>>> lr.getAccuracy()
   Cluster   AR
0      0.0  1.0
1      1.0  1.0
2      2.0  0.9
"""
class LabelRegression:

    def __init__(self, X, y, d, t, kernel, c, gamma):
        self.labels, self.accuracy = _main(X, y, d, t, kernel, c, gamma)

    def getLabels(self):
        return self.labels

    def getAccuracy(self):
        return self.accuracy
