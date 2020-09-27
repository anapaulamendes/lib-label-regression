from main import _main


"""
Exemplo:

>>> from label_regression import LabelRegression
>>> from sklearn.datasets import load_iris
>>> iris = load_iris()
>>> lr = LabelRegression(iris.feature_names, iris.data, iris.target, 0.1, 0.2, 'linear', 100, 'auto')
>>> lr.getLabels()
   Cluster           Atributo  min_faixa  max_faixa
0      0.0  petal length (cm)        1.0        1.9
0      1.0  petal length (cm)        3.0        5.1
1      1.0   petal width (cm)        1.0        1.8
0      2.0   petal width (cm)        1.8        2.5
>>> lr.getAccuracy()
   Cluster   AR
0      0.0  1.0
1      1.0  1.0
2      2.0  0.9
>>> lr.getPrecision()
    Cluster           Atributo  min_faixa  max_faixa  Precision
0       0.0  petal length (cm)       1.00       1.90       1.00
3       0.0   petal width (cm)       0.10       0.60       1.00
12      0.0   sepal width (cm)       2.97       4.40       0.96
6       0.0  sepal length (cm)       4.30       5.54       0.94
1       1.0  petal length (cm)       3.00       5.10       1.00
4       1.0   petal width (cm)       1.00       1.80       1.00
7       1.0  sepal length (cm)       5.44       6.16       0.56
11      1.0   sepal width (cm)       2.61       3.04       0.52
9       1.0   sepal width (cm)       2.00       2.20       0.06
5       2.0   petal width (cm)       1.80       2.50       0.90
2       2.0  petal length (cm)       4.96       6.90       0.88
8       2.0  sepal length (cm)       5.98       7.90       0.86
10      2.0   sepal width (cm)       2.20       3.02       0.66
"""
class LabelRegression:

    def __init__(self, attr_names, X, y, d, t, kernel, c, gamma):
        self.labels, self.accuracy, self.precision = _main(attr_names, X, y, d, t, kernel, c, gamma)

    def getLabels(self):
        return self.labels

    def getAccuracy(self):
        return self.accuracy

    def getPrecision(self):
        return self.precision
