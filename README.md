# Lib Label Regression

Criação de uma biblioteca para o trabalho "Rotulação Automática de Grupos Baseada em Análise de Erro de Regressão" do mestrado em Ciência da Computação de Lúcia Emília Soares Silva, do Programa de Pós-Graduação em Ciência da Computação da Universidade Federal do Piauí.


## Proposta da biblioteca:

O objetivo é transformar o código feito no trabalho "Rotulação Automática de Grupos Baseada em Análise de Erro de Regressão" em uma biblioteca Python. Desta maneira, será mais fácil de ser utilizada em outros trabalhos com parâmetros ajustáveis.


## Uso:

```
import label_regression

label_regression.call(dataset, d, t, kernel, c, gamma)
```

*Exemplo Prático*

```
>>> import label_regression
>>> label_regression.call("./databases/iris.csv", 0.1, 0.2, 'linear', 100, 'auto')

------------ Rótulos dos grupos para base de dados Iris ------------

   Cluster      Atributo  min_faixa  max_faixa  Precision
0      1.0  petal_length        1.0        1.9       1.00
0      2.0  petal_length        3.0        5.1       1.00
1      2.0   petal_width        1.0        1.8       1.00
0      3.0  petal_length        4.9        6.9       0.94
```

Descrição dos parâmetros:

 - dataset (string): Caminho para a base de dados que será utilizada. A base deve estar classificada. A classe de rótulos deve possuir o nome da coluna: classe
 - d (float): A diferença máxima entre as curvas. A faixa de valores é de 0 a 1.
 - t (float): O erro aceitável. A faixa de valores é de 0 a 1.
 - kernel (string): Especifica o tipo de kernel a ser usado no algoritmo de regressão. Deve ser 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ou chamável. Se nenhum for fornecido, 'rbf' será usado. Se um chamável for fornecido, ele será usado para pré-calcular a matriz do kernel. Este parâmetro é primário da biblioteca scikit-learn. Para mais informações, acesse: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
 - c (float): O valor default é 1.0. Parâmetro de regularização. A intensidade da regularização é inversamente proporcional a C. Deve ser estritamente positiva. A penalidade é uma penalidade de 12 ao quadrado. Este parâmetro é primário da biblioteca scikit-learn. Para mais informações, acesse: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
 - gamma (string ou float): As opções podem ser {'scale', 'auto'} ou um valor float. Coeficiente de kernel para 'rbf', 'poly' e 'sigmoid'. Se o gamma='scale' (default) é passado, então é usado 1/(n_features * X.var()) como valor de gamma. Se 'auto' é passado, usa 1/n_features. Este parâmetro é primário da biblioteca scikit-learn. Para mais informações, acesse: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html


## Repositório do código original do trabalho:

[luciaemiliaeu/LabelRegression](https://github.com/luciaemiliaeu/LabelRegression)


## Resumo extraído da qualificação de mestrado:

Os Modelos de Rotulação de Grupos propõem a aplicação de técnicas de Inteligência Artificial para extração das principais características dos grupos, a fim de fornecer uma ferramenta para interpretação de agrupamentos oriundos dos mais diversos tipos de algoritmos de clustering. Para isso, diferentes técnincas, como Aprendizagem de Máquina, Lógica Fuzzy e discretização de dados são utilizadas na identificação dos atributos mais relevantes para formação dos grupos e dos intervalos de valores associados a eles. Esse trabalho apresenta um modelo de rotulação de grupos baseado no uso de regressão para delimitação de intervalos de valores dos atributos que revelem os pares atributo-intervalo que melhor resumem os grupos. A relevância de um atributo para o agrupamento é determinada pelos intervalos de valores dos atributos em que o erro de predição da regressão é mínimo, resultando em rótulos específicos e capazes de representar a maioria dos elementos dos grupos. Os resultados obtidos nos experimentos mostram que o modelo é eficaz em rotular os grupos, apresentando Taxas de Concordâncias entre 0,90 e 1,0 para as bases de dados utilizadas, além de garantir rótulos exclusivos para cada grupo por meio da análise da Taxa de Concordância dos rótulos em grupos distintos.


## Abstract:

Cluster Labeling Models apply Artificial Intelligence techniques to extract the main characteristics of data partitioned into clusters, in order to provide a tool for the interpretation of the clustering. For this, different techniques, such as Machine Learning, Fuzzy Logic, and Data Discretization, are used to identify the most important attributes for forming clusters and the ranges of values associated with them. This paper presents a Cluster Labeling Model based on the use of regression to delimit the ranges of values of attributes that reveal the attribute–range pairs that best summarize the groups. In the proposed model, the importance of an attribute to the clustering is determined by the ranges of the values of the attributes with which the prediction error of the regression is minimal,
resulting in specific labels and capable of representing the majority of the elements of the groups. The results obtained in the experiments show that the model is effective, providing labels that represent between 90% and 100% of the elements of the clusters for the databases used, in addition to guaranteeing exclusive labels for each cluster.


## Requisitos:

 - Python3
 - Pandas, versão 0.25.3
 - Scikit-learn
 - Matplotlib

## Instalação:

Para instalar os pacotes necessários, execute:

```
pip install -r requirements.txt
```

## Referências:

SILVA, Lúcia Emília Soares. Rotulação Automática de Grupos Baseada em Análise de Erro de Regressão. Qualificação de Mestrado, Programa de Pós Graduação em Ciência da Computação, UFPI, 2020.
