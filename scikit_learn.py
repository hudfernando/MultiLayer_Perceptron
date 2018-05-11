# -*- coding: utf-8 -*-
"""
Created on Thu May 10 17:05:57 2018

@author: HUDSON
"""

from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entrada = iris.data
saidas = iris.target

redeNeural = MLPClassifier(verbose=True, 
                           max_iter=1000, 
                           tol=0.00001,
                           activation='logistic',
                           learning_rate_init=0.001,
                           )#mostra o erro
redeNeural.fit(entrada, saidas)

redeNeural.predict([[5, 7.2, 5.1, 2.2]])
