# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:31:43 2018

@author: HUDSON
EXEMPLO DO XOR
"""
import numpy as np
from pybrain.tools.shortcuts import buildNetwork #atalho para construir a rede
from pybrain.datasets import SupervisedDataSet # 
from pybrain.supervised.trainers import BackpropTrainer # para fazer o backpropagation
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import SigmoidLayer

#criando a rede neural
"""
Parametro 1: quantidade de neuronios na camada 1 
Parametro 2: quantidade de neuronios na camada oculta  
Parametro 3: quantidade de neuronios na camada de saida 

rede = buildNetwork(2, 3, 1, outclass = SoftmaxLayer, hiddenclass= SigmoidLayer, bias = False)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])
"""
rede = buildNetwork(2, 3, 1,)
base = SupervisedDataSet(2,1) # base de dados: tem dois att previsores e uma classe
#adiciona os elementos e o resultado
base.addSample((0,0), (0, ))
base.addSample((0,1), (1, ))
base.addSample((1,0), (1, ))
base.addSample((1,1), (0, ))
#print(base['input'])
treinamento = BackpropTrainer(rede, dataset = base, learningrate = 0.01, momentum = 0.06)
for i in range(1, 30000):
    erro = treinamento.train()
    if(i % 1000 == 0):  
        print("Erro: %s" % erro)
    
print(np.round(rede.activate([0,0]), 3))
print(rede.activate([1,0]))
print(rede.activate([0,1]))
print(np.round(rede.activate([1,1]), 3))