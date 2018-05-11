# -*- coding: utf-8 -*-
"""
Created on Thu May 10 16:17:20 2018

@author: HUDSON
"""
"""
Constroi a estrutura de uma rede neural artificial
"""
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection

rede = FeedForwardNetwork()

#cria a camada de entrada
camadaEntrada = LinearLayer(2)

#cria a camada oculta
camadaOculta = SigmoidLayer(3)

#cria camada saida
camadaSaida = SigmoidLayer(1)

bias1 = BiasUnit()
bias2 = BiasUnit()

#adiciona as camadas dentro da rede
rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias1)
rede.addModule(bias2)

#ligação da camada de entrada e camada oculta
entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida   = FullConnection(camadaOculta, camadaSaida)
biasOculta    = FullConnection(bias1, camadaOculta)
biasSaida    = FullConnection(bias2, camadaSaida)

rede.sortModules()

#print(rede)
#print(entradaOculta.params)
#print(ocultaSaida.params)
#print(biasOculta.params)
#print(biasSaida.params)
