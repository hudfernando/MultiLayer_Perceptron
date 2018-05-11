# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:40:26 2018

@author: HUDSON
"""

import numpy as np
from sklearn import datasets

#função que representa o função sigmoid
def sigmoid(soma):
    return 1 /(1 + np.exp(-soma))
#np.exp() retorna o numero de euler(matematica) = 2.71828184590451
#O numero que esta no parenteses e o que vai elevar esse numero
# a = sigmoid(50) o retorno e 1
    
# derivando a funcao sigmoid
def sigmoidDerivada(sig):
    return sig * (1 - sig)

base = datasets.load_breast_cancer()
entradas = base.data 
valoresSaida = base.target
saidas   = np.empty([569,1], dtype=int)

for i in range(569):
    saidas[i] = valoresSaida[i]
 
pesos0 =  2 * np.random.random((30,3)) - 1
pesos1 =  2 * np.random.random((3,1)) - 1

epocas = 10
taxaAprendizagem = 0.3
momento = 1




for epoca in range(epocas):
#implementando o somatorio da primeira camada para achar os valores de ativação de segunda camanda(camada oculta)
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
#implementando o somatorio da camada oculta para achar os valores da camada de saida  
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    erroCamadaSaida = saidas - camadaSaida #calcula do erro: erro = respostaCorreta - respostaCalculada
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida)) #soma os erros e divide pela quantidade de variaveis (XOR)
    print("Erro: " + str(mediaAbsoluta) + " Epoca:"+str(epoca))
#Calculando o delta para a camada de saida:  deltaSaida = Erro * derivadaSigmoide
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida

#Calculando o delta para a camada oculta: DeltaEscondida = derivadaSigmoide * peso * DeltaSaida
    #primeiro calcular a matriz transposta
    pesos1Transposta = pesos1.T
    deltaSaidaVezesPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaVezesPeso * sigmoidDerivada(camadaOculta)

#BackPropagation = peso_n+1 = (peso_n * momento)+(entrada*delta*taxaDeAprendizagem)
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
#reajustando os pesos de acordo com a formula abaixo
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
#Atualização dos pesos da camada de entrada para a camada oculta   
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    