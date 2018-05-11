# -*- coding: utf-8 -*-
"""
Created on Thu May 10 09:13:47 2018

@author: HUDSON
"""

#IMPLEMENTAÇÃO DA FUNCAO SIGMOIDE
import numpy as np



entradas = np.array([[0,0], [0,1], [1,0], [1,1]]) #OPERADOR XOR
saidas   = np.array([[0],[1], [1], [0]])
#pesos0   = np.array([[-0.424, -0.740, -0.961], #inicializa com valores aleatorios
#                    [0.358, -0.577,-0.469]])
    
#pesos1   = np.array([[-0.017], [-0.893],[ 0.148]]) #peso da camada oculta que foi encontrado
epocas = 500000
taxaAprendizagem = 0.6
momento = 1

pesos0 =  2 * np.random.random((2,3)) - 1
pesos1 =  2 * np.random.random((3,1)) - 1

#função que representa o função sigmoid
def sigmoid(soma):
    return 1 /(1 + np.exp(-soma))
#np.exp() retorna o numero de euler(matematica) = 2.71828184590451
#O numero que esta no parenteses e o que vai elevar esse numero
# a = sigmoid(50) o retorno e 1
    
# derivando a funcao sigmoid
def sigmoidDerivada(sig):
    return sig * (1 - sig)

#a = sigmoid(0.5)
#b = sigmoidDerivada(a)
    


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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    