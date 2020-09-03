'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score, mean_squared_error

from kernel import Kernel
import time
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = 'data/'
rnd_seed = 2019
MAX_ITER = 100


def calcfitness(pop, params):
    '''
    Reads the data of type params['type'] and with the data path params['data'],
    then fits the SVC or SVR model depending on the params['type'] with the custom
    kernel, determined by the "pop" parameter. Calculates the resulting metrics for the
    input population.

    :param pop: Population, which will determine the custom kernel for SVM model
    :param params: Parameters, containing the info about population and about the task we are solving
    :return: -MSE for regression task, Accuracy * 100 for the binary and multi classification tasks
    '''

    models = []
    fitness = []
    #if params['type'] == 'binary':
    # here to return all code i wrote for Arabic Sentiment
    ##################################3 to not repeat ##################
    trainX = params['trainX']
    testX = params['testX']
    trainY = params['trainY']
    testY = params['testY']
    ##################################### end of not repeat ################
    shapetrainX = trainX.shape
    shapetrainY = trainY.shape
    for i in range(params['popSize']):
        ind = pop[i]  # Population consists of params['popSize'] kernel variations
        k = Kernel(ind)
        # k.kernel
        #svm1 = SVC(max_iter=MAX_ITER, kernel='linear', probability=True)  # create an SVM model with custom kernel

        svm = SVC(max_iter=MAX_ITER, kernel=k.kernel, probability=True)
        svm.fit(trainX, trainY)
        label = svm.predict(testX)
        models.append(svm)

        #L = np.sum(label != testY) / len(testY)
        L = np.sum((label - testY) ** 2) / len(testY)
        print(f'The value of loss {L}\n')
        #fitness = accuracy_score(testY, label) * 100
        #mse = (100 - fitness) / 100

        # removed from pranthesis
        fitness.append((100 - (accuracy_score(testY, label) * 100)))
        print(f'Accuracy : {fitness}\n')
        #fitness.append(mse)
    return fitness, models, testX, testY
