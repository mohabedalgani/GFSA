#!/usr/bin/env python
# coding: utf-8

# In[1]:
# step 1 Import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
import pandas as pd
from inipop import inipop
from genpop import genpop
from tipicalsvm import typicalsvm
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore', 'Solver terminated early.*')

PATH = 'data/'
rnd_seed = 2019

# create folder for graphs generated during the run
if not os.path.exists('images/'):
    os.makedirs('images/')
else:
    files = glob.glob('images/*')
    for f in files:
        os.remove(f)

print('Type the maximum length of the chromosome: ')
max_chromosome_length = int(input())  # the maximum total length of the chromosome

params = dict()
params['type'] = 'binary'  # problem type
params['trainData'] = 'ArabSen_train_85_percentage.csv'  # path to data file
params['testData'] = 'ArabSen_test_15_percentage.csv'  # path to data file
params['data'] = 'ArabSen.csv'  # path to data file
params['kernel'] = 'rbf,linear,polynomial,gf'  #
params['mutProb'] = 0.7  # mutation probability
params['crossProb'] = 0.5  # crossover probability
params['maxGen'] = 10  # max generation
params['popSize'] = 50  # population size
params['crossVal'] = 2  # number of cross validation slits
params['opList'] = ['Plus_s', 'Minus_s', 'Multi_s', 'Plus_v', 'Minus_v', 'x', 'y']  # Operators and operands
params['review'] = "ماعندك شي ابد"


##################################### change the text file format ########################
models = []
fitness = []
params['data'] = 'testArabicText.txt'
if params['type'] == 'binary':
    if params['data'] == 'testArabicText.txt':
        with open(PATH + 'binary/' + params['data']) as f:
            M = np.array([])
            file = f.read().split('\n')
            for val in file:
                tmp = np.array(val.split(','))
                if len(tmp) <= 1:
                    continue
                if M.shape[0] == 0:
                    M = tmp
                else:
                    M = np.vstack([M, tmp])
            tmpX = M[:, :-1]
            tmpY = M[:, -1]
        tmpX = pd.DataFrame(tmpX)
        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
        for i in range(tmpX.shape[1]):
            try:
                tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
            except:
                tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])

        tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
        tmpX = tmpX.values
        tmpX = MinMaxScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range
trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.85, random_state=rnd_seed)
######################################### End of preprocessing ################



########################################## New Param inserted ################
params['trainX'] = trainX  # number of cross validation slits
params['testX'] = testX  # number of cross validation slits
params['trainY'] = trainY  # number of cross validation slits
params['testY'] = testY  # number of cross validation slits
########################################## ned of New Param inserted ################
#print(f'''Data Set : {DATA_PATH + params['data']}\n\n''')
kernels = ['poly', 'rbf', 'linear', 'gf'] #, 'gf'

totalTime = dict()
for ker in kernels:
    totalTime[ker] = list()

for i in range(2):
    for index, kernel in enumerate(kernels):
        params['kernel'] = kernel
        print(f'''SVM Kernel : {params['kernel']} \n''')
        if kernel == 'gf':
            print(f'''Max Generation : {params['maxGen']}\n''')
            print(f'''Population Size : {params['popSize']}\n''')
            print(f'''CrossOver Probability : {params['crossProb']}\n''')
            print(f'''Mutation Probability : {params['mutProb']}\n\n''')
            pop = inipop(params, max_chromosome_length)  # generate initial population
            mse = genpop(pop, params, i)  # get the best population from the initial one
        else:
            mse = typicalsvm(params)
        totalTime[kernel].append(mse)
        print('\n')


###############################################3 نهاية ملفي التحديث الاخير

# Boxplot of errors for each kernel
plt.figure(figsize = (7, 7))
kernels = ['poly', 'rbf', 'linear', 'gf']
plt.boxplot([totalTime['poly'], totalTime['rbf'], totalTime['linear'], totalTime['gf']]) # , totalTime['gf']
plt.xticks(np.arange(1, 5), kernels)
plt.title('MSE for each svm kernel')
plt.xlabel('SVM kernel')
plt.ylabel('Mean Square Error')
plt.ioff()
plt.savefig('images/mse.png')
plt.show()