'''
   This file is part of GFLIB toolbox
   First Version Sept. 2018

   Cite this project as:
   Mezher M., Abbod M. (2011) Genetic Folding: A New Class of Evolutionary Algorithms.
   In: Bramer M., Petridis M., Hopgood A. (eds) Research and Development in Intelligent Systems XXVII.
   SGAI 2010. Springer, London

   Copyright (C) 20011-2018 Mohd A. Mezher (mohabedalgani@gmail.com)
'''

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
import time
from sklearn import svm
from sklearn.metrics import f1_score #classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

PATH = 'data/'
rnd_seed = 2019
def typicalsvm(params):
    '''
    Loads the dataset specified in the params dictionary, applies preprocessing and scaling and fits
    the SVM with the kernel type specified in the params
    :param params: Parameters including kernel types, number of crossvalidation splits, current dataset path and problem type.
    :return: Metrics measured on the test set with the current SVM model
    '''

    ## train Data
    trainData = pd.read_csv(params['trainData'])
    ## test Data
    testData = pd.read_csv(params['testData'])

    ##trainData.sample(frac=1).head(5)

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    train_vectors = vectorizer.fit_transform(trainData['Content'])
    test_vectors = vectorizer.transform(testData['Content'])
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel=params['kernel'])
    t0 = time.time()
    classifier_linear.fit(train_vectors, trainData['Label'])
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1 - t0
    time_linear_predict = t2 - t1

    # results
    ##print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    ##report = classification_report(testData['Label'], prediction_linear, output_dict=True)

    ##print('positive: ', report['pos'])
    ##print('negative: ', report['neg'])
   # review = "مرحبا"
    ##review_vector = vectorizer.transform([params['review']])  # vectorizing
    ##print(classifier_linear.predict(review_vector))
    ##score = report['pos']['f1-score'] - report['neg']['f1-score']



    ## if params['type'] == 'binary': #multi
      ##  if params['data'] == 'Half_Tweets_text.txt':
      ##      with open(PATH + 'multi/' + params['data']) as f:
      ##          M = np.array([])
      ##          file = f.read().split('\n')
      ##          for val in file:
      ##              tmp = np.array(val.split(','))
      ##              if len(tmp) <= 1:
      ##                  continue
      ##              if M.shape[0] == 0:
      ##                  M = tmp
      ##              else:
      ##                 M = np.vstack([M, tmp])
      ##          tmpX = M[:, :-1]
      ##         tmpY = M[:, -1]
      ##  vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
      ##  tmpX = pd.DataFrame(tmpX)
        # For each feature in data tmpX, encode feature with label encoder in case of categorical variable
      ##  for i in range(tmpX.shape[1]):
      ##      try:
      ##          tmpX.iloc[:, i] = tmpX.iloc[:, i].map(float)
      ##      except:
      ##          tmpX.iloc[:, i] = LabelEncoder().fit_transform(tmpX.iloc[:, i])
      ##  tmpY = LabelEncoder().fit_transform(tmpY)  # Transforms the label column (Y) in case it is a categorical feature
      ##  tmpX = tmpX.values
      ##  tmpX = StandardScaler().fit_transform(tmpX)  # Scales the data, so all variables will be in the same range

      ##  trainX, testX, trainY, testY = train_test_split(tmpX, tmpY, train_size=.75, random_state=rnd_seed)

    ##################################3 to not repeat ##################
    #trainX = params['trainX']
    #testX = params['testX']
    #trainY = params['trainY']
    #testY = params['testY']
    ##################################### end of not repeat ################

    # Create feature vectors
    #vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)
    ##train_vectors = vectorizer.fit_transform(trainX)
    ##test_vectors = vectorizer.transform(testX)

    # Perform classification with SVM, kernel=linear
    #classifier_linear = svm.SVC(kernel=params['kernel'])
    #t0 = time.time()
    #classifier_linear.fit(trainX, trainY)
   #t1 = time.time()
    #prediction_linear = classifier_linear.predict(testX)
   # t2 = time.time()
    #time_linear_train = t1 - t0
   # time_linear_predict = t2 - t1

    # results
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    #report = classification_report(testY, prediction_linear, output_dict=True)
    #score = f1_score(testY, prediction_linear, average='weighted', labels=np.unique(prediction_linear))
    #np.sum((yHat - y)**2) / y.size
    L = np.sum((prediction_linear - testData['Label'])**2) / len(testData['Label'])
    print(f'The value of loss {L}\n')
    fitness = accuracy_score(testData['Label'], prediction_linear) * 100
    mse = (100 - fitness) / 100
    print(f'Accuracy : {fitness}\n')

    #score = accuracy_score(testY, prediction_linear)*100

    #print("score of f1 score is:", score)
    ##print('negative: ', report['0'])
    ##print('positive: ', report['1'])
    ##print('NEUTRAL: ', report['2'])
    ##print('positive: ', report['3'])
    ##print('NEUTRAL: ', report['4'])
    #review_vector = vectorizer.transform([params['review']])  # vectorizing
    #print(classifier_linear.predict(review_vector))
   # score = report['1']['f1-score'] - report['3']['f1-score']



    return mse