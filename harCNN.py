from tabnanny import verbose
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D

from tensorflow.keras.utils import to_categorical


import pandas as pd
import numpy as np
import math

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from torch import int64

import dataPreprocess

#write code so that it is reproducible -> set the random seed for tf, numpy, os for python


###################################  Sensitive Attributes   #######################################
#INDEXES OF SENTSITIVE ATTRIBUTES ZERO INDEXED
#the indexes of males "1"
male_indexes = [0, 1, 3, 5, 8, 10, 11, 12, 13, 14, 16, 19, 20, 21]
#the datasets of females "0"
female_indexes = [2, 4, 6, 7, 9, 15, 17, 18, 22, 23]


def pool_by_attribute(attr_indexes, participantData):
    for p in range(0, len(participantData)):

        attr_test_X = np.array([])
        attr_test_y = np.array([])

        # this is an index associated with attribute
        if(p in attr_indexes):
            attr_test_X = np.concatenate( [attr_test_X, participantData[p][2]] ) if len( attr_test_X ) != 0 else participantData[p][2]
            attr_test_y = np.concatenate( [attr_test_y, participantData[p][3]] ) if len(attr_test_y) != 0 else participantData[p][3]

    #return training data and labels that have been pooled by an attribute
    return( [attr_test_X, attr_test_y] ) 


def check_fairness(model, PooledTest, prot_test_truth, prot_test_pred, unprot_test_truth, unprot_test_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=prot_test_truth, y_pred=prot_test_pred, labels=[0, 1, 2, 3, 4, 5]).ravel()

    prot_tpr = tp / (tp + fn)
    prot_fpr = fp/(fp+tn)
    prot_dp = (tp+fp)/len(prot_test_truth)

    tn, fp, fn, tp = confusion_matrix(y_true=prot_test_truth, y_pred=prot_test_pred, labels=[0, 1, 2, 3, 4, 5]).ravel()
    unprot_tpr = tp / (tp + fn)
    unprot_fpr = fp/(fp+tn)
    unprot_dp = (tp+fp)/len(unprot_test_truth)
    print("DI:", max(prot_tpr / unprot_tpr, unprot_tpr / prot_tpr))
    print("EOP:", abs(prot_tpr-unprot_tpr))
    print("Avg EP diff:", 0.5 * (abs(prot_tpr-unprot_tpr) + abs(prot_fpr-unprot_fpr)))
    print("SPD:", abs(prot_dp-unprot_dp))



################################### PREPROCESSING FUNCTIONS #######################################




# by calculating window size I won't have to contatenate windows/numpy arrays (slow)
def calcWindowAmount(participantData, windowSize):
    trainCounter = 0
    testCounter = 0

    start = 0
    end = start + windowSize

    #loop through training dataset
    while(end < len(participantData[0])):
        x = participantData[1][start]
        y = participantData[1][end-1]
        if participantData[1][start].all() == participantData[1][end-1].all():
            trainCounter +=1

        start = end
        end = end+ windowSize

    start = 0
    end = start + windowSize

    while(end < len(participantData[2])):

        if participantData[3][start].all() == participantData[3][end-1].all():
            testCounter +=1

        start = end
        end = end+ windowSize

    return [trainCounter, testCounter]
    
def windowData(participantData, windowSize):
    #go through the data looking at each 50 window -> creating a window for this if label is same for first and last
    #calculate the number of viable windows -> initialize a dataframe (3 dimensions) / numpy array of zeros (50 reduced -> 1)

    trainSize, testSize = calcWindowAmount(participantData, windowSize)
    trainSet = np.zeros( (trainSize, windowSize, 12) ) 
    trainLabels = np.zeros( (trainSize, 6), dtype= np.int64 )#should be 2d one hot vectors ##REWORK NECESSARY!
    testSet = np.zeros( (testSize, windowSize, 12) )
    testLabels = np.zeros( (testSize, 6) , dtype= np.int64) #should be 2d one hot vectors ##REWORK NECESSARY!

    trainCounter = 0
    testCounter = 0

    start = 0
    end = start + windowSize

    #loop through training dataset
    while(end < len(participantData[0])):

        # if the label is consistent throughout the window
        if(participantData[1][start].all() == participantData[1][end-1].all()):
            windowArray = participantData[0].iloc[start:end, :].to_numpy().astype(np.float32)
            trainSet[trainCounter] = windowArray
            trainLabels[trainCounter] = participantData[1][start]
            trainCounter +=1
        start = end
        end = end+ windowSize

    start = 0
    end = start + windowSize

    while(end < len(participantData[2])):

        # if the label is consistent throughout the window
        if(participantData[3][start].all() == participantData[3][end-1].all()):
            windowArray = participantData[2].iloc[start:end, :].to_numpy().astype(np.float32)
            testSet[testCounter] = windowArray
            testLabels[testCounter] = participantData[3][start]
            testCounter +=1

        start = end
        end = end+ windowSize

    return [trainSet, trainLabels, testSet, testLabels]

def getPoolSize(windowDatasets, windowSize):

    pool_train_size = 0
    pool_test_size = 0

    for participant in windowDatasets:
        pool_train_size += len(participant[0])
        pool_test_size += len(participant[2])

    return [pool_train_size, pool_test_size]

def poolWindows(windowDatasets, windowSize):
    
    trainSize, testSize = getPoolSize(windowDatasets, windowSize)

    pool_train_windows = np.zeros( (trainSize, windowSize, 12) )
    pool_train_labels = np.zeros( (trainSize, 6), dtype=np.int64)
    pool_test_windows = np.zeros( (testSize, windowSize, 12) )
    pool_test_labels = np.zeros( (testSize, 6), dtype=np.int64)


    start_train = 0
    start_test = 0

    for participant in windowDatasets:
        xc = len(participant[0])
        pool_train_windows[start_train:(start_train + len(participant[0]))] = participant[0]
        pool_train_labels[start_train:(start_train + len(participant[1]))] = participant[1]
        pool_test_windows[start_test:(start_test + len(participant[2]))] = participant[2]
        pool_test_labels[start_test:(start_test + len(participant[3]))] = participant[3]

        start_train += len(participant[0])
        start_test += len(participant[2])
    
    return[pool_train_windows, pool_train_labels, pool_test_windows, pool_test_labels]

def participantWindows(participantData, windowSize):
    partWindows = []

    for participant in participantData:
        partWindows.append(windowData(participant, windowSize))
    
    return partWindows

########################## CNN / TRAINING FUNCTIONS #####################################################

def sensor_activity( n_timesteps, n_features, n_outputs): #(64, 50, 12) labels should be in form this is what I'm passing into the cnn.
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=2, activation='hard_sigmoid', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Conv1D(filters=128, kernel_size=8, activation='hard_sigmoid'))
    model.add(MaxPooling1D(pool_size=8))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #I HAVE TAKEN OUT COMPILE LINE FOR FL DP

    #print(model.summary())

    return model

def printAverages(acc_list, loss_list):

    sum_acc = 0
    sum_loss = 0

    for acc in acc_list:
        sum_acc += acc
    for loss in loss_list:
        sum_loss += loss
    print("Average Loss -> ", (sum_loss/len(loss_list)))
    print("Average Accuracy -> ", (sum_acc/len(acc_list)))


def individual_cnn(readyData):
    counter = 1
    avg_accuracy = []
    avg_loss = []
    for participant in readyData:
        #print("Participant", counter,  "Local Training ->")
        david_cnn = sensor_activity(n_timesteps=50, n_features=12, n_outputs=6)

        david_cnn.fit(participant[0], participant[1], batch_size=32, epochs=20, verbose=0, validation_data=(participant[2], participant[3]))
        score = david_cnn.evaluate(participant[2], participant[3], verbose=0)
        #print('Test loss:', score[0]) 
        print('participant -> ' + str(counter) +' Test accuracy:', score[1])

        avg_accuracy.append(score[1])
        avg_loss.append(score[0])

        counter +=1
    
    printAverages(avg_accuracy, avg_loss)

#not set up for k fold validation
def centralized_cnn(pooledData):
    # counter = 1
    # avg_accuracy = []
    # avg_loss = []

    #print("Participant", counter,  "Local Training ->")
    david_cnn = sensor_activity(n_timesteps=50, n_features=12, n_outputs=6)

    david_cnn.fit(pooledData[0], pooledData[1], batch_size=64, epochs=20, verbose=0, validation_data=(pooledData[2], pooledData[3]))
    score = david_cnn.evaluate(pooledData[2], pooledData[3], verbose=0)
    #print('Test loss:', score[0]) 
    print('Pooled Test accuracy:', score[1])

    # avg_accuracy.append(score[1])
    # avg_loss.append(score[0])

    # counter +=1
    
    #printAverages(avg_accuracy, avg_loss)






if __name__ == "__main__":
    #this is the data for individual training and pooled (hasn't been put into windows yet)
    partData = dataPreprocess.getIndividualDatasets(24)
    dataPreprocess.normalizeParticipants(partData)
    pooledData = dataPreprocess.getCentralDataset(partData)

    part_windows = participantWindows(partData, 50)
    pooled_windows = poolWindows(part_windows, 50)

    #tests individuals with their own data
    individual_cnn(part_windows)
    #tests centralized (all participants pool)
    centralized_cnn(pooled_windows)

"""
Individual Training Results (no k fold)
participant -> 1 Test accuracy: 0.6706827282905579
participant -> 2 Test accuracy: 0.7269076108932495
participant -> 3 Test accuracy: 0.6959999799728394
participant -> 4 Test accuracy: 0.7321428656578064
participant -> 5 Test accuracy: 0.6985645890235901
participant -> 6 Test accuracy: 0.7903929948806763
participant -> 7 Test accuracy: 0.7591836452484131
participant -> 8 Test accuracy: 0.7736625671386719
participant -> 9 Test accuracy: 0.7456140518188477
participant -> 10 Test accuracy: 0.723849356174469
participant -> 11 Test accuracy: 0.7890295386314392
participant -> 12 Test accuracy: 0.8356807231903076
participant -> 13 Test accuracy: 0.8872548937797546
participant -> 14 Test accuracy: 0.8141592741012573
participant -> 15 Test accuracy: 0.7299578189849854
participant -> 16 Test accuracy: 0.7777777910232544
participant -> 17 Test accuracy: 0.837837815284729
participant -> 18 Test accuracy: 0.8127490282058716
participant -> 19 Test accuracy: 0.9094076752662659
participant -> 20 Test accuracy: 0.699999988079071
participant -> 21 Test accuracy: 0.8581818342208862
participant -> 22 Test accuracy: 0.8035714030265808
participant -> 23 Test accuracy: 0.7155963182449341
participant -> 24 Test accuracy: 0.8970588445663452
Average Loss ->  0.6901249003907045
Average Accuracy ->  0.7785526389877001

Centralized Training results (no participant fold)
Pooled Test accuracy: 0.9530474543571472

predictions after normalization
participant -> 1 Test accuracy: 0.718875527381897
participant -> 2 Test accuracy: 0.8634538054466248
participant -> 3 Test accuracy: 0.9440000057220459
participant -> 4 Test accuracy: 0.7589285969734192
participant -> 5 Test accuracy: 0.7129186391830444
participant -> 6 Test accuracy: 0.7729257345199585
participant -> 7 Test accuracy: 0.8938775658607483
participant -> 8 Test accuracy: 0.6666666865348816
participant -> 9 Test accuracy: 0.7763158082962036
participant -> 10 Test accuracy: 0.8870292901992798
participant -> 11 Test accuracy: 0.8565400838851929
participant -> 12 Test accuracy: 0.8028169274330139
participant -> 13 Test accuracy: 0.8529411554336548
participant -> 14 Test accuracy: 0.8230088353157043
participant -> 15 Test accuracy: 0.8312236070632935
participant -> 16 Test accuracy: 0.8352490663528442
participant -> 17 Test accuracy: 0.7297297120094299
participant -> 18 Test accuracy: 0.8167330622673035
participant -> 19 Test accuracy: 0.8850173950195312
participant -> 20 Test accuracy: 0.7545454502105713
participant -> 21 Test accuracy: 0.8145454525947571
participant -> 22 Test accuracy: 0.7008928656578064
participant -> 23 Test accuracy: 0.8165137767791748
participant -> 24 Test accuracy: 0.8529411554336548
Average Loss ->  0.5666292483607928
Average Accuracy ->  0.8069870918989182
Pooled Test accuracy: 0.9512757062911987

second run
participant -> 1 Test accuracy: 0.7028112411499023
participant -> 2 Test accuracy: 0.8634538054466248
participant -> 3 Test accuracy: 0.9399999976158142
participant -> 4 Test accuracy: 0.78125
participant -> 5 Test accuracy: 0.7224880456924438
participant -> 6 Test accuracy: 0.8427947759628296
participant -> 7 Test accuracy: 0.8775510191917419
participant -> 8 Test accuracy: 0.8148148059844971
participant -> 9 Test accuracy: 0.8070175647735596
participant -> 10 Test accuracy: 0.6778242588043213
participant -> 11 Test accuracy: 0.8649789094924927
participant -> 12 Test accuracy: 0.8215962648391724
participant -> 13 Test accuracy: 0.8284313678741455
participant -> 14 Test accuracy: 0.8274336457252502
participant -> 15 Test accuracy: 0.8354430198669434
participant -> 16 Test accuracy: 0.8544061183929443
participant -> 17 Test accuracy: 0.8108108043670654
participant -> 18 Test accuracy: 0.8884462118148804
participant -> 19 Test accuracy: 0.9442508816719055
participant -> 20 Test accuracy: 0.7409090995788574
participant -> 21 Test accuracy: 0.7854545712471008
participant -> 22 Test accuracy: 0.875
participant -> 23 Test accuracy: 0.8486238718032837
participant -> 24 Test accuracy: 0.8921568393707275
Average Loss ->  0.5418892620752255
Average Accuracy ->  0.8269977966944376
Pooled Test accuracy: 0.9489723443984985



# CL
# FL 
# FL + DP 
# dpsgd ? code from sikha, disparate impact 



#64 batch
Average Loss ->  0.568134089310964
Average Accuracy ->  0.8258574182788531

#32 batch
Average Loss ->  0.3842432089149952
Average Accuracy ->  0.8938409214218458

test 2 (32 is better for the fit function)
Average Loss ->  0.37107059949388105
Average Accuracy ->  0.8927095904946327
"""