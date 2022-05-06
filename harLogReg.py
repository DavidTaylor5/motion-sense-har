
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix

import dataPreprocess

def individualTrain(partData):
    counter = 1
    every_accuracy = []

    for participant in partData:

        logreg = LogisticRegression(max_iter=150)
        logreg.fit(participant[0], participant[1])
        y_pred = logreg.predict(participant[2])

        print("Data for Participant " + str(counter) + " ----->")
        a_accuracy = accuracy_score(participant[3], y_pred)
        print("Accuracy:", a_accuracy)
        every_accuracy.append(a_accuracy)
        # print("Precision:", recall_score(participant[3], y_pred, average='micro'))
        # print("Recall:", recall_score(participant[3], y_pred, average='micro'))

        with open('logregAccuracy.txt', 'a') as f:
            f.write("Accuracy for participant " + str(counter) + "-> " + str(a_accuracy) + "\n")

        counter +=1
    
    accuracy_total = 0
    for acc in every_accuracy:
        accuracy_total += acc
    avg_accuracy = accuracy_total / len(every_accuracy)

    with open('logregAccuracy.txt', 'a') as f:
        f.write("Average accuracy for participant -> " + str(avg_accuracy) + "\n")



def centralTrain(pooledData):
    
    logreg = LogisticRegression()
    logreg.fit(pooledData[0][0], pooledData[0][1])
    y_pred = logreg.predict(pooledData[0][2])

    print("Data for Centralized Training ----->")
    print("Accuracy:", accuracy_score(pooledData[0][3], y_pred))
    # print("Precision:", recall_score(pooledData[0][3], y_pred, average='micro'))
    # print("Recall:", recall_score(pooledData[0][3], y_pred, average='micro'))

    with open('logregAccuracy.txt', 'a') as f:
        f.write("Accuracy for Centralized Training -> " + str(accuracy_score(pooledData[0][3], y_pred)) + '\n')


def vector_to_class(partData):
    for participant in partData:
        participant[1] = participant[1].argmax(axis=1)
        participant[3] = participant[3].argmax(axis=1)


if __name__ =="__main__":


    partData = dataPreprocess.getIndividualDatasets(24)
    dataPreprocess.normalizeParticipants(partData)
    vector_to_class(partData)
    pooledData = dataPreprocess.getCentralDataset(partData)

    individualTrain(partData)
    centralTrain(pooledData)