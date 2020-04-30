import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plots

from astropy.io import ascii


def main():
    linClassifier(2)
    # dataPlotter() # Used in task 1b

def linClassifier(inputSize):
    alpha = 0.005
    batchSize = 30
    iterations = 1000
    trainingSize = 30
    validationSize = 50 - trainingSize

    setosa = [0,0,1]
    versicolor = [0,1,0]
    virginica = [1,0,0]
    labels = (setosa, versicolor, virginica)

    names = {   2 : 'setosa',
                1 : 'versicolor',
                0 : 'virginica'}

    W_k = np.zeros(shape=(3,inputSize))
    for n in range(inputSize):
        W_k[0][n] = random.random()
        W_k[1][n] = random.random()
        W_k[2][n] = random.random()
    
    # Load data
    (xallTraining), (validationSet, validationLabels) = getDataSets(loadData(),labels, trainingSize, validationSize, 5, columns=[2,1,0])
    # Train model
    W_k = train(xallTraining, (validationSet, validationLabels), labels, names, W_k, batchSize, inputSize, iterations, alpha)

    # Generate confusion matrix for training data
    allTrainingData, allTrainingLabels = getBatch(xallTraining, labels, batchSize, inputSize)
    trainingPreds, trainingActuals = runOnData(allTrainingData, allTrainingLabels, names, W_k)
    trainingConfusionMatrix, trainingErrorRate = getConfMatrix(trainingPreds, trainingActuals, list(names.values()))
    #Generate confusion matrix for validation data
    validationPreds, validationActuals = runOnData(validationSet, validationLabels, names, W_k) 
    validationConfusionMatrix, validationErrorRate = getConfMatrix(validationPreds, validationActuals, list(names.values()))

    plots.plotConfusionMatrix(validationConfusionMatrix, "Validation confusion matrix\nError rate = {}".format(validationErrorRate), names)
    plots.plotConfusionMatrix(trainingConfusionMatrix, "Training confusion matrix\nError rate = {}".format(trainingErrorRate), names)


def runOnData(data, labels, names, W_k):
    inputSize = data.shape[1]
    print("inputsize ", inputSize)
    actuals = np.array([])
    predictions = np.array([])

    for index in range(labels.shape[0]): 
        x_k = data[:][index].reshape(inputSize, 1)
        t_k = labels[:][index].reshape(3, 1)
        z_k = np.matmul(W_k, x_k)
        g_k = sigmoid(z_k)
        prediction = getPredictionFromOutput(g_k)
        predictions = np.append(predictions, names[np.argmax(prediction)])
        actuals = np.append(actuals, names[np.argmax(t_k)])

    return predictions, actuals

def train(xallTraining, validation, labels, names, W_k, batchSize, inputSize, iterations, alpha, errorMargin = 0.04):
    trainingError = 1000
    prevTrainingError = 2000
    counter = 1
    eRate = 1
    while eRate > errorMargin:
        if prevTrainingError > eRate:
            # print("it++")
            iterations += 200
        else:
            alpha *= 0.8
            # print("alpha: ", alpha)
        prevTrainingError = eRate
        trainingError = 0
        for it in range(iterations):
            # Get new batch
            batch, batchLabels = getBatch(xallTraining, labels, batchSize, inputSize)
            # dataSize, nrOfFeatures = batch.shape

            grad_MSE_W = 0
            MSE = 0
            for index in range(batchSize):
                x_k = batch[:][index].reshape(inputSize, 1)
                t_k = batchLabels[:][index].reshape(3, 1)
                z_k = np.matmul(W_k, x_k)
                g_k = sigmoid(z_k)
                MSE += float(0.5*np.matmul(np.transpose(g_k-t_k),(g_k-t_k)))
                grad_MSE_gk = g_k-t_k
                grad_g_zk = np.multiply(g_k,(1-g_k))
                grad_W_zk = np.transpose(x_k)
                grad_MSE_W += np.matmul(np.multiply(grad_MSE_gk, grad_g_zk), grad_W_zk)
            # print("Iteration nr: ", it, "\tMSE/batchSize = ", MSE/batchSize)
            W_k = W_k - alpha*grad_MSE_W
            trainingError += MSE/batchSize
        trainingError = trainingError/iterations

        (validationSet, validationLabels) = validation
        validationPreds, validationActuals = runOnData(validationSet, validationLabels, names, W_k) 
        validationConfusionMatrix, eRate = getConfMatrix(validationPreds, validationActuals, list(names.values()))

        print("Training epoch: ", counter, "\t\tError: ", eRate)
        counter += 1
    
    return W_k

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loadData():
    x1data = ascii.read("/home/simon/repos/TTT4275_classification_project/Iris_TTT4275/class_1", delimiter=',').as_array()
    x2data = ascii.read("/home/simon/repos/TTT4275_classification_project/Iris_TTT4275/class_2", delimiter=',').as_array()
    x3data = ascii.read("/home/simon/repos/TTT4275_classification_project/Iris_TTT4275/class_3", delimiter=',').as_array()
    return (x1data, x2data, x3data)

def getDataSets(dataList, labels, trainingSize, validationSize, inputSize, columns=None):
    x1data, x2data, x3data = dataList
    setosa, versicolor, virginica = labels
    # Traning data
    x1allTraining = np.array([[x1data[i][0], x1data[i][1], x1data[i][2], x1data[i][3], 1] for i in range(trainingSize)])
    x2allTraining = np.array([[x2data[i][0], x2data[i][1], x2data[i][2], x2data[i][3], 1] for i in range(trainingSize)])
    x3allTraining = np.array([[x3data[i][0], x3data[i][1], x3data[i][2], x3data[i][3], 1] for i in range(trainingSize)])
    # Validation data
    x1allValidation = np.array([[x1data[i][0], x1data[i][1], x1data[i][2], x1data[i][3], 1] for i in range(trainingSize,50)])
    x2allValidation = np.array([[x2data[i][0], x2data[i][1], x2data[i][2], x2data[i][3], 1] for i in range(trainingSize,50)])
    x3allValidation = np.array([[x3data[i][0], x3data[i][1], x3data[i][2], x3data[i][3], 1] for i in range(trainingSize,50)])

    validationSet = x1allValidation[:][:validationSize]
    validationSet = np.append(validationSet, x2allValidation[:][:validationSize])
    validationSet = np.append(validationSet, x3allValidation[:][:validationSize])
    validationSet = validationSet.reshape(validationSize*3, inputSize)
    validation_t_k = [setosa for i in range(validationSize)]
    validation_t_k = np.append(validation_t_k, [versicolor for i in range(validationSize)])
    validation_t_k = np.append(validation_t_k, [virginica for i in range(validationSize)])
    validation_t_k = validation_t_k.reshape(validationSize*3, 3)

    # Partion and shuffle data
    random.shuffle(x1allTraining)
    random.shuffle(x2allTraining)
    random.shuffle(x3allTraining)

    # Remove specified columns, if any
    if columns != None:
        for col in columns:
            x1allTraining = np.delete(x1allTraining, col, axis=1)
            x2allTraining = np.delete(x2allTraining, col, axis=1)
            x3allTraining = np.delete(x3allTraining, col, axis=1)
            validationSet = np.delete(validationSet, col, axis=1)

    return (x1allTraining, x2allTraining, x3allTraining), (validationSet, validation_t_k)

def getBatch(data, labels, batchSize, inputSize):
    x1allTraining, x2allTraining, x3allTraining = data
    setosa, versicolor, virginica = labels

    batch = x1allTraining[:][:int(batchSize/3)]
    batch = np.append(batch, x2allTraining[:][:int(batchSize/3)])
    batch = np.append(batch, x3allTraining[:][:int(batchSize/3)])
    batch = batch.reshape(batchSize, inputSize)

    # labels
    batch_t_k = [setosa for i in range(int(batchSize/3))]
    batch_t_k = np.append(batch_t_k, [versicolor for i in range(int(batchSize/3))])
    batch_t_k = np.append(batch_t_k, [virginica for i in range(int(batchSize/3))])
    batch_t_k = batch_t_k.reshape(batchSize, 3)

    return (batch, batch_t_k)

def getPredictionFromOutput(output):
    prediction = np.zeros(shape=(3,1))
    index = np.argmax(output, axis=0)
    prediction[index] = 1
    return prediction.reshape(1,3)

def getConfMatrix(predictions, labels, labelNames):
    actual = pd.Categorical(labels, categories=labelNames)
    predicted = pd.Categorical(predictions, categories=labelNames)
    confMatrix = pd.crosstab(actual, predicted, normalize=False)
    errorRate = 0
    matrix = confMatrix.to_numpy()
    for row in range(confMatrix.shape[0]):
        for col in range(confMatrix.shape[1]):
            if row != col:
                errorRate += matrix[row][col]
    errorRate = errorRate/predictions.shape[0]
    return confMatrix, errorRate


if __name__ == "__main__":
    main()