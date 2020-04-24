import random
import math
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii


def main():
    linClassifier(5)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

<<<<<<< HEAD


def linClassifier(inputSize):
    alpha = 0.005
    batchSize = 90
    iterations = 1000

    validationSize = 20

    setosa = [0,0,1]
    versicolor = [0,1,0]
    virginica = [1,0,0]


    W_k = np.zeros(shape=(3,inputSize))
    for n in range(inputSize):
        W_k[0][n] = random.random()
        W_k[1][n] = random.random()
        W_k[2][n] = random.random()
    # W_k = W_k.reshape(inputSize, 3)
    
    # Loading data
=======
def loadData():
>>>>>>> 9eaf473a45ba3b4a1f91213d7600e72668f8236a
    x1data = ascii.read("/home/simon/repos/TTT4275_classification_project/Iris_TTT4275/class_1", delimiter=',').as_array()
    x2data = ascii.read("/home/simon/repos/TTT4275_classification_project/Iris_TTT4275/class_2", delimiter=',').as_array()
    x3data = ascii.read("/home/simon/repos/TTT4275_classification_project/Iris_TTT4275/class_3", delimiter=',').as_array()
    return list(x1data, x2data, x3data)

def getDataSets(dataList, labels, validationSize, inputSize):
    x1data, x2data, x3data = dataList
    setosa, versicolor, virginica = labels
    # Traning data
    x1allTraining = np.array([[x1data[i][0], x1data[i][1], x1data[i][2], x1data[i][3], 1] for i in range(30)])
    x2allTraining = np.array([[x2data[i][0], x2data[i][1], x2data[i][2], x2data[i][3], 1] for i in range(30)])
    x3allTraining = np.array([[x3data[i][0], x3data[i][1], x3data[i][2], x3data[i][3], 1] for i in range(30)])
    # Validation data
    x1allValidation = np.array([[x1data[i][0], x1data[i][1], x1data[i][2], x1data[i][3], 1] for i in range(30,50)])
    x2allValidation = np.array([[x2data[i][0], x2data[i][1], x2data[i][2], x2data[i][3], 1] for i in range(30,50)])
    x3allValidation = np.array([[x3data[i][0], x3data[i][1], x3data[i][2], x3data[i][3], 1] for i in range(30,50)])

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

    return list(x1allTraining, x2allTraining, x3allTraining), list(validationSet, validation_t_k)

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

    return list(batch, batch_t_k)

def linClassifier(inputSize):
    alpha = 0.005
    batchSize = 90
    iterations = 1000
    validationSize = 20

    setosa = [0,0,1]
    versicolor = [0,1,0]
    virginica = [1,0,0]
    labels = list(setosa, versicolor, virginica)

    W_k = np.zeros(shape=(3,inputSize))
    for n in range(inputSize):
        W_k[0][n] = random.random()
        W_k[1][n] = random.random()
        W_k[2][n] = random.random()
    
    # Loading data
    (xallTraining), (validationset, validationLabels) = getDataSets(loadData(),labels, validationSize, inputSize)
    
    trainingError = 1
    prevTrainingError = 2
    errorMargin = 0.1
    while trainingError > errorMargin:
        if prevTrainingError > trainingError:
            iterations *= 2
        else:
            alpha *= 0.8
        trainingError = 0
        for it in range(iterations):
            # Get new batch
            batch, batchLabels = getBatch(xallTraining, labels, batchSize, inputSize)
            dataSize, nrOfFeatures = batch.shape

            grad_MSE_W = 0
            MSE = 0
            for index in range(batchSize):
                x_k = batch[:][index].reshape(inputSize, 1)
                t_k = batchLabels[:][index].reshape(3, 1)
                # (grad_MSE_W_temp, MSE_temp) = train(x_k, t_k, W_k, dataSize, MSE, grad_MSE_W)

                z_k = np.matmul(W_k, x_k)
                g_k = sigmoid(z_k)
                MSE += float(0.5*np.matmul(np.transpose(g_k-t_k),(g_k-t_k)))
                grad_MSE_gk = g_k-t_k
                grad_g_zk = np.multiply(g_k,(1-g_k))
                grad_W_zk = np.transpose(x_k)
                grad_MSE_W += np.matmul(np.multiply(grad_MSE_gk, grad_g_zk), grad_W_zk)
            print("Iteration nr: ", it, "\tMSE/batchSize = ", MSE/batchSize)
            W_k = W_k - alpha*grad_MSE_W
            trainingError += MSE/batchSize
        trainingError = trainingError/iterations

    validationSize, nrOfFeatures = validationSet.shape
    
    MSE_valid = 0
    for index in range(validationSize):
        x_k = validationSet[:][index]
        t_k = validation_t_k[:][index]
        MSE_temp, g_k = runOnNetwork(x_k, t_k, W_k, MSE_valid)
        MSE_valid = MSE_temp
        print("\nOutput:\t\t", g_k, "\nActual Class:\t", t_k)
    print("Validation error: ", MSE_valid)


def runOnNetwork(x_k, t_k, W_k, MSE):
    z_k = np.matmul(W_k, x_k)
    g_k = sigmoid(z_k)
    MSE += float(0.5*np.matmul(np.transpose(g_k-t_k),(g_k-t_k)))
    return MSE, g_k

def train(x_k, t_k, W_k, dataSize, MSE, grad_MSE_W):
    z_k = np.matmul(W_k, x_k)
    g_k = sigmoid(z_k)
    MSE += float(0.5*np.matmul(np.transpose(g_k-t_k),(g_k-t_k)))
    grad_MSE_gk = g_k-t_k
    grad_g_zk = np.multiply(g_k,(1-g_k))
    grad_W_zk = np.transpose(x_k)
    grad_MSE_W += np.matmul(np.multiply(grad_MSE_gk, grad_g_zk), grad_W_zk)
    return grad_MSE_W, MSE

def updateConfMatrix(true, classified, confMatrix):
    confMatrix[true][classified] += 1
    return confMatrix    

def plotConfusionMatrix(conf):    
    data=[[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5],[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5],[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5],[10,10,10,5,5,5,2,2,2,4,4,4,4,4,4,5,5,5]]
    fig= plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    rows=['AB','BC','SK','MB']
    cols=["Gas", "Wk/Wk", "Yr/Yr","Oil", "Wk/Wk", "Yr/Yr","Bit", "Wk/Wk", "Yr/Yr","Total", "Wk/Wk", "Yr/Yr","Hor", "Wk/Wk", "Yr/Yr","Vert/Dir", "Wk/Wk", "Yr/Yr"]
    table = ax.table(cellText=data, colLabels=cols,rowLabels=rows, loc='upper center',cellLoc='center')
    table.auto_set_font_size(True)
    table.scale(1.5,1.5)
    #plt.savefig(filenameTemplate2, format='png',bbox_inches='tight')


if __name__ == "__main__":
    main()