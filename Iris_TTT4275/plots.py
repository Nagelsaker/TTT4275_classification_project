import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plotHistogram(axs, data, minVal, maxVal, nrOfBins):
    bins = np.zeros(nrOfBins)
    xlabels = np.zeros(nrOfBins)
    for i in range(nrOfBins):
        xlabels[i] = round((minVal + i*(maxVal-minVal)/nrOfBins), 2)
        bins[i] = minVal + i*(maxVal-minVal)/nrOfBins
    axs.hist(data, bins, edgecolor='black', linewidth=1.2)
    plt.xticks(xlabels)

def plotConfusionMatrix(confMatrix, title, names):
    plt.imshow(confMatrix)
    plt.colorbar()
    plt.title(title)
    ticks = [name for name in names.values()]
    vals = [0, 1, 2]
    plt.xticks(vals, ticks)
    plt.yticks(vals, ticks)
    plt.show()

def dataPlotter():
    (x1, x2, x3) = loadData()

    class1 = np.array([[x1[i][0], x1[i][1], x1[i][2], x1[i][3], 1] for i in range(50)])
    class2 = np.array([[x2[i][0], x2[i][1], x2[i][2], x2[i][3], 1] for i in range(50)])
    class3 = np.array([[x3[i][0], x3[i][1], x3[i][2], x3[i][3], 1] for i in range(50)])

    fig, axs = plt.subplots(3, 4, sharex='col', sharey='row')
    fig.set_size_inches(10, 5.5)
    
    axs[0,0].set(ylabel='Setosa')
    axs[1,0].set(ylabel='Versicolor')
    axs[2,0].set(ylabel='Virginica')
    axs[0,0].set_title("Sepal length")
    axs[0,1].set_title("Sepal width")
    axs[0,2].set_title("Petal length")
    axs[0,3].set_title("Petal width")
    bins = 12

    # Class 1
    # Sepal length
    plotHistogram(axs[0,0], class1[:,0], 4.3, 7.9, bins)
    # Sepal width
    plotHistogram(axs[0,1], class1[:,1], 2.0, 4.4, bins)
    # Petal length
    plotHistogram(axs[0,2], class1[:,2], 1.0, 6.9, bins)
    # Petal width
    plotHistogram(axs[0,3], class1[:,3], 0.1, 2.5, bins)
    
    # Class 2
    # Sepal length
    plotHistogram(axs[1,0], class2[:,0], 4.3, 7.9, bins)
    # Sepal width
    plotHistogram(axs[1,1], class2[:,1], 2.0, 4.4, bins)
    # Petal length
    plotHistogram(axs[1,2], class2[:,2], 1.0, 6.9, bins)
    # Petal width
    plotHistogram(axs[1,3], class2[:,3], 0.1, 2.5, bins)
    
    # Class 3
    # Sepal length
    plotHistogram(axs[2,0], class3[:,0], 4.3, 7.9, bins)
    # Sepal width
    plotHistogram(axs[2,1], class3[:,1], 2.0, 4.4, bins)
    # Petal length
    plotHistogram(axs[2,2], class3[:,2], 1.0, 6.9, bins)
    # Petal width
    plotHistogram(axs[2,3], class3[:,3], 0.1, 2.5, bins)
    

    for ax in axs.flat:
        ax.label_outer()
    plt.show()