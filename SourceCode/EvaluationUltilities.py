import numpy as np

NUM_CLASSES = 10

# Maps labels as integers to labels as strings
LABELS_STR = np.array([
    "Airplanes",
    "Automobiles",
    "Birds",
    "Cats",
    "Deer",
    "Dogs",
    "Frogs",
    "Horses",
    "Ships",
    "Trucks"
])

def printConfusionMatrix(classifications, labels, modelName):
    # Converts to numpy
    classifications = np.array(classifications, dtype=int)
    labels = np.array(labels, dtype=int)
    
    # Sets the matrix
    confusionMatrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for c, l in zip(classifications, labels):
        confusionMatrix[c][l] += 1

    print("\n\n" + modelName + " Confusion Matrix:\n")

    header = " " * 13 
    for labelStr in LABELS_STR:
        header += f"{labelStr[:13]:<13}"
    print(header)

    for row in range(len(confusionMatrix)):
        rowStr = f"{LABELS_STR[row]:<13}"
        for column in range(len(confusionMatrix[row])):
            rowStr += f"{confusionMatrix[row][column]:<13}"
        print(rowStr)

def findAccuracy(classifications, labels):
    numCorrect = 0
    for i in range(1000):
        if(classifications[i] == labels[i]):
            numCorrect += 1
    return numCorrect / 1000

def findPrecision(classifications, labels):
    truePositives = [0] * 10
    predictedPositives = [0] * 10
    for i in range(1000):
        predictedPositives[int(classifications[i])] += 1
        if (classifications[i] == labels[i]):
            truePositives[int(labels[i])] += 1
    sumPrecision = 0.0
    for c in range(10):
        if predictedPositives[c] == 0:
            continue
        sumPrecision += truePositives[c] / predictedPositives[c]
    return sumPrecision / 10.0

def findRecall(classifications, labels):
    truePositives = [0] * 10
    for i in range(1000):
        if(classifications[i] == labels[i]):
            truePositives[int(labels[i])] += 1
    sumRecall = 0.0
    for c in range(10):
        sumRecall += truePositives[c] / 100.0
    return sumRecall / 10.0

def findF1(classifications, labels):
    precision = findPrecision(classifications, labels)
    recall = findRecall(classifications, labels)
    if(precision + recall == 0):
        return 0.0
    f1 = 2.0 * (precision * recall) / (precision + recall)
    return f1
