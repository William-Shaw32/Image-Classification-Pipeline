import numpy as np
from sklearn.naive_bayes import GaussianNB
from EvaluationUltilities import printConfusionMatrix, findAccuracy, findPrecision, findRecall, findF1

def printConfusionMatrices(classificationVectors, testingLabels, modelNames):
    for c, n in zip(classificationVectors, modelNames):
        printConfusionMatrix(c, testingLabels, n)

# Logarithm of the Gaussian probability density function
def logPDF(x, u, v):
    term1 = -0.5 * np.log(2.0*np.pi*v)
    term2 = (x-u)**2.0 / (2.0*v)
    return term1 - term2

# Function to computes the mean training vector for a given class
def computeMeanVector(vectorsInClass):
    sumVector = np.zeros(len(vectorsInClass[0])) # Empty sum vector
    for v in vectorsInClass: 
        sumVector += v
    meanVector = sumVector / len(vectorsInClass) # Sets mean vector
    return meanVector 

# Function to compute the variance training vector for a given class
def computeVarianceVector(vectorsInClass, meanVector):
    sumSquaresVector = np.zeros(len(vectorsInClass[0])) # Empty sum of squares vector
    for v in vectorsInClass:
        sumSquaresVector += v**2
    meanSquaresVector = sumSquaresVector / len(vectorsInClass) # Sets mean of squares vector
    varianceVector = meanSquaresVector - meanVector**2 # Var(X) = E[X^2] - E[X]^2
    return varianceVector

# Function to classify a given image (Testing vector)
def classify(testingVector, meanMatrix, varianceMatrix):
    scores = []
    # Loops through all 10 classes
    for c in range(10):
        meanVector = meanMatrix[c] # Mean vector for class c
        varianceVector = varianceMatrix[c] # Variance vector for class c
        # Finds the log-probability density of each feature given the class c: log(P(x) | c)
        densityVector = logPDF(testingVector, meanVector, varianceVector)
        score = np.sum(densityVector) # Sums the densities together to get the score for class c
        scores.append(score)  # Adds the score to the list
    classification = np.argmax(scores) # Classifies based on the highest score
    return classification 

# Function to classify all images (Testing vectors) 
def classifyAll(testingVectors, meanMatrix, varianceMatrix):
    classifications = []
    for v in testingVectors:
        classification = classify(v, meanMatrix, varianceMatrix) # Classifies given image (Testing vector)
        classifications.append(classification) # Appends classification
    return classifications

# Main Naive Bayes training and testing function
def trainTestNaiveBayes():
    # Retrieves the cached feature data
    cacheData = np.load("../SavedFeatureData/FeatureData.npz")
    trainingVectors = cacheData["trainingFeatureVectors"]
    trainingLabels = cacheData["trainingLabels"]
    testingVectors = cacheData["testingFeatureVectors"]
    testingLabels = cacheData["testingLabels"]

    # Splits the training vectors into separate lists by class
    # Training vectors are already ordered by class : Class 0: 0-499, Class 1: 500-999, Class 2: 1000-1499, ...
    trainingVectorsByClass = []
    for c in range(10):
        startIndex = c * 500
        endIndex = (c+1) * 500
        trainingVectorsByClass.append(trainingVectors[startIndex:endIndex])

    # Calculates the mean vector and variance vectors for each class and puts them in matrices
    meanMatrix = [] # 10 x 50 (classes x dimensions)
    varianceMatrix = [] # 10 x 50 (classes x dimensions)
    for c in range(10):
        meanVector = computeMeanVector(trainingVectorsByClass[c])
        varianceVector = computeVarianceVector(trainingVectorsByClass[c], meanVector)
        meanMatrix.append(meanVector)
        varianceMatrix.append(varianceVector)

    # Classifies all testing images (Testing vectors)
    customClassifications = classifyAll(testingVectors, meanMatrix, varianceMatrix)

    scikitClassifier = GaussianNB()
    scikitClassifier.fit(trainingVectors, trainingLabels)
    scikitClassifications = scikitClassifier.predict(testingVectors)

    classificationVectors = [customClassifications, scikitClassifications]
    modelNames = ["CustomNaiveBayesClassifier", "ScikitNaiveBayesClassifier"]

    printConfusionMatrices(classificationVectors, testingLabels, modelNames)

    accuracies = []
    precisions = []
    recalls = []
    f1Measures = []
    for c in classificationVectors:
        accuracies.append(findAccuracy(c, testingLabels))
        precisions.append(findPrecision(c, testingLabels))
        recalls.append(findRecall(c, testingLabels))
        f1Measures.append(findF1(c, testingLabels))
    
    return accuracies, precisions, recalls, f1Measures

    
if __name__ == "__main__":
    trainTestNaiveBayes()

