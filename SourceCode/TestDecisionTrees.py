import numpy as np
import pickle
from ModelDecisionTree import DecisionTree # Custom Decision Tree
from sklearn.tree import DecisionTreeClassifier # Scikit Decision Tree
from EvaluationUltilities import printConfusionMatrix, findAccuracy, findPrecision, findRecall, findF1

# Evaluates classifications 

def printConfusionMatrices(classificationVectors, testingLabels, modelNames):
    for c, n in zip(classificationVectors, modelNames):
        printConfusionMatrix(c, testingLabels, n)

def eval(classifications, testingLabels):
    numCorrect = 0
    for i in range(1000):
        if(classifications[i] == testingLabels[i]):
            numCorrect += 1
    print("Test: " + str(numCorrect) + " / 1000")

def loadModel(filePath):
    with open(filePath, "rb") as fileObj:
        return pickle.load(fileObj)
    
def testCustomDecisionTree(decisionTree, testingVectors):
    # Classifies the training vectors
    classifications = np.zeros(1000)
    for i in range(1000):
        classifications[i] = decisionTree.classify(testingVectors[i]) # Classifies a given image (Training feature vector)
    return classifications

def testDecisionTrees():
    cacheData = np.load("../SavedFeatureData/FeatureData.npz")
    testingVectors = cacheData["testingFeatureVectors"]
    testingLabels = cacheData["testingLabels"]

    customTreeDepth10 = loadModel("../SavedModels/CustomTreeDepth10.pkl")
    customTreeDepth20 = loadModel("../SavedModels/CustomTreeDepth20.pkl")
    customTreeDepth50 = loadModel("../SavedModels/CustomTreeDepth50.pkl")
    scikitTreeDepth10 = loadModel("../SavedModels/ScikitTreeDepth10.pkl")
    scikitTreeDepth20 = loadModel("../SavedModels/ScikitTreeDepth20.pkl")
    scikitTreeDepth50 = loadModel("../SavedModels/ScikitTreeDepth50.pkl")

    trees = [customTreeDepth10, customTreeDepth20, customTreeDepth50, scikitTreeDepth10, scikitTreeDepth20, scikitTreeDepth50]
    modelNames = ["CustomTreeDepth10", "CustomTreeDepth20", "CustomTreeDepth50", "ScikitTreeDepth10", "ScikitTreeDepth20", "ScikitTreeDepth50"]

    classificationVectors = []
    for i in range(3):
        classifications = testCustomDecisionTree(trees[i], testingVectors)
        classificationVectors.append(classifications)
    for i in range(3,6):
        classifications = trees[i].predict(testingVectors)
        classificationVectors.append(classifications)

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
    testDecisionTrees()