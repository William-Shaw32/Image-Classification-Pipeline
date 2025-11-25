print("Running Decision Tree Training:")
import numpy as np
from ModelDecisionTree import DecisionTree # Custom Decision Tree
from sklearn.tree import DecisionTreeClassifier # Scikit Decision Tree
import pickle

# Evaluates classifications 
def eval(classifications, testingLabels):
    numCorrect = 0
    for i in range(1000):
        if(classifications[i] == testingLabels[i]):
            numCorrect += 1
    print("Test: " + str(numCorrect) + " / 1000")


def trainCustomDecisionTree(maxDepth, trainingVectors, trainingLabels, testingVectors):
    print("\nMax depth = " + str(maxDepth))
    decisionTree = DecisionTree(maxDepth=maxDepth) # Initializes custom decision tree
    print("Building custom decision tree...")
    decisionTree.fit(trainingVectors, trainingLabels) # Trains / fits the decision tree
    # Saves the tree
    fileName = f"../SavedModels/CustomTreeDepth{maxDepth}.pkl"
    with open(fileName, "wb") as fileObj:
        pickle.dump(decisionTree, fileObj)
    print("Tree successfully saved!")
    # Classifies the training vectors
    classifications = np.zeros(1000)
    for i in range(1000):
        classifications[i] = decisionTree.classify(testingVectors[i]) # Classifies a given image (Training feature vector)
    return classifications

def trainScikitDecisionTree(maxDepth, trainingVectors, trainingLabels, testingVectors):
    print("\nMax depth = " + str(maxDepth))
    decisionTree = DecisionTreeClassifier(max_depth=maxDepth, criterion="gini", random_state=0) # Initializes Scikit decision tree
    print("Building scikit decision tree...")
    decisionTree.fit(trainingVectors, trainingLabels) # Trains / fits the decision tree
    # Saves the tree
    fileName = f"../SavedModels/ScikitTreeDepth{maxDepth}.pkl"
    with open(fileName, "wb") as fileObj:
        pickle.dump(decisionTree, fileObj)
    print("Tree successfully saved!")
     # Classifies the training vectors
    classifications = decisionTree.predict(testingVectors)
    return classifications

# Main Decision Tree training function
def trainDecisionTrees():
    # Retrieves the cached feature data
    cacheData = np.load("../SavedFeatureData/FeatureData.npz")
    trainingVectors = cacheData["trainingFeatureVectors"]
    trainingLabels = cacheData["trainingLabels"]
    testingVectors = cacheData["testingFeatureVectors"]
    testingLabels = cacheData["testingLabels"]

    # Variations in max depth
    maxDepths = [10, 20, 50]

    # Trains and tests the custom decision trees for a range of depths
    print("\nStarting Custom Decision Trees...")
    for maxDepth in maxDepths:
        classifications = trainCustomDecisionTree(maxDepth, trainingVectors, trainingLabels, testingVectors)
        eval(classifications, testingLabels)

    # Trains and tests the scikit decision trees for a range of depths
    print("\nStarting Scikit Decision Trees...")
    for maxDepth in maxDepths:
        classifications = trainScikitDecisionTree(maxDepth, trainingVectors, trainingLabels, testingVectors)
        eval(classifications, testingLabels)

if __name__ == "__main__":
    trainDecisionTrees()

    