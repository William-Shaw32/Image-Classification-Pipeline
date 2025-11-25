import numpy as np

NUM_DIMENSIONS = 50 # Dimensionality of feature vectors
NUM_CLASSES = 10 # Number of classes

# Leaf class, used for leaf nodes
class Leaf:
    # Constructor
    def __init__(self):
        self.classification = None # The leaf's classification

    # Leaf's classification
    def findClassification(self, labels):
        classCounts = [0] * NUM_CLASSES # Empty list of class counts
        # Counts classes in subset of labels in leaf
        for label in labels:
            classCounts[label] += 1
        self.classification = np.argmax(classCounts) # Takes the maximum

    # Polymorphic method for recursive classification (See Node version of method)
    def route(self, trainingvector):
        return self.classification

# Node class
class Node:
    # Constructor
    def __init__(self):
        self.threshold = 0 # The node's threshold
        self.comparisonDimension = 0  # The node's dimension of comparison
        self.leftChild = None
        self.rightChild = None

    # Method to find the best split at a given node (Comparison dimension and threshold)
    def findBestSplit(self, trainingVectors, trainingLabels):
        bestThreshold = None # Best threshold accross all dimensions
        bestImpurity = float("inf") # Best impurity across all dimensions - Initialized at highest
        bestDimension = None # Best dimension
        # Loops through each dimension
        for i in range(NUM_DIMENSIONS):
           dimensionVector = trainingVectors[:,i] # Creates dimension vector for given dimension
           # Finds the best threshold and impurity in a given dimension
           bestThresholdInDimension, bestImpurityInDimension = self._findBestThresholdInDimension(dimensionVector, trainingLabels)
           # Updates the best running comparison information
           if(bestImpurityInDimension < bestImpurity):
                bestThreshold = bestThresholdInDimension
                bestImpurity = bestImpurityInDimension
                bestDimension = i
        # Stores the best threshold and dimension, forgets the best impurity (not needed anymore)
        self.threshold = bestThreshold
        self.comparisonDimension = bestDimension
    
    # Method to find the best threshold in a given dimension
    def _findBestThresholdInDimension(self, dimensionVector, labels):
        # Sorts the dimension vector and labels
        sortedIndices = np.argsort(dimensionVector)
        sortedDimensionVector = dimensionVector[sortedIndices]
        sortedLabels = labels[sortedIndices]
        bestThreshold = None # Best threshhold in the given dimension
        bestImpurity = float("inf") # Best impurity in the given dimension - Initialized at highest
        n = len(sortedDimensionVector) # 5000
        leftClassCounts = [0] * NUM_CLASSES # Counts classes in the left child
        rightClassCounts = [0] * NUM_CLASSES # Counts classes in the right child
        # Places all the labels on the right side
        for label in sortedLabels:
            rightClassCounts[label] += 1
        # Loops through all the features
        for i in range(n-1):
            label = sortedLabels[i] # Gets the label
            # Shifts a label from the right side to the left side
            leftClassCounts[label]  += 1
            rightClassCounts[label] -= 1
            # If the features are identical, no threshold can be made
            if(sortedDimensionVector[i] == sortedDimensionVector[i+1]): 
                continue
            # Calculates threshold
            threshold = (sortedDimensionVector[i] + sortedDimensionVector[i+1]) / 2
            # Shifts the sizes in accordance with the labelss
            leftSize = i + 1
            rightSize = n - leftSize
            # Calculates the weighted impurity at the given threshold accross both children
            impurity = self._calculateWeightedGiniImpurity(leftClassCounts, rightClassCounts, leftSize, rightSize)
            # Updates the best running comparison information
            if(impurity < bestImpurity):
                bestThreshold = threshold
                bestImpurity = impurity
        return bestThreshold, bestImpurity
    
    # Method to calculate the weighted gini impurity at a given threshold accross both children
    def _calculateWeightedGiniImpurity(self, leftClassCounts, rightClassCounts, leftSize, rightSize):
        nL = leftSize # Number of labels in left side            
        nR = rightSize # Number of labels in right side
        n = nL + nR # Total number of labels
        # Calculates left child impurity
        leftChildImpurity = self._calcuateGiniImpurity(leftClassCounts, nL)
        # Calculates right child impurity
        rightChildImpurity = self._calcuateGiniImpurity(rightClassCounts, nR)
        # Calculates the weighted impurity
        weightedImpurity = (nL/n) * leftChildImpurity + (nR/n) * rightChildImpurity
        return weightedImpurity
    
    # Method to calculate the gini impurity of a single node or leaf
    def _calcuateGiniImpurity(self, classCounts, totalCount):
        sum = 0
        # Loops through each class
        for i in range(NUM_CLASSES):
            p = classCounts[i] / totalCount # Probability of class c out of all the labels
            sum += p**2 # Adds the square of the probability to the sum
        return 1 - sum  # Returns sum - 1 (Impurity)
    
    # Splits the training data once a threshold and dimension have been determined
    def splitTrainingData(self, trainingVectorsSubset, trainingLabelsSubset):
        leftVectors = [] # Vectors being sent to the left child
        rightVectors = [] # Vectors being sent to the right child
        leftLabels = [] # Labels being sent to the left child
        rightLabels = [] # Labels being sent to the right child
        n = len(trainingVectorsSubset)
        
        # Loops through the training vector subset
        for i in range(n):
            # Any vectors with component smaller than the threshold in the comparision dimension sent left
            if(trainingVectorsSubset[i][self.comparisonDimension] <= self.threshold):
                leftVectors.append(trainingVectorsSubset[i])
                leftLabels.append(trainingLabelsSubset[i])
            # Otherwise sent right
            else:
                rightVectors.append(trainingVectorsSubset[i])
                rightLabels.append(trainingLabelsSubset[i])
        # Conversion to numpy arrays
        leftVectors = np.array(leftVectors)
        rightVectors = np.array(rightVectors)
        leftLabels = np.array(leftLabels)
        rightLabels = np.array(rightLabels)
        return leftVectors, leftLabels, rightVectors, rightLabels
    
    # Recursive routing method to route a testing vector down the decision tree
    def route(self, testingVector):
        # No explicit base case is needed because route() is called polymorphically on Leaf
        # The stopping condition is when the method is called on a leaf node
        # Any vectors with component smaller than the threshold in the comparision dimension sent left
        if(testingVector[self.comparisonDimension] <= self.threshold):
            return self.leftChild.route(testingVector)
        # Otherwise sent right
        else:
            return self.rightChild.route(testingVector)

# Decision Tree class
class DecisionTree:
    # Constructor
    def __init__(self, maxDepth=50):
        self.root = None
        self.maxDepth = maxDepth
        self.numNodesCreated = 0

    # Fits or trains the tree on the training vectors
    def fit(self, trainingVectors, trainingLabels):
        depth = 0
        # Recursively builds the tree from the root node
        self.root = self._buildTree(trainingVectors, trainingLabels, depth)
        # Prints the total number of nodes created in the tree
        print("    Number of nodes created: " + str(self.numNodesCreated))
    
    # Recursively builds the tree from the root node
    def _buildTree(self, trainingVectorsSubset, trainingLabelsSubset, depth):
        # Checks for leaf node (Base cases)
        baseCase1 = depth >= self.maxDepth-1 # Hit max depth
        baseCase2 = np.all(trainingLabelsSubset == trainingLabelsSubset[0]) # Pure leaf node
        if(baseCase1 or baseCase2):
            leaf = Leaf()
            leaf.findClassification(trainingLabelsSubset) # Sets leaf node classification
            return leaf
        # Otherwise creates node
        node = Node()
        node.findBestSplit(trainingVectorsSubset, trainingLabelsSubset) # Sets threshold and comparisonDimension in node
        leftVectors, leftLabels, rightVectors, rightLabels = node.splitTrainingData(trainingVectorsSubset, trainingLabelsSubset) # Node splits training data
        self.numNodesCreated += 1
        if(self.numNodesCreated % 100 == 0 or self.numNodesCreated == 1): # Periodically prints out the number of nodes created
            print("    Number of nodes created: " + str(self.numNodesCreated))
        node.leftChild = self._buildTree(leftVectors, leftLabels, depth+1) # Left child recursive assignment
        node.rightChild = self._buildTree(rightVectors, rightLabels, depth+1) # Right child recursive assignment
        return node # Returns non-leaf
    
    # Classifies a given testing vector
    def classify(self, testingVector):
        return self.root.route(testingVector)