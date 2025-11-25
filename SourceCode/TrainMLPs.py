print("Running Multilayer Perceptron Training:")
print("Importing Pytorch libraries...")
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from ModelsMLP import NormalMLP, DeepenedMLP, NarrowedMLP, WidenedMLP

# Class to store information about a given epoch
class Epoch():
    # Constructor
    def __init__(self, epochNum, avgLoss, totalCorrect):
        self.epochNum = epochNum
        self.avgLoss = avgLoss
        self.totalCorrect = totalCorrect
    # __str__ overwrite
    def __str__(self):
        return ("    Epoch " + str(self.epochNum) + ": \n"
        + "        Avg loss = " + str(self.avgLoss) + "\n"
        + "        Test: " + str(self.totalCorrect) + " / 1000\n")
    
# Function to classify the testing vectors using the set of weights from a given epoch
def testEpoch(model, testingLoader, device):
    model.eval() # Puts the model in evaluation mode
    classifications = []
    testingLabels  = []
    with torch.no_grad(): # Disables computational graph / gradient tracking (No training or backpropogation)
        for batchVectors, batchLabels in testingLoader:
            batchVectors = batchVectors.to(device) # Puts batch of vectors on the GPU
            logits = model(batchVectors) # Produces logits (Forward Pass)
            batchClassifications = torch.argmax(logits, dim=1) # Classifies each image in the batch (Feature vector)
            batchClassifications = batchClassifications.cpu().tolist() # Puts the classifications on the CPU as a list
            batchLabels = batchLabels.tolist() # Changes the labels to a list
            classifications.extend(batchClassifications) # Adds the batch classifications to the total classifications
            testingLabels.extend(batchLabels) # Adds the batch labels to the total labels
    # Prints the total number of correct classifications
    totalCorrect = 0
    for i in range(1000):
        if (classifications[i] == testingLabels[i]):
            totalCorrect += 1
    return totalCorrect

def trainModel(model, modelName, device, trainingLoader, testingLoader):
    # Instantiates the loss function and the optimizer
    lossFunction = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training hyper parameters
    numEpochs = 20
    numBatches = len(trainingLoader)
    bestEpoch = Epoch(0,0,0)

    # Path for saving weights
    savePath = "../SavedModels/" + modelName + ".pth"

    # Trains the neural network
    print("\nTraining " + modelName + "...")
    for i in range(numEpochs):
        model.train() # Puts the model in training mode
        sumLoss = 0
        for batchVectors, batchLabels in trainingLoader:
            batchVectors = batchVectors.to(device) # Puts batch of vectors on the GPU
            batchLabels = batchLabels.to(device)  # Puts batch of labels on the GPU
            optimizer.zero_grad() # Resets the gradients
            logits = model(batchVectors) # Produces logits (Forward Pass)
            loss = lossFunction(logits, batchLabels) # Calculates / initializes the loss
            sumLoss += loss.item() # Adds the loss to the sum
            loss.backward() # Backpropogates
            optimizer.step() # Updates the weights
        print("    Epoch " + str(i+1) + ": ")
        avgLoss = sumLoss / numBatches # Calculates the average loss
        print("        Avg loss = " + str(avgLoss))
        totalCorrect = testEpoch(model, testingLoader, device) # Tests the model
        print("        Test: " + str(totalCorrect) + " / 1000")
        # Updates the best running epoch
        if(totalCorrect > bestEpoch.totalCorrect):
            bestEpoch = Epoch(i+1, avgLoss, totalCorrect)
            torch.save(model.state_dict(), savePath)
    
    # Prints the best epoch
    print("\nBest Epoch: ")
    print(bestEpoch)

    # Success message
    print(modelName + " Successfully Trained!")

# Main Multilayer Perceptron training function
def trainMLPs():
    # Retrieves the cached feature data
    cacheData = np.load("../SavedFeatureData/FeatureData.npz")
    trainingVectors = cacheData["trainingFeatureVectors"]
    trainingLabels = cacheData["trainingLabels"]
    testingVectors = cacheData["testingFeatureVectors"]
    testingLabels = cacheData["testingLabels"]

    # Converts the cached feature data into Pytorch tensors
    trainingVectors = torch.from_numpy(trainingVectors).float()
    testingVectors  = torch.from_numpy(testingVectors).float()
    trainingLabels = torch.from_numpy(trainingLabels).long()
    testingLabels  = torch.from_numpy(testingLabels).long()

    # Combines the vectors and labels into a single dataset
    trainingDataset = TensorDataset(trainingVectors, trainingLabels)
    testingDataset = TensorDataset(testingVectors, testingLabels)

    # Creates the training and testing loaders
    batchSize = 32
    trainingLoader = DataLoader(trainingDataset, batch_size=batchSize, shuffle=True)
    testingLoader = DataLoader(testingDataset, batch_size=batchSize, shuffle=False)

    # Instantiates models
    normalMLP = NormalMLP()
    deepenedMLP = DeepenedMLP()
    narrowedMLP = NarrowedMLP()
    widenedMLP = WidenedMLP()
    models = [normalMLP, deepenedMLP, narrowedMLP, widenedMLP]
    modelNames = ["NormalMLP", "DeepenedMLP", "NarrowedMLP", "WidenedMLP"]

    # Set the device as GPU if available
    print("Checking GPU availability...")
    if (torch.cuda.is_available()):
        device = "cuda"
        print("Moving to GPU...")
    else:
        device = "cpu"
        print("staying on CPU...")

    # Trains models
    for model, modelName in zip(models, modelNames):
        model = model.to(device)
        trainModel(model, modelName, device, trainingLoader, testingLoader)

if __name__ == "__main__":
    trainMLPs()