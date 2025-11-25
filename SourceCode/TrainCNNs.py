print("Running Convolutional Neural Network Training:")
print("Importing Pytorch libraries...")
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from ModelsCNN import NormalVGG11, ShallowedVGG11, SmallerKernelVGG11, LargerKernelVGG11

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

# Function to collect the indices of the first 500 images of each class  
def findIndices(dataset, numPerClass):
    indicesByClass = [[] for _ in range(10)] # Empty matrix of indices
    countsByClass = [0]*10 # Empty list of counts
    # Loops through dataset
    for i in range(len(dataset)):
        labelInt = dataset.targets[i] # Extracts label
        # Keeps adding indices for given class until limit is reached 
        if(len(indicesByClass[labelInt]) < numPerClass):
            indicesByClass[labelInt].append(i)
            countsByClass[labelInt] += 1
            # Breaks when enough indices are gathered in each class
            if(sum(countsByClass) == numPerClass * 10):
                break
    # Flattens the matrix of indices
    indices = []
    for i in indicesByClass:
        indices.extend(i)
    return indices

# Function to classify the testing vectors using the set of weights from a given epoch
def test(model, testingLoader, device):
    model.eval() # Puts the model in evaluation mode
    classifications = []
    testingLabels  = []
    with torch.no_grad(): # Disables computational graph / gradient tracking (No training or backpropogation)
        for batchVectors, batchLabels in testingLoader:
            batchVectors = batchVectors.to(device) # Puts batch of images on the GPU
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
    numEpochs = 15
    numBatches = len(trainingLoader)
    bestEpoch = Epoch(0,0,0)

    # Path for saving weights
    savePath = "../SavedModels/" + modelName + ".pth"

    # Trains the neural network
    print("\nTraining " + modelName + "...")
    for i in range(numEpochs):
        model.train() # Puts the model in training mode
        sumLoss = 0
        for batchImages, batchLabels in trainingLoader:
            batchImages = batchImages.to(device) # Puts batch of vectors on the GPU
            batchLabels = batchLabels.to(device) # Puts batch of labels on the GPU
            optimizer.zero_grad() # Resets the gradients
            logits = model(batchImages) # Produces logits (Forward Pass)
            loss = lossFunction(logits, batchLabels) # Calculates / initializes the loss
            sumLoss += loss.item() # Adds the loss to the sum
            loss.backward() # Backpropogates
            optimizer.step() # Updates the weights
        print("    Epoch " + str(i+1) + ": ")
        avgLoss = sumLoss / numBatches # Calculates the average loss
        print("        Avg loss = " + str(avgLoss))
        totalCorrect = test(model, testingLoader, device) # Tests the model
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


# Main Convolutional Neural Network training function
def trainCNNs():

    transformToTensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    # Retrieves the training and testing sets
    print("Retrieving images...")
    trainingSet = datasets.CIFAR10(root="../Images", train=True, download=True, transform=transformToTensor)
    testingSet = datasets.CIFAR10(root='../Images', train=False, download=True, transform=transformToTensor)

    # Finds the indices of the first 500 images of each class  
    trainingIndices = findIndices(trainingSet, 500)
    testingIndices = findIndices(testingSet, 100)
    
    # Creates the training and testing subsets
    trainingSubset = Subset(trainingSet, trainingIndices)
    testingSubset = Subset(testingSet, testingIndices)

    # Creates the training and testing loaders
    batchSize = 32
    trainingLoader = DataLoader(trainingSubset, batch_size=batchSize, shuffle=True)
    testingLoader = DataLoader(testingSubset, batch_size=batchSize, shuffle=False)

    # Instantiates models
    normalVGG11 = NormalVGG11()
    shallowedVGG11 = ShallowedVGG11()
    smallerKernelVGG11 = SmallerKernelVGG11()
    largerKernelVGG11 = LargerKernelVGG11()
    models = [normalVGG11, shallowedVGG11, smallerKernelVGG11, largerKernelVGG11]
    modelNames = ["NormalVGG11", "ShallowedVGG11", "SmallerKernelVGG11", "LargerKernelVGG11"]

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
    trainCNNs()
