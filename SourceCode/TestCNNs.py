from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Subset
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from ModelsCNN import NormalVGG11, ShallowedVGG11, SmallerKernelVGG11, LargerKernelVGG11
from EvaluationUltilities import printConfusionMatrix, findAccuracy, findPrecision, findRecall, findF1

def printConfusionMatrices(classificationVectors, testingLabels, modelNames):
    for c, n in zip(classificationVectors, modelNames):
        printConfusionMatrix(c, testingLabels, n)

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

def testCNN(model, testingLoader, device):
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
    return classifications, testingLabels

def testCNNs():
    transformToTensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    )

    # Retrieves the training and testing sets
    testingSet = datasets.CIFAR10(root='../Images', train=False, download=True, transform=transformToTensor)

    # Finds the indices of the first 500 images of each class 
    testingIndices = findIndices(testingSet, 100)

    # Creates the training and testing subsets
    testingSubset = Subset(testingSet, testingIndices)

     # Creates the training and testing loaders
    batchSize = 32
    testingLoader = DataLoader(testingSubset, batch_size=batchSize, shuffle=False)

    # Instantiates models
    normalVGG11 = NormalVGG11()
    shallowedVGG11 = ShallowedVGG11()
    smallerKernelVGG11 = SmallerKernelVGG11()
    largerKernelVGG11 = LargerKernelVGG11()

     # Loads the models
    state_dict = torch.load("../SavedModels/NormalVGG11.pth", map_location="cpu", weights_only=True) 
    normalVGG11.load_state_dict(state_dict)
    state_dict = torch.load("../SavedModels/ShallowedVGG11.pth", map_location="cpu", weights_only=True) 
    shallowedVGG11.load_state_dict(state_dict)
    state_dict = torch.load("../SavedModels/SmallerKernelVGG11.pth", map_location="cpu", weights_only=True) 
    smallerKernelVGG11.load_state_dict(state_dict)
    state_dict = torch.load("../SavedModels/LargerKernelVGG11.pth", map_location="cpu", weights_only=True) 
    largerKernelVGG11.load_state_dict(state_dict)

    models = [normalVGG11, shallowedVGG11, smallerKernelVGG11, largerKernelVGG11]
    modelNames = ["NormalVGG11", "ShallowedVGG11", "SmallerKernelVGG11", "LargerKernelVGG11"]

    # Set the device as GPU if available
    if (torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"

    classificationVectors = []
    # Test the models
    for model in models:
        model.to(device)
        classifications, testingLabels = testCNN(model, testingLoader, device)
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
    testCNNs()