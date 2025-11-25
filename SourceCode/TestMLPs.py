import numpy as np
from ModelsMLP import NormalMLP, DeepenedMLP, NarrowedMLP, WidenedMLP
import torch
from torch.utils.data import TensorDataset, DataLoader
from EvaluationUltilities import printConfusionMatrix, findAccuracy, findPrecision, findRecall, findF1

def printConfusionMatrices(classificationVectors, testingLabels, modelNames):
    for c, n in zip(classificationVectors, modelNames):
        printConfusionMatrix(c, testingLabels, n)

def testMLP(model, testingLoader, device):
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
    return classifications

def testMLPs():
    # Retrieves the cached feature data
    cacheData = np.load("../SavedFeatureData/FeatureData.npz")
    testingVectors = cacheData["testingFeatureVectors"]
    testingLabels = cacheData["testingLabels"]

    # Converts the cached feature data into Pytorch tensors
    testingVectors  = torch.from_numpy(testingVectors).float()
    testingLabels  = torch.from_numpy(testingLabels).long()

    # Combines the vectors and labels into a single dataset
    testingDataset = TensorDataset(testingVectors, testingLabels)

    # Creates the testing loader
    batchSize = 32
    testingLoader = DataLoader(testingDataset, batch_size=batchSize, shuffle=False)

    # Instantiates models
    normalMLP = NormalMLP()
    deepenedMLP = DeepenedMLP()
    narrowedMLP = NarrowedMLP()
    widenedMLP = WidenedMLP()

    # Loads the models
    state_dict = torch.load("../SavedModels/NormalMLP.pth", map_location="cpu", weights_only=True) 
    normalMLP.load_state_dict(state_dict)
    state_dict = torch.load("../SavedModels/DeepenedMLP.pth", map_location="cpu", weights_only=True) 
    deepenedMLP.load_state_dict(state_dict)
    state_dict = torch.load("../SavedModels/NarrowedMLP.pth", map_location="cpu", weights_only=True) 
    narrowedMLP.load_state_dict(state_dict)
    state_dict = torch.load("../SavedModels/WidenedMLP.pth", map_location="cpu", weights_only=True) 
    widenedMLP.load_state_dict(state_dict)

    models = [normalMLP, deepenedMLP, narrowedMLP, widenedMLP]
    modelNames = ["NormalMLP", "DeepenedMLP", "NarrowedMLP", "WidenedMLP"]

    # Set the device as GPU if available
    if (torch.cuda.is_available()):
        device = "cuda"
    else:
        device = "cpu"

    classificationVectors = []
    # Test the models
    for model in models:
        model.to(device)
        classifications = testMLP(model, testingLoader, device)
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
    testMLPs()