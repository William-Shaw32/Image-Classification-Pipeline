print("Running Image Preprocessing: ")
print("Importing Pytorch libraries...")
from torchvision import datasets
from torchvision import transforms
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np

# Function to extract feature vectors from images
def extractFeatures(model, loader, device):
    featureVectorBatchList = []
    labelsBatchList = []
    with torch.no_grad(): # Disables computational graph / gradient tracking (No training or backpropogation)
        # Loops through each batch
        for batchImages, batchLabels in loader: 
            batchImages = batchImages.to(device) # Puts batch of images on the GPU
            featureVectorBatch = model(batchImages) # Extracts feature vectors (Forward Pass)
            featureVectorBatch = featureVectorBatch.cpu() # Puts the batch of vectors on the CPU
            featureVectorBatchList.append(featureVectorBatch) # Appends new batch of vectors to the list
            labelsBatchList.append(batchLabels) # Appends new batch of labels to the list
    # Flattens the lists
    featureVectors = torch.cat(featureVectorBatchList, dim=0)
    labels = torch.cat(labelsBatchList, dim=0)
    return featureVectors, labels

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

# Main Preprocessing Function
def runPreprocessing():

    # Transform for the images
    transform224by224 = transforms.Compose(
        [
            # Resize: 224 x 224
            transforms.Resize((224, 224)),
            # Convert to Tensor
            transforms.ToTensor(),
            # Normalize: Values from https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # Retrieves the training and testing sets
    print("Retrieving images...")
    trainingSet = datasets.CIFAR10(root="../Images", train=True, download=True, transform=transform224by224)
    testingSet = datasets.CIFAR10(root='../Images', train=False, download=True, transform=transform224by224)

    # Finds the indices of the first 500 images of each class  
    trainingIndices = findIndices(trainingSet, 500)
    testingIndices = findIndices(testingSet, 100)

    # Creates the training and testing subsets
    # Training subset is ordered: Class 0: 0-499, Class 1: 500-999, Class 2: 1000-1499, ...
    trainingSubset = Subset(trainingSet, trainingIndices)
    # Testing subset is ordered: Class 0: 0-99, Class 1: 100-199, Class 2: 200-299, ...
    testingSubset = Subset(testingSet, testingIndices)

    # Creates the training and testing loaders
    trainingLoader = DataLoader(trainingSubset, batch_size=32, shuffle=False)
    testingLoader = DataLoader(testingSubset, batch_size=32, shuffle=False)

    # Instantiates the model with pretrained weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Removes fully connected layer
    model.fc = nn.Identity()
    # Puts the model into evaluation mode
    model.eval()

    # Set the device as GPU if available
    print("Checking GPU availability...")
    if (torch.cuda.is_available()):
        device = "cuda"
        print("Moving to GPU...")
    else:
        device = "cpu"
        print("Staying on CPU...")
    model = model.to(device)

    # Extracts the feature vectors from the images
    print("Extracting training feature vectors (5000)...")
    trainingFeatureVectors, trainingLabels = extractFeatures(model, trainingLoader, device)
    print("Extracting testing feature vectors (1000)...")
    testingFeatureVectors, testingLabels = extractFeatures(model, testingLoader, device)

    # Converts all the tensors into numpy arrays
    trainingFeatureVectors = trainingFeatureVectors.numpy()
    trainingLabels = trainingLabels.numpy()
    testingFeatureVectors = testingFeatureVectors.numpy()
    testingLabels = testingLabels.numpy()

    # Condenses the vectors into 50 dimensions (Deterministically)
    print("Condensing vectors...")
    pca = PCA(n_components=50, svd_solver="full") # svd_solver="full" for determinism
    # Fits once
    trainingFeatureVectors = pca.fit_transform(trainingFeatureVectors) # Condenses training vectors
    # Uses the same fit
    testingFeatureVectors = pca.transform(testingFeatureVectors) # Condenses testing vectors

    # Caches all the feature data
    print("Saving feature data...")
    np.savez(
        "../SavedFeatureData/FeatureData.npz",
        trainingFeatureVectors=trainingFeatureVectors,
        trainingLabels=trainingLabels,
        testingFeatureVectors=testingFeatureVectors,
        testingLabels=testingLabels)
    
    print("Successfully extracted and cached feature data")

if __name__ == "__main__":
    runPreprocessing()