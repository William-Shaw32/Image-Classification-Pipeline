import torch.nn as nn
import torch

# Normal class for the VGG11 convolutional neural network
class NormalVGG11(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()

        kernelSize = 3
        padding = 1
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernelSize, 1, padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernelSize, 1, padding),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fourth block
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fifth block
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    # Passes a vector through the convolutional layers
    def passConvBlocks(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
    # Passes a vector through the entire neural network (Forward pass)
    def forward(self, x):
        x = self.passConvBlocks(x)
        x = torch.flatten(x, 1)
        x  = self.classifier(x)
        return x
    
# Shallowed by removing the 5th block
class ShallowedVGG11(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()

        kernelSize = 3
        padding = 1
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernelSize, 1, padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernelSize, 1, padding),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fourth block
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.MaxPool2d(2,2)
        )
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    # Passes a vector through the convolutional layers
    def passConvBlocks(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
    
    # Passes a vector through the entire neural network (Forward pass)
    def forward(self, x):
        x = self.passConvBlocks(x)
        x = torch.flatten(x, 1)
        x  = self.classifier(x)
        return x
    

# Decresed kernel size to 1 for the VGG11 convolutional neural network
class SmallerKernelVGG11(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()

        kernelSize = 1
        padding = 0
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernelSize, 1, padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernelSize, 1, padding),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fourth block
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fifth block
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    # Passes a vector through the convolutional layers
    def passConvBlocks(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
    # Passes a vector through the entire neural network (Forward pass)
    def forward(self, x):
        x = self.passConvBlocks(x)
        x = torch.flatten(x, 1)
        x  = self.classifier(x)
        return x
    


# Decresed kernel size to 5 for the VGG11 convolutional neural network
class LargerKernelVGG11(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()

        kernelSize = 5
        padding = 2
        # First block
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernelSize, 1, padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Second block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernelSize, 1, padding),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Third block
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernelSize, 1, padding),  
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fourth block
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fifth block
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernelSize, 1, padding),  
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    # Passes a vector through the convolutional layers
    def passConvBlocks(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x
    
    # Passes a vector through the entire neural network (Forward pass)
    def forward(self, x):
        x = self.passConvBlocks(x)
        x = torch.flatten(x, 1)
        x  = self.classifier(x)
        return x