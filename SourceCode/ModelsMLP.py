import torch.nn as nn

# Normal multilayer perceptron
class NormalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = nn.Sequential(
            # First layer
            nn.Linear(50, 512),
            nn.ReLU(),
            # Second layer
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Third layer
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.arch(x)

# Deepened to 4 layers
class DeepenedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = nn.Sequential(
            # First layer
            nn.Linear(50, 512),
            nn.ReLU(),
            # Second layer
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Third layer
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            # Fourth layer
            nn.Linear(512, 10)
        )
    def forward(self, x):
        return self.arch(x)

# Narrowed to 128   
class NarrowedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = nn.Sequential(
            # First layer
            nn.Linear(50, 128),
            nn.ReLU(),
            # Second layer
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # Third layer
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.arch(x)

# Widened to 2048   
class WidenedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.arch = nn.Sequential(
            # First layer
            nn.Linear(50, 2048),
            nn.ReLU(),
            # Second layer
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            # Third layer
            nn.Linear(2048, 10)
        )
    def forward(self, x):
        return self.arch(x)