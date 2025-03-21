# Modèle de classification avec plus de couches
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, output_dim=10):
        super(CNN, self).__init__()

        # Définir les couches convolutives
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)  # Images MNIST (1 canal)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Couches fully-connected
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Taille après réduction par les convolutions (28x28 -> 3x3 après convolutions et pooling)
        self.fc2 = nn.Linear(512, output_dim)  # 10 classes pour MNIST

        # Couches de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 + ReLU + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 + ReLU + Pooling
        #x = self.pool(F.relu(self.conv3(x)))  # Conv3 + ReLU + Pooling
        x = x.view(-1, 64 * 7 * 7)  # Aplatir les données pour la couche fully-connected
        x = F.relu(self.fc1(x))  # Fully connected 1
        x = self.fc2(x)  # Fully connected 2 (sortie)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        
        # Couches du bloc résiduel
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # Couches de normalisation
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Couches de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # La connexion résiduelle : ajuster les dimensions si nécessaire
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)  # Connexion résiduelle

        # Application des convolutions + activation + normalisation
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Ajouter la connexion résiduelle à l'output
        out += identity
        out = F.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, output_dim=10):
        super(ResNet, self).__init__()
        
        # Première couche de convolution
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Blocs résiduels
        self.layer1 = BasicBlock(32, 64)
        self.layer2 = BasicBlock(64, 128)
        
        # Couches de pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Couches fully-connected
        self.fc1 = nn.Linear(128 * 7 * 7, 512)  # 128 canaux, 7x7 après convolutions et pooling
        self.fc2 = nn.Linear(512, output_dim)   # 10 classes pour MNIST
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Première couche de convolution + BatchNorm + ReLU
        x = self.pool(x)  # Pooling après la première couche
        
        # Passer les données à travers les blocs résiduels
        x = self.layer1(x)
        x = self.pool(x)  # Pooling après le premier bloc résiduel
        x = self.layer2(x)
        
        # Aplatir la sortie pour la couche fully-connected
        x = x.view(x.size(0), -1)
        
        # Passer à travers les couches fully-connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
