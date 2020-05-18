## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    # inspired by the NaimishNet architecture: https://arxiv.org/pdf/1710.00977.pdf

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv1 = nn.Conv2d(in_channels= 1, out_channels=32, kernel_size=5, stride=1)
        self.drop1 = nn.Dropout(0.1)
        
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.drop2 = nn.Dropout(0.2)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.drop3 = nn.Dropout(0.3)
        
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.drop4 = nn.Dropout(0.4)
        
        self.dense1 = nn.Linear(256 * 12 *12, 1000)
        self.drop5 = nn.Dropout(0.5)
        
        self.dense2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(0.6)
        
        self.dense3 = nn.Linear(1000, 136)

        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = f.elu(self.conv1(x))   # (32, 220, 220)
        x = self.max_pool(x)       # (32, 110, 110)
        x = self.drop1(x)          # (32, 110, 110)
        
        x = f.elu(self.conv2(x))   # (64, 107, 107)
        x = self.max_pool(x)       # (64, 53, 53)
        x = self.drop2(x)          # (64, 53, 53)
        
        x = f.elu(self.conv3(x))   # (128, 51, 51)
        x = self.max_pool(x)       # (128, 25, 25)
        x = self.drop3(x)          # (128, 25, 25)

        x = f.elu(self.conv4(x))   # (256, 24, 24)
        x = self.max_pool(x)       # (256, 12, 12)
        x = self.drop4(x)          # (256, 12, 12)

        # Flatten layer
        x = x.view(x.size(0), -1)  # 256 * 12 *12
        
        x = f.elu(self.dense1(x))  # 1000
        x = self.drop5(x)

        x = f.relu(self.dense2(x)) # 1000
        x = self.drop6(x)

        x = self.dense3(x)         # 136
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
