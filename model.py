# TRAINING A CNN


# TYPICAL CNN ARCHITECTURE
# NOTE: Pooling layers are used to automatically learn certain features on the input images
# NOTE: When defining the convolutional layers, its important to properly define the padding, 
# or else the filter wont be able to parse through all the available regions in the image
# NOTE: Max pooling is used to down sample images by applying a maximum filter to regions.
# For example, a 2x2 max-pool will apply a maximum filter to a 2x2 region. Reduces computational cost
# reduces overfitting by providing an abstracted version of the input data


import torch.nn as nn
import torch

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st param: num input channels (leave as 3) 
        # 2nd param: num output channels generally we want to keep this number small in the early layers of the model
        # around 32 to 64 (to prevent overfitting)
        # 3rd param: kernel size - usually, we want smaller kernel size (3x3) in early layers to capture fine details
        # we can move up in kernel size in deeper layers to capture global details

        # a larger kernel size means a more aggressive downsampling of the input dimensions
        # a smaller kernel size perserves the more fine grained details of the input feature map
        # in the early layers, you want to capture the fine details so keep kernel size small

        # stride influences the downsampling factor
        # larger stride means a more aggressive of the input spacial dimensions
        # if you want to reduce spacial dimensions more rapidly, increase stride
        # larger strides introduce overlap between neighboring regions
        # the choice of stride impacts how much info is retained from adjacent regions
        # in early layers, smaller strides are more common as we want to capture fine details
        # in deeper layers, we will increase stride to get a general idea
        # output size of max pooling layer - (input size - kernel size)/stride + 1

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=1)
        self.relu5 = nn.ReLU()


        self.pool = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1681, 64)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu7 = nn.ReLU()

        # Output layers
        self.steering_output = nn.Linear(64, 1)
        self.throttle_output = nn.Linear(64, 1)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x) 
        x = self.relu3(x)
        x = self.pool(x) # no memory D: 
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)

        # Output layers
        steering_output = self.steering_output(x)
        throttle_output = self.throttle_output(x)

        return steering_output, throttle_output


class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1682, 64)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu7 = nn.ReLU()

        # Output layers
        self.steering_output = nn.Linear(64, 1)
        self.throttle_output = nn.Linear(64, 1)

    def forward(self, state, action):
        # Convolutional layers
        x = self.conv1(state)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x) 
        x = self.relu3(x)
        x = self.pool(x) # no memory D: 
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        # Fully connected layers 
        x = self.flatten(x)
        x = torch.cat([x, action], dim=1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)

        # Output layers
        steering_output = self.steering_output(x)
        throttle_output = self.throttle_output(x)

        return steering_output, throttle_output
