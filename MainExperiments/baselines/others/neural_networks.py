import torch.nn as nn
import torch

import random
import numpy as np

device = 'cuda'


class PlainLeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_seed = rand# Instantiate the given model configuration, here using VGG16

    # If model_name is not in cfgs, raise AssertionError with parameter content Instantiate the given configuration model, here using VGG16
# **kwargs represents variable length dictionary parameters passed when calling VGG function
    def make_vgg(self, model_name, feature_map_dim,**kwargs):
    # If model_name not in cfgs, will raise AssertionError with parameter contentandint(0,5000)
        torch.manual_seed(self.rand_seed)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, stride=1, padding=3 // 2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # kernel_size, stride
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=3 // 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 10)
        )
        self.conv_list = [self.conv1, self.conv2, self.conv3]
        self.fc_list = [self.fc1, self.fc2, self.fc3]
        self.model_list = [self.conv1, self.conv2, self.conv3, self.fc1, self.fc2, self.fc3]

    def forward(self, input_x):
        out = input_x
        for conv in self.conv_list:
            out = conv(out)
        out = out.view(input_x.shape[0], -1)
        for fc in self.fc_list:
            out = fc(out)
        return nn.functional.softmax(out)


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(500)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=1,padding=3//2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(256, 256, 3, stride=1,padding=3//2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, 3, stride=1, padding=3//2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )
        self.model_list = [0,1,2,3,4,5]

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return nn.functional.softmax(output)

    def save_model(self):
        torch.save(self.conv, "CIFAR-10-conv.plt")
        torch.save(self.fc, "CIFAR-10-fc.plt")

    def load_model(self):
        self.conv = torch.load("CIFAR-10-conv.plt")
        self.fc = torch.load("CIFAR-10-fc.plt")
        self.conv.to(device)
        self.fc.to(device)

# The following is the definition of VGG network
class VGG(nn.Module):
    # init(): Initialize and declare definitions of model layers
    # features: make_features(cfg: list) generates feature extraction network structure
    # num_classes: Number of classification categories
    # init_weights: Whether to initialize network weights
    def __init__(self, features, num_classes=100, init_weights=False, feature_map_dim=7*7):
        # super: Initialize child class using parent class's initialization method
        super(VGG, self).__init__()
        # Generate feature extraction network structure
        self.features = features
        # Generate classification network structure
        # Sequential: Custom sequential connection of layers to form network structure
        self.classifier = nn.Sequential(
            # Dropout: Randomly set 50% of input neuron activations to 0, removing some nodes to prevent overfitting
            nn.Dropout(p=0.5),
            nn.Linear(512*feature_map_dim, 4096), # Number of features after feature extraction
            # ReLU(inplace=True): Modify tensor directly without intermediate variables, saving memory
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        # If True, initialize network parameters
        if init_weights:
            self._initialize_weights()
 
    # forward(): Define forward propagation process, describing connections between layers
    def forward(self, x):
        # Input data to feature extraction network structure, N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        # After image goes through feature extraction network, we get a 7*7*512 feature matrix, then flatten it
        # Flatten(): Flattens tensor (multidimensional array), dim 0 is batch_size, so Flatten() starts from dim 1
        x = torch.flatten(x, start_dim=1)
        # Input data to classification network structure, N x 512*7*7
        x = self.classifier(x)
        return nn.functional.softmax(x)
 
    # Initialize network structure parameters
    def _initialize_weights(self):
        # Iterate through each layer in the network
        # Use method from nn.Module class: self.modules(), which returns all modules in the network
        for m in self.modules():
            # isinstance(object, type): Returns True if object is of specified type
            # If it's a convolutional layer
            if isinstance(m, nn.Conv2d):
                # uniform_(tensor, a=0, b=1): Initialize with uniform distribution U(a,b)
                nn.init.xavier_uniform_(m.weight)
                # If bias exists, initialize it to zero
                if m.bias is not None:
                    # constant_(tensor, val): Initialize entire matrix as constant val
                    nn.init.constant_(m.bias, 0)
            # If it's a fully connected layer
            elif isinstance(m, nn.Linear):
                # Initialize with normal distribution
                nn.init.xavier_uniform_(m.weight)
                # Set all biases to 0
                nn.init.constant_(m.bias, 0)
 
 
# Generate feature extraction network structure
# Parameter is network configuration variable, passed as a list
def make_features(cfg: list):
    # Define empty list to store each layer structure
    layers = []
    # Input image is RGB color image
    in_channels = 3
    # Loop through configuration list to get a list of convolutional and pooling layers
    for v in cfg:
        # If list value is 'M', it indicates a max pooling layer
        if v == "M":
            # Create max pooling layer, in VGG all max pooling layers have kernel_size=2, stride=2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # Otherwise it's a convolutional layer
        else:
            # in_channels: input feature matrix depth, v: output feature matrix depth (number of filters)
            # In VGG, all conv layers have padding=1, stride=1
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=1, padding=1)
            # Add conv layer and ReLU to list
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v        # Return list as non-keyword arguments, *layers can accept any number of parameters
    return nn.Sequential(*layers)
 
 
# Define cfgs dictionary, each key represents a model configuration, e.g., VGG11 represents config A, an 11-layer network
# Numbers represent number of filters in conv layers, 'M' represents pooling layer structure
# Generate feature extraction network structure using make_features(cfg: list) function
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
 
 
# Instantiate the given configuration model. Here, VGG16 is used.
# **kwargs represents a dictionary variable of variable length, which is passed as a dictionary variable when calling the VGG function.
def make_vgg(model_name, feature_map_dim,**kwargs):
    # If `model_name` is not in `cfgs`, the program will throw an `AssertionError`, and the error message will be "The parameter content is ' '."
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)        # Get the list corresponding to VGG16
    cfg = cfgs[model_name]
    # Instantiate VGG network
    # This dictionary contains number of classes and boolean for weight initialization
    model = VGG(make_features(cfg), feature_map_dim=feature_map_dim, **kwargs)
    return model


class LeNet4ImgNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(500)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=1,padding=3//2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(256, 256, 3, stride=1,padding=3//2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, 3, stride=1, padding=3//2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
        self.model_list = [0,1,2,3,4,5]

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return nn.functional.softmax(output)

    def save_model(self):
        torch.save(self.conv, "CIFAR-10-conv.plt")
        torch.save(self.fc, "CIFAR-10-fc.plt")

    def load_model(self):
        self.conv = torch.load("CIFAR-10-conv.plt")
        self.fc = torch.load("CIFAR-10-fc.plt")
        self.conv.to(device)
        self.fc.to(device)

class ReLuNeuralNetWork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        #self.flatten = nn.Flatten()
        self.classification = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Sigmoid(),
            nn.Softmax()
        )
        self.model_list = ['0', '2', '4']
        

    def set_weight(self, level, weight_outspace, bias_outspace):
        self.classification._modules[self.model_list[level]].weight.data = nn.Parameter(weight_outspace)
        self.classification._modules[self.model_list[level]].bias.data = nn.Parameter(bias_outspace)

    def get_weight(self, level):
        #print(self.classification._modules['classification']._modules[self.model_list[level]].weight)
        return self.classification._modules['classification']._modules[self.model_list[level]].weight.data.numpy(), self.classification._modules['classification']._modules[self.model_list[level]].bias.data.numpy()

    def save_model(self):
        torch.save(self.classification,"NumberMNIST.plt")
    
    def load_model(self):
        self.classification = torch.load("NumberMNIST.plt")
        self.classification.to(device)

    def forward(self, x:torch.Tensor):
        x = x.view(x.shape[0], -1)
        logits = self.classification(x)
        return logits


class BasicBlock(nn.Module):  # 2 conv layers, F(X) and X have same dimensions
    # expansion is the multiplier of F(X) dimension relative to X
    expansion = 1  # Whether residual mapping F(X) dimension changes, 1 means no change, downsample=None

    # in_channel: input feature matrix depth (image channels, e.g., RGB has 3 channels)
    # out_channel: output feature matrix depth (number of conv filters)
    # stride: convolution stride
    # downsample: used to make residual data and conv data shapes match for addition
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)  # BN layer between conv and relu layers

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # out=F(X)+X
        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):  # 3 conv layers, F(X) and X dimensions are different
    """
    Note: In the original paper, in the dotted residual structure's main branch, 
    the first 1x1 conv layer has stride 2, and the second 3x3 conv layer has stride 1.
    However, in PyTorch's official implementation, the first 1x1 conv layer has stride 1,
    and the second 3x3 conv layer has stride 2. This improves top1 accuracy by about 0.5%.
    """
    # expansion is the multiplier for F(X) dimension relative to X dimension
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups
        # Here width equals out_channel
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=width,kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=width, out_channels=width, groups=groups,kernel_size=3, stride=stride, bias=False, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=width, out_channels=out_channel * self.expansion,kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),
        )
        self.downsample = downsample
        self.relu_last = nn.ReLU(inplace=True)

        


    def forward(self, x):
        identity = x
        # downsample is used to make residual data and conv data shapes match for direct addition
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.forw(x)
        # out=F(X)+X
        out += identity
        out = self.relu_last(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,  # Type of residual block to use
                 blocks_num,  # Number of residual blocks for each conv layer
                 num_classes=1000,  # Number of classes in training set
                 include_top=True,  # Whether to add pooling, fc, softmax after residual structure
                 groups=1,
                 width_per_group=64):

        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # Depth of first conv layer output feature matrix, also input depth for subsequent layers

        self.groups = groups
        self.width_per_group = width_per_group

        # Input layer has RGB 3 components, making input feature matrix depth 3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # _make_layer(block type, number of filters in first conv layer, number of blocks, stride): Generate multiple consecutive residual blocks
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:  # Default True, add pooling, fc, softmax
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive average pooling, output size will be 1x1 regardless of input shape
            # Flatten matrix to vector, e.g. (W,H,C)->(1,1,W*H*C), depth becomes W*H*C
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # Fully connected layer, input depth is 512 * block.expansion, output is num_classes

        for m in self.modules():  # initialization
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # _make_layer() function: Generate multiple consecutive residual blocks (block type, number of filters in first conv layer, number of blocks, stride)
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None

        # Find: When stride != 1 or depth expansion changes, causing F(X) and X shapes to differ, define downsampling for X to match shapes
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        # layers stores consecutive residual blocks in sequence
        # In each residual structure, first block needs X downsampling, subsequent blocks don't
        layers = []
        # Add first residual block, which needs X downsampling
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion
        # Subsequent residual blocks don't need X downsampling
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        # Pass layers list as non-keyword arguments to Sequential(), connecting residual blocks into one structure
        return nn.Sequential(*layers)
    
    def forward(self, x):
        forward_list = []
        x = self.conv1(x)
        forward_list.append(x)
        x = self.bn1(x)
        forward_list.append(x)
        x = self.relu(x)
        forward_list.append(x)
        x = self.maxpool(x)
        forward_list.append(x)

        x = self.layer1(x)
        forward_list.append(x)
        x = self.layer2(x)
        forward_list.append(x)
        x = self.layer3(x)
        forward_list.append(x)
        x = self.layer4(x)
        forward_list.append(x)

        if self.include_top:  # Usually True
            x = self.avgpool(x)
            forward_list.append(x)
            x = torch.flatten(x, 1)
            forward_list.append(x)
            x = self.fc(x)
            forward_list.append(x)

        return x, forward_list



def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    #torch.manual_seed(500)
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

