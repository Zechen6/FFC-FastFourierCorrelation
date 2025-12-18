"""
Implementation of Integrated Gradients (IG) using Captum library for model interpretability.
This script applies IG attribution method on ResNet50 and ViT models using ImageNet validation dataset.
"""

import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import torchvision.models as md
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import matplotlib
matplotlib.use('agg')  # Set matplotlib backend to 'agg' for non-interactive mode
import matplotlib.pyplot as plt
from torchvision.models import ResNet152_Weights, DenseNet201_Weights, Inception_V3_Weights, ViT_B_32_Weights,ResNet50_Weights
from torchvision.models import resnet152, densenet201, inception_v3, vit_b_32,resnet50

# Utility transforms for tensor conversion
To_tensor = transforms.ToTensor()
To_image = transforms.ToPILImage()
device = "cuda:3"  # Set GPU device

# Define image transformations for preprocessing
data_transform = transforms.Compose([
        transforms.Resize(224),          # Resize images to 224x224
        transforms.CenterCrop(224),      # Center crop to ensure consistent size
        transforms.ToTensor(),           # Convert to tensor
        transforms.Normalize(            # Normalize with ImageNet mean and std
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

# Load ImageNet validation dataset
val_dataset = datasets.ImageFolder(
    root='/data01/img_net_dataset/val',
    transform=data_transform)

# Create data loader for batch processing
img_loader = DataLoader(val_dataset, batch_size=16)

# Iterate through different models (ResNet50 and ViT)
for m in ['Resnet50','ViT']:
    print(m,"is explaining")
    # Initialize the selected model with pre-trained weights
    if m == "ViT":
        model = vit_b_32(weights=ViT_B_32_Weights.DEFAULT)
    else:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)    
    model.to(device)  # Move model to GPU
    
    # Initialize Integrated Gradients attributor
    ig = IntegratedGradients(model)
    time_sum = 0

    # Process each batch of images
    for batch,(X, y) in enumerate(img_loader):
        # Prepare input and target
        input1 = X.to(device)
        y = y.to(device)
        input1.requires_grad = True  # Enable gradient computation
        
        # Create zero baseline for attribution
        baseline = torch.zeros_like(input1, dtype=torch.float32).to(device)

        # Calculate attributions using Integrated Gradients
        start_time = time.time()
        attributions, approximation_error = ig.attribute(input1,
                                                        baselines=baseline,
                                                        target=y,
                                                        method='gausslegendre',
                                                        return_convergence_delta=True)
        
        # Save attribution results
        torch.save(attributions, "/data01/lzc/InterpretRes/IG/"+m+"/"+str(batch))
        ed_time = time.time()
        torch.cuda.empty_cache()  # Clear GPU memory
        
        # Calculate and print average processing time (skip first batch)
        if batch == 0:
            continue
        time_sum += ed_time - start_time
        print(time_sum/(batch))  # Print average time per batch

