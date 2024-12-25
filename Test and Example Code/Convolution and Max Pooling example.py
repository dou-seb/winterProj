import torch
import torch.nn as nn

# Step 1: Define a low-resolution grayscale image
image = torch.tensor([[
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 2, 0],
    [2, 3, 0, 1, 2],
    [0, 1, 2, 3, 0]
]], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 1, 5, 5)

# Step 2: Define a convolutional layer
conv_layer = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
conv_layer.weight.data = torch.tensor([[[[-1, -1, -1],
                                         [ 0,  0,  0],
                                         [ 1,  1,  1]]]], dtype=torch.float32)
conv_layer.bias.data.zero_()

# Apply the convolution
conv_output = conv_layer(image)

# Step 3: Apply max pooling
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
pooled_output = pool_layer(conv_output)

# Display results
print("Original Image Tensor:")
print(image)
print("\nConvolution Output:")
print(conv_output)
print("\nMax Pooled Output:")
print(pooled_output)
