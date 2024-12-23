import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image

img_path = r'C:\Users\Sebastien Douse\Documents\GitHub\Winter-Mini-Project\cats-in-the-wild-image-classification\versions\1\train\AFRICAN LEOPARD\001.jpg'
image = Image.open(img_path)
transform = transforms.ToTensor()
ImgTensor = transform(image)
print(ImgTensor.shape)