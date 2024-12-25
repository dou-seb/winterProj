import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

##IMPORTANT TASK: CREATING CUSTOM PYTORCH DATASET FOR BIG CAT CLASSIF.

class WildcatDataset(Dataset):
    def __init__(self, csv_file, root_dir, dataset_type, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with image data.
            root_dir (string): Directory with all the images.
            dataset_type (string): Filter for what section of the database is used for what purpose
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file) #gets csv and reads it
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_type = dataset_type  # 'train', 'valid', or 'test'

        # Filter the data based on the 'dataset' column
        self.data_frame = self.data_frame[self.data_frame['data set'] == self.dataset_type]

    def __len__(self):
        return len(self.data_frame) #returns the number of rows in the csv

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1])  # grabs the filepath column from the csv
        image = Image.open(img_name).convert('RGB')  # Load image and converts into RGB values from grabbed filepath
        label = torch.tensor(self.data_frame.iloc[idx, 2], dtype=torch.long)  # gets the label from the label column of the csv
        
        if self.transform:
            image = self.transform(image)

        return image, label



"""
#Receiving a singular image from the training dataset to begin with
#Converts image to tensor
img_path = r'C:\Users\Sebastien Douse\Documents\GitHub\Winter-Mini-Project\cats-in-the-wild-image-classification\versions\1\train\AFRICAN LEOPARD\001.jpg'
image = Image.open(img_path)
transform = transforms.ToTensor()
ImgTensor = transform(image)
print(ImgTensor.shape)
"""

class BigCatModel(nn.Module):
    """
    Args:
        nn.Module (library call)
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(128 * 56 * 56, 512)  # Adjust dimensions for your input
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 10)  # 5 classes for big cats

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.pool3(x)
        x = self.flat(x)
        x = self.act4(self.fc1(x))
        x = self.drop4(x)
        x = self.fc2(x)
        return x

# Define your transformations (e.g., resize, normalization) in case an image or not the correct format
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained models
])

#Create datasets and dataloaders
train_dataset = WildcatDataset(csv_file=r'cats-in-the-wild-image-classification\versions\1\WILDCATS.csv', root_dir=r'cats-in-the-wild-image-classification\versions\1', dataset_type='train', transform=transform)
valid_dataset = WildcatDataset(csv_file=r'cats-in-the-wild-image-classification\versions\1\WILDCATS.csv', root_dir=r'cats-in-the-wild-image-classification\versions\1', dataset_type='valid', transform=transform)
test_dataset = WildcatDataset(csv_file=r'cats-in-the-wild-image-classification\versions\1\WILDCATS.csv', root_dir=r'cats-in-the-wild-image-classification\versions\1', dataset_type='test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

"""
# Example: Getting one batch of images and labels
for images, labels in train_loader:
    print(images.size(), labels.size())
    break
"""
#Creates the model
model = BigCatModel()

# Training loop: Trains the model
criterion = nn.CrossEntropyLoss() #Backpropogation error value calculation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")