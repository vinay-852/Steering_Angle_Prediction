import torch
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_list):
        super(SpatialPyramidPooling, self).__init__()
        self.pool_list = pool_list

    def forward(self, x):
        batch_size, num_channels, _, _ = x.size()
        pooled_outputs = []

        for pool_size in self.pool_list:
            pooled = F.adaptive_max_pool2d(x, output_size=(pool_size, pool_size))
            pooled = pooled.view(batch_size, num_channels, -1)
            pooled_outputs.append(pooled)

        output = torch.cat(pooled_outputs, dim=2)
        output = output.view(batch_size, -1)

        return output

# Define the backbone model (ResNet)
backbone = models.resnet18(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-2])
input_channels = 512
backbone = backbone.to(device)

# Define the basic CNN architecture to predict steering angle
class SteeringCNN(nn.Module):
    def __init__(self, backbone, input_shape, hidden_size, output_size):
        super(SteeringCNN, self).__init__()
        self.backbone = backbone
        self.spp = SpatialPyramidPooling(pool_list=[1, 2, 4])
        self.fc1 = nn.Linear(input_shape * (1 + 4 + 16), hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)  # Add dropout layer

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

class SteeringDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.read_data_file()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, angle = self.data[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)
        
        # Normalize the angle to [-1, 1] range
        angle = ((angle + 450) % 900) - 450
        normalized_angle = angle / 450  # Normalize to [-1, 1]
        normalized_angle = normalized_angle * np.pi  # Convert to radians for the model

        if self.transform:
            image = self.transform(image)
        return image, normalized_angle

    def read_data_file(self):
        data_file = os.path.join(self.root_dir, 'data.txt')
        with open(data_file, 'r') as file:
            lines = file.readlines()
            data = [(line.split()[0], float(line.split()[1])) for line in lines]
        return data

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define hyperparameters
input_size = 512
hidden_size = 256
output_size = 1
learning_rate = 0.001  # Adjusted learning rate
batch_size = 32
num_epochs = 70

# Create dataset and dataloaders
dataset = SteeringDataset(root_dir=r"/kaggle/input/driving-dataset/driving_dataset", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Create models
model = SteeringCNN(backbone, input_channels, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Add learning rate scheduler

# Initialize the list to store training losses
training_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for images, angles in tqdm(train_dataloader, desc=f'Epoch {epoch+1}', leave=False):
        images = images.to(device)
        angles = angles.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, angles.float())
        train_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Calculate average training loss
    train_loss /= len(train_dataloader)
    training_losses.append(train_loss)
    
    # Debug prints
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, angles in tqdm(val_dataloader, desc=f'Validation Epoch {epoch+1}', leave=False):
            images = images.to(device)
            angles = angles.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, angles.float())
            val_loss += loss.item()
    
    # Calculate average validation loss
    val_loss /= len(val_dataloader)
    
    # Step the learning rate scheduler
    scheduler.step()
    
    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}')
    print(f'Outputs: {outputs[:5].view(-1).cpu().numpy()}')  # Print first 5 outputs for inspection
    print(f'Angles: {angles[:5].view(-1).cpu().numpy()}')    # Print first 5 angles for inspection
torch.save(model.state_dict(), 'steering_model.pth')
# Testing
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, angles in tqdm(test_dataloader,desc=f'Epoch {epoch+1}', leave=False):
        images = images.to(device)
        angles = angles.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, angles.unsqueeze(1).float())

        test_loss += loss.item()
print(f"TestLoss: {test_loss:.4f}")