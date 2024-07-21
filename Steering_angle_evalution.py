import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import numpy as np
import time
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()

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

        output = torch.cat(pooled_outputs, dim=-1)
        output = output.view(batch_size, -1)

        return output

# Define the backbone model (ResNet or MobileNet)
backbone = models.resnet18(pretrained=True)
# backbone = models.mobilenet_v2(pretrained=True)

# Remove the fully connected layer at the end
backbone = nn.Sequential(*list(backbone.children())[:-1])
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

    def forward(self, x):
        x = self.backbone(x)
        x = self.spp(x)
        x = torch.relu(self.fc1(x))
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
        angle = angle * 3.14159265 / 180
        if self.transform:
            image = self.transform(image)
        return image, angle

    def read_data_file(self):
        data_file = os.path.join(self.root_dir, 'data.txt')
        with open(data_file, 'r') as file:
            lines = file.readlines()
            data = [(line.split()[0], float(line.split()[1])) for line in lines]
        return data

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize input images to match backbone input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Define hyperparameters
input_size = 512  # Output feature size of ResNet18
hidden_size = 256
output_size = 1
learning_rate = 0.001
batch_size = 32
num_epochs = 20

# Create dataset and data loaders
dataset = SteeringDataset(root_dir='Test_Images', transform=transform)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# Create model
model = SteeringCNN(backbone, input_channels, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load pre-trained model
model_path = 'steering_model.pth'
if not torch.cuda.is_available():
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluate model
test_loss = 0.0
mse = nn.MSELoss()
mae = nn.L1Loss()
def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_residual / ss_total
    return abs(r2-int(r2))

y_true = []
y_pred = []

with torch.no_grad():
    for images, angles in tqdm(val_loader, desc='Evaluating', leave=False):
        images = images.to(device)
        angles = angles.to(device)

        # Forward pass
        outputs = model(images)
        y_true.extend(angles.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())
        loss = criterion(outputs, angles.unsqueeze(1).float())

        test_loss += loss.item()

y_true = torch.tensor(y_true)
y_pred = torch.tensor(y_pred)
print(y_true,y_pred)
mse_value = mse(y_true, y_pred)
mae_value = mae(y_true, y_pred)
r2_value = r2_score(y_true, y_pred)

print(f"Test Loss: {test_loss / len(val_loader):.4f}")
print(f'MSE: {mse_value.item()}')
print(f'MAE: {mae_value.item()}')
print(f'R² Score: {r2_value.item()}')

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")