# src/train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import os

# Set MLflow experiment
mlflow.set_experiment("FashionMNIST-CNN")

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Data
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# Training Function
def train(model, epochs=5, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

    return model

# Test Function
def evaluate(model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            _, pred = torch.max(output.data, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Main Training with MLflow
with mlflow.start_run(run_name="CNN_Fashion"):
    model = CNN()
    model = train(model, epochs=5, lr=0.001)
    acc = evaluate(model)

    # Log params, metrics, and model
    mlflow.log_param("epochs", 5)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("test_accuracy", acc)

    # Save model
    model_path = "models/cnn_model.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)
