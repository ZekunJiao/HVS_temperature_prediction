import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import TemperatureDataset
from model import SimpleCNN

def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

def visualize_predictions(model, test_dataset, num_samples=5, device='cpu'):
    model.eval()
    for i in range(min(num_samples, len(test_dataset))):
        input_sample, target_sample = test_dataset[i]
        with torch.no_grad():
            prediction = model(input_sample.unsqueeze(0).to(device)).cpu().squeeze()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Partial Input")
        plt.imshow(input_sample[0], cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(target_sample.squeeze(), cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(prediction, cmap='viridis')
        plt.colorbar()

        plt.tight_layout()
        plt.show()

def main():
    # Set grid and dataset parameters
    nx, ny = 100, 100
    dx, dy = 0.01, 0.01
    num_simulations = 1000  # Number of different simulation samples

    # Create dataset
    dataset = TemperatureDataset(num_simulations, nx, ny, dx, dy)

    # Define the split ratio (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Randomly split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders for the training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set up model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=50, device=device)

    # Visualize predictions
    visualize_predictions(model, test_dataset, num_samples=5, device=device)

if __name__ == "__main__":
    main()
