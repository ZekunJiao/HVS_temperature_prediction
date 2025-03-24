import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split

from dataset import TemperatureDataset
from model import SimpleCNN, ComplexCNN, GlobalDilatedCNN
import os
from datetime import datetime


from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
writer = SummaryWriter(f"runs/{timestamp}")

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
        writer.add_scalar("Loss/train", epoch_loss, epoch)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.6f}")

        # Log training & test loss to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
    writer.close()


def visualize_predictions(model, test_dataset, num_samples=5, device='cpu', save_folder="result",  filename='predictions.png'):
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    for i in range(min(num_samples, len(test_dataset))):
        input_sample, target_sample = test_dataset[i]
        with torch.no_grad():
            prediction = model(input_sample.unsqueeze(0).to(device)).cpu().squeeze()

        # Select row of subplots
        ax1, ax2, ax3 = axes[i] if num_samples > 1 else axes  # Handle single sample case

        ax1.set_title("Partial Input")
        im1 = ax1.imshow(input_sample[0].cpu().numpy().T, cmap='viridis', origin="lower")
        fig.colorbar(im1, ax=ax1)

        ax2.set_title("Ground Truth")
        im2 = ax2.imshow(target_sample.squeeze().cpu().numpy().T, cmap='viridis', origin="lower")
        fig.colorbar(im2, ax=ax2)

        ax3.set_title("Prediction")
        im3 = ax3.imshow(prediction.T, cmap='viridis', origin="lower")
        fig.colorbar(im3, ax=ax3)

    plt.tight_layout()

    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    print(f"Figure saved to {save_path}")
    plt.show()


def main():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script folder
    os.chdir(script_dir)  # Set script directory as working directory
    data_file_name = "temperature_data.pt"
    save_path = os.path.join(script_dir, "datasets", data_file_name)

    # Create dataset
    if os.path.exists(save_path):
        print(f" ############## DATASET: {data_file_name} ##################")
        dataset = torch.load(save_path, weights_only=False)
        print("Dataset size", len(dataset))

    # Define the split ratio (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Randomly split the dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    subset_indices = list(range(32))
    train_dataset = Subset(dataset, subset_indices)
    test_dataset = Subset(test_dataset, subset_indices)

    # Create dataloaders for the training and testing sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Set up model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: ", device)
    model = GlobalDilatedCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, epochs=1000, device=device)

    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    # Visualize predictions
    filename = f"{timestamp}.png"
    visualize_predictions(model, test_dataset, num_samples=10, device=device, filename=filename)

if __name__ == "__main__":
    main()
