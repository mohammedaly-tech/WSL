
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_loader import ThyroidDataset
from model import UNet

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 20

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load TG3K and TN3K datasets
train_dataset = ThyroidDataset(image_dir="path_to_tg3k_images", mask_dir="path_to_tg3k_masks")
test_dataset = ThyroidDataset(image_dir="path_to_tn3k_images", mask_dir="path_to_tn3k_masks")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model, loss function, optimizer
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()

            # Optimize weights
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")

# Test the model
def test(model, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    print(f"Test Loss: {total_loss/len(test_loader)}")

# Run training and testing
if __name__ == '__main__':
    train(model, train_loader, criterion, optimizer, EPOCHS)
    test(model, test_loader, criterion)
