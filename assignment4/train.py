import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import get_loaders

# Reproducibility & Device Setup
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001

train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

# ==========================================
# 1. Regular CNN
# ==========================================
class RegularCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# ==========================================
# 2. Small U-Net
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.block(x)

class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(3, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(32, 64)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32, 16)
        
        self.out_conv = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        u1 = self.up1(b)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)
        
        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)
        
        return self.out_conv(d2)

# ==========================================
# Training Function
# ==========================================
def train_model(model, criterion, optimizer, target_type='labels', epochs=EPOCHS):
    model.to(device)
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        
        for images, targets in train_loader:
            images = images.to(device)
            if target_type == 'labels':
                targets = targets[0].to(device) # labels
            else:
                targets = targets[1].to(device) # masks
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            
        epoch_loss = running_loss / total_samples
        loss_history.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
        
    return model, loss_history

if __name__ == "__main__":
    print("\n--- Training Regular CNN ---")
    cnn_model = RegularCNN()
    criterion_cls = nn.CrossEntropyLoss()
    optimizer_cnn = optim.Adam(cnn_model.parameters(), lr=LR)
    cnn_model, cnn_losses = train_model(cnn_model, criterion_cls, optimizer_cnn, target_type='labels')
    torch.save(cnn_model.state_dict(), "cnn.pt")

    print("\n--- Training ResNet-18 ---")
    import torchvision.models as models
    resnet_model = models.resnet18(weights=None)
    resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
    optimizer_resnet = optim.Adam(resnet_model.parameters(), lr=LR)
    resnet_model, resnet_losses = train_model(resnet_model, criterion_cls, optimizer_resnet, target_type='labels')
    torch.save(resnet_model.state_dict(), "resnet18.pt")

    print("\n--- Training Small U-Net ---")
    unet_model = SmallUNet()
    optimizer_unet = optim.Adam(unet_model.parameters(), lr=LR)
    unet_model, unet_losses = train_model(unet_model, criterion_cls, optimizer_unet, target_type='masks')
    torch.save(unet_model.state_dict(), "unet.pt")
    
    print("Training complete. Models saved locally.")
