import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 100
batch_size = 32  # Reduced batch size
num_epochs = 20  # Reduced epochs for testing
lr = 0.0002
beta1 = 0.5

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('generated_samples', exist_ok=True)

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=0  # Reduced workers
)

# Simple Generator Network
class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(SimpleGenerator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)  # 28*28 = 784
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output between -1 and 1
        return x.view(-1, 1, 28, 28)

# Simple Discriminator Network
class SimpleDiscriminator(nn.Module):
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.fc1 = nn.Linear(784, 512)  # 28*28 = 784
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Conditional Generator
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        self.fc1 = nn.Linear(latent_dim * 2, 256)  # noise + label
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)
        
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([noise, label_embedding], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x.view(-1, 1, 28, 28)

# Initialize networks
generator = SimpleGenerator(latent_dim).to(device)
discriminator = SimpleDiscriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

def train_gan():
    print("Starting GAN training...")
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = generator(noise)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)
            
            g_loss.backward()
            g_optimizer.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_sample_images(epoch + 1)
    
    # Save the trained models
    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
    print("Training completed! Models saved.")

def save_sample_images(epoch):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, latent_dim).to(device)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().detach()
        
        # Denormalize images
        fake_images = (fake_images + 1) / 2.0
        
        # Save grid of images
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_images[i].squeeze(), cmap='gray')
            ax.axis('off')
        
        plt.suptitle(f'Generated Images - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'generated_samples/epoch_{epoch}.png')
        plt.close()
    generator.train()

def train_conditional_gan():
    print("Training Conditional GAN for specific digit generation...")
    
    # Initialize conditional generator
    c_generator = ConditionalGenerator(latent_dim).to(device)
    c_optimizer = optim.Adam(c_generator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Train Generator
            c_optimizer.zero_grad()
            
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = c_generator(noise, labels)
            
            # Use discriminator to evaluate fake images
            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, torch.ones(batch_size, 1).to(device))
            
            g_loss.backward()
            c_optimizer.step()
            
            if i % 100 == 0:
                print(f'Conditional Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'G_loss: {g_loss.item():.4f}')
        
        # Save sample images for each digit
        if (epoch + 1) % 5 == 0:
            save_conditional_samples(c_generator, epoch + 1)
    
    # Save the conditional generator
    torch.save(c_generator.state_dict(), 'models/conditional_generator.pth')
    print("Conditional GAN training completed!")

def save_conditional_samples(generator, epoch):
    generator.eval()
    with torch.no_grad():
        for digit in range(10):
            noise = torch.randn(5, latent_dim).to(device)
            labels = torch.full((5,), digit, dtype=torch.long).to(device)
            fake_images = generator(noise, labels)
            fake_images = fake_images.cpu().detach()
            
            # Denormalize images
            fake_images = (fake_images + 1) / 2.0
            
            # Save individual images
            for j in range(5):
                plt.figure(figsize=(2, 2))
                plt.imshow(fake_images[j].squeeze(), cmap='gray')
                plt.title(f'Digit {digit}')
                plt.axis('off')
                plt.savefig(f'generated_samples/digit_{digit}_sample_{j}_epoch_{epoch}.png')
                plt.close()
    generator.train()

if __name__ == "__main__":
    print("Starting MNIST Digit Generation Training (Simplified)...")
    
    # Train basic GAN first
    train_gan()
    
    # Train conditional GAN for specific digit generation
    train_conditional_gan()
    
    print("All training completed!") 