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
batch_size = 64
num_epochs = 50
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
    num_workers=2
)

# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State: 256 x 4 x 4
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State: 128 x 8 x 8
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State: 64 x 16 x 16
            
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias=False),
            nn.Tanh()
            # Output: 1 x 28 x 28
        )
    
    def forward(self, x):
        return self.main(x)

# Fixed Discriminator Network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),            # Output: 64 x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),          # Output: 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),         # Output: 256 x 3 x 3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),           # Output: 1 x 1 x 1
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.main(x)
        return output.view(output.size(0), -1)  # Flatten to [batch_size, 1]

# Initialize networks
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Training function
def train_gan():
    print("Starting GAN training...")
    
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Labels - now just [batch_size, 1] to match discriminator output
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real images
            real_outputs = discriminator(real_images)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
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
        
        # Save sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_sample_images(epoch + 1)
    
    # Save the trained models
    torch.save(generator.state_dict(), 'models/generator.pth')
    torch.save(discriminator.state_dict(), 'models/discriminator.pth')
    print("Training completed! Models saved.")

def save_sample_images(epoch):
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, latent_dim, 1, 1).to(device)
        fake_images = generator(noise)
        fake_images = fake_images.cpu().detach()
        
        # Denormalize images
        fake_images = (fake_images + 1) / 2.0
        
        # Save grid of images
        grid = torchvision.utils.make_grid(fake_images, nrow=4, padding=2, normalize=False)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray')
        plt.title(f'Generated Images - Epoch {epoch}')
        plt.axis('off')
        plt.savefig(f'generated_samples/epoch_{epoch}.png')
        plt.close()
    generator.train()

# Conditional Generator for specific digits
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim, num_classes=10):
        super(ConditionalGenerator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        self.main = nn.Sequential(
            # Input: (latent_dim + label_embedding) x 1 x 1
            nn.ConvTranspose2d(latent_dim * 2, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 1, 4, 2, 3, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels)
        x = torch.cat([noise, label_embedding], 1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.main(x)

# Fixed Conditional Discriminator
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(ConditionalDiscriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, 28*28)
        
        self.main = nn.Sequential(
            # Input: 2 x 28 x 28 (image + label embedding reshaped)
            nn.Conv2d(2, 64, 4, 2, 1, bias=False),            # Output: 64 x 14 x 14
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),          # Output: 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),         # Output: 256 x 3 x 3
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 3, 1, 0, bias=False),           # Output: 1 x 1 x 1
            nn.Sigmoid()
        )
    
    def forward(self, images, labels):
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(label_embedding.size(0), 1, 28, 28)
        
        x = torch.cat([images, label_embedding], 1)
        output = self.main(x)
        return output.view(output.size(0), -1)

def train_conditional_gan():
    print("Training Conditional GAN for specific digit generation...")
    
    # Initialize conditional generator and discriminator
    c_generator = ConditionalGenerator(latent_dim).to(device)
    c_discriminator = ConditionalDiscriminator().to(device)
    
    c_g_optimizer = optim.Adam(c_generator.parameters(), lr=lr, betas=(beta1, 0.999))
    c_d_optimizer = optim.Adam(c_discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    for epoch in range(num_epochs):
        for i, (real_images, labels) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            # Labels for discriminator
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            c_d_optimizer.zero_grad()
            
            # Real images
            real_outputs = c_discriminator(real_images, labels)
            d_loss_real = criterion(real_outputs, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_images = c_generator(noise, labels)
            fake_outputs = c_discriminator(fake_images.detach(), labels)
            d_loss_fake = criterion(fake_outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            c_d_optimizer.step()
            
            # Train Generator
            c_g_optimizer.zero_grad()
            
            fake_outputs = c_discriminator(fake_images, labels)
            g_loss = criterion(fake_outputs, real_labels)
            
            g_loss.backward()
            c_g_optimizer.step()
            
            if i % 100 == 0:
                print(f'Conditional Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}')
        
        # Save sample images for each digit
        if (epoch + 1) % 10 == 0:
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

def generate_digit_samples(digit, num_samples=5):
    """Generate samples for a specific digit using the trained conditional generator"""
    # Load the trained conditional generator
    c_generator = ConditionalGenerator(latent_dim).to(device)
    c_generator.load_state_dict(torch.load('models/conditional_generator.pth', map_location=device))
    c_generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        fake_images = c_generator(noise, labels)
        fake_images = fake_images.cpu().detach()
        
        # Denormalize images
        fake_images = (fake_images + 1) / 2.0
        
        return fake_images

if __name__ == "__main__":
    print("Starting MNIST Digit Generation Training...")
    
    # Train basic GAN first
    train_gan()
    
    # Train conditional GAN for specific digit generation
    train_conditional_gan()
    
    print("All training completed!")
    
    # Test generating samples for digit 7
    print("\nTesting digit generation...")
    try:
        samples = generate_digit_samples(7, 5)
        print(f"Successfully generated {samples.shape[0]} samples for digit 7")
        
        # Display the samples
        fig, axes = plt.subplots(1, 5, figsize=(10, 2))
        for i in range(5):
            axes[i].imshow(samples[i].squeeze(), cmap='gray')
            axes[i].set_title(f'Sample {i+1}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig('generated_samples/test_digit_7.png')
        plt.show()
        
    except FileNotFoundError:
        print("Model not found. Please train the model first.")