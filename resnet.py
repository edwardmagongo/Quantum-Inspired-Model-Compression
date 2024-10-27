# resnet.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import optuna
import time

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Efficient Channel Attention (ECA) Block
class ECA(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(in_channels).float()) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

# Quantum-Inspired ResNet with ECA and Multi-Scale Fusion
class QuantumInspiredResNet(nn.Module):
    def __init__(self):
        super(QuantumInspiredResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)

        # Modify the fully connected layer to fit CIFAR-10 (10 classes)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1280)

        # ECA Block for channel attention
        self.eca_block = ECA(in_channels=1280)

        # Multi-scale feature fusion
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(1280, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1280, kernel_size=3, padding=1),
            ECA(1280)
        )

        # Final classification layer
        self.fc = nn.Linear(1280, 10)  # CIFAR-10 has 10 classes

    def forward(self, x):
        x = self.resnet(x)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Add height and width dims back
        x = self.multi_scale_fusion(x)

        # Global pooling and classifier
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x

# Data preparation (CIFAR-10)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split the train dataset into training and validation sets (80/20 split)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

# QAOA-inspired pruning
def quantum_inspired_pruning(model, pruning_iterations=5, pruning_fraction=0.3):
    """Applies QAOA-inspired pruning to the model."""
    for _ in range(pruning_iterations):
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Calculate importance as the absolute value of weights
                importance = torch.abs(module.weight.data)
                # Convert to probabilities using softmax
                probabilities = torch.softmax(importance.view(-1), dim=0)

                # Determine weights to retain based on probabilities
                retain_mask = (torch.rand(probabilities.size(), device=module.weight.device) < (1 - pruning_fraction)).float()

                # Apply the mask to the weights
                module.weight.data.mul_(retain_mask.view_as(module.weight.data))

                # Entangled pruning: consider neighboring weights
                for i in range(len(retain_mask)):
                    if retain_mask[i] == 0:  # If this weight is pruned
                        # 50% chance to prune left neighbor
                        if i > 0 and torch.rand(1, device=module.weight.device).item() < 0.5:
                            retain_mask[i - 1] = 0
                        # 50% chance to prune right neighbor
                        if i < len(retain_mask) - 1 and torch.rand(1, device=module.weight.device).item() < 0.5:
                            retain_mask[i + 1] = 0

                # Final weight update after considering entanglement
                module.weight.data.mul_(retain_mask.view_as(module.weight.data))

# QCL-inspired tensor decomposition
def tensor_decomposition(layer, rank):
    """Performs tensor decomposition on the layer weights using SVD."""
    weight = layer.weight.data
    weight_shape = weight.shape
    weight_flat = weight.view(weight.size(0), -1)

    # Singular Value Decomposition
    u, s, v = torch.svd(weight_flat)

    # Compress matrices
    u_compressed = u[:, :rank]
    s_compressed = s[:rank]
    v_compressed = v[:, :rank]

    # Reconstruct compressed weight
    compressed_weight = torch.matmul(u_compressed, torch.diag(s_compressed) @ v_compressed.t())
    layer.weight.data = compressed_weight.view(weight_shape)

# Quantum Annealing-inspired matrix factorization
def quantum_annealing_factorization(weights, rank):
    """Applies quantum annealing-inspired matrix factorization to the weights."""
    weight_shape = weights.shape
    weights_flat = weights.view(weights.size(0), -1)

    # Loss function for optimization
    def loss_function(params):
        W1 = params[:weights_flat.size(0) * rank].view(weights_flat.size(0), rank)
        W2 = params[weights_flat.size(0) * rank:].view(rank, weights_flat.size(1))
        approx = torch.matmul(W1, W2)
        return torch.mean((weights_flat - approx) ** 2)

    # Initialize parameters
    W1 = torch.randn(weights_flat.size(0), rank, device=weights.device)
    W2 = torch.randn(rank, weights_flat.size(1), device=weights.device)
    params_init = torch.cat([W1.flatten(), W2.flatten()])
    params = torch.nn.Parameter(params_init, requires_grad=True)
    optimizer = torch.optim.LBFGS([params], max_iter=20)

    # Closure function for optimizer
    def closure():
        optimizer.zero_grad()
        loss = loss_function(params)
        loss.backward()
        return loss

    # Optimize parameters
    optimizer.step(closure)

    # Extract optimized matrices
    W1_optimized = params[:weights_flat.size(0) * rank].view(weights_flat.size(0), rank)
    W2_optimized = params[weights_flat.size(0) * rank:].view(rank, weights_flat.size(1))

    # Reconstruct compressed weights
    compressed_weights = torch.matmul(W1_optimized, W2_optimized).view(weight_shape)
    return compressed_weights

# Apply quantum-inspired methods to the model
def apply_compression_methods(model, layer_sparsity=0.3, rank=10):
    """Applies quantum-inspired compression methods to the model."""
    quantum_inspired_pruning(model, pruning_iterations=5, pruning_fraction=layer_sparsity)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            tensor_decomposition(module, rank)
            compressed_weights = quantum_annealing_factorization(module.weight.data, rank)
            module.weight.data = compressed_weights
            
# Training function
def train_model(model, dataloader, optimizer, criterion, scaler, scheduler=None):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# Evaluation function
def evaluate_model(model, dataloader, criterion):
    model.eval()
    correct, total = 0, 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += ()
    accuracy = 100 * correct / total
    return accuracy, epoch_loss

# Optuna objective function
def objective(trial):
    layer_sparsity = trial.suggest_float('layer_sparsity', 0.1, 0.9)
    rank = trial.suggest_int('rank', 5, 50)

    print(f"Starting trial with layer_sparsity={layer_sparsity}, rank={rank}")

    # Initialize the Quantum-Inspired GoogLeNet model
    quantum_googlenet = QuantumInspiredGoogLeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_googlenet = optim.Adam(quantum_googlenet.parameters(), lr=0.001)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer_googlenet, T_max=50)

    # Apply quantum-inspired compression methods
    apply_compression_methods(quantum_googlenet, layer_sparsity=layer_sparsity, rank=rank)

    # Train for a limited number of epochs for evaluation
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train_model(quantum_googlenet, train_loader, optimizer_googlenet, criterion, scaler, scheduler)
        val_accuracy, val_loss = evaluate_model(quantum_googlenet, val_loader, criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    return val_loss  # Return the validation loss as the objective metrics

# Optuna study setup
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Print best trial results
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print(f"  Parameters: {trial.params}")

# Use best hyperparameters to train the final model
best_layer_sparsity = trial.params["layer_sparsity"]
best_rank = trial.params["rank"]

# Final model training with best hyperparameters
quantum_resnet = QuantumInspiredResNet().to(device)
apply_compression_methods(quantum_resnet, layer_sparsity=best_layer_sparsity, rank=best_rank)
criterion = nn.CrossEntropyLoss()
optimizer_resnet = optim.Adam(quantum_resnet.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer_resnet, T_max=50)
num_epochs = 50
scaler = GradScaler()

# Training the model
for epoch in range(num_epochs):
    train_loss = train_model(quantum_resnet, train_loader, optimizer_final, criterion, scaler, scheduler=scheduler_final)
    val_accuracy, val_loss = evaluate_model(quantum_resnet, val_loader, criterion)
    print(f"Final Model Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

# Measure inference.eval()
    total_time = 0.0
    num_batches = len(dataloader)
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            start_time = time.time()
            _ = model(inputs)
            end_time = time.time()
            total_time += (end_time - start_time)

    average_time_per_image = total_time / (num_batches * dataloader.batch_size)
    return average_time_per_image

compressed_inference_time = measure_inference_time(final_model, test_loader)
print(f"Compressed Inference Time per Image: {compressed_inference_time:.4f} seconds")

# Save the model
torch.save(final_model.state_dict(), 'quantum_resnet_best.pth')
