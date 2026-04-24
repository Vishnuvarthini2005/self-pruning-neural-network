"""
Self-Pruning Neural Network for CIFAR-10 Classification
=========================================================
Tredence AI Engineering Intern – Case Study Solution

This script implements a feed-forward neural network that learns to prune
itself during training using learnable gate parameters and L1 sparsity
regularization.

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────────────────────
# Part 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom linear layer augmented with learnable gate parameters.

    Each weight in this layer has a corresponding scalar 'gate_score'.
    During the forward pass, gate_scores are passed through a Sigmoid
    to produce gates in [0, 1]. Weights are then element-wise multiplied
    by these gates before performing the linear transformation.

    When a gate collapses to 0, the associated weight is effectively
    pruned from the network. The L1 sparsity loss (applied externally)
    drives gates toward 0 during training.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
    """

    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()

        # Standard weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores: one per weight, same shape as weight tensor.
        # Initialized to small positive values so gates start near 0.5
        # (sigmoid(0) = 0.5), giving the optimizer room to push either way.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Kaiming uniform initialization for weights (good for ReLU networks)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def get_gates(self) -> torch.Tensor:
        """Return current gate values (after sigmoid transformation)."""
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        1. Transform gate_scores through Sigmoid → gates ∈ (0, 1)
        2. Compute pruned_weights = weight ⊙ gates  (element-wise)
        3. Apply linear transformation: x @ pruned_weights.T + bias

        Gradients flow through both self.weight and self.gate_scores
        automatically via autograd, because all operations are differentiable.
        """
        gates = torch.sigmoid(self.gate_scores)         # (out, in)
        pruned_weights = self.weight * gates            # element-wise
        return F.linear(x, pruned_weights, self.bias)   # standard linear op


# ─────────────────────────────────────────────────────────────
# Neural Network using PrunableLinear Layers
# ─────────────────────────────────────────────────────────────

class SelfPruningNetwork(nn.Module):
    """
    Feed-forward network for CIFAR-10 (32×32 RGB → 10 classes).
    All fully-connected layers use PrunableLinear.

    Architecture:
        Input (3072) → 1024 → 512 → 256 → 10
    """

    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)   # Flatten (B, 3, 32, 32) → (B, 3072)
        return self.net(x)

    def get_all_prunable_layers(self):
        """Return all PrunableLinear layers in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


# ─────────────────────────────────────────────────────────────
# Part 2: Sparsity Regularization Loss
# ─────────────────────────────────────────────────────────────

def sparsity_loss(model: SelfPruningNetwork) -> torch.Tensor:
    """
    Compute the L1 sparsity regularization term.

    L1 norm of all gate values (sum of |gate| across all PrunableLinear
    layers). Since sigmoid output is always positive, |gate| = gate.

    Why L1 encourages sparsity:
        The L1 norm has a non-zero, constant gradient (±1) everywhere
        except 0. This means it applies a constant "push" toward zero
        on every gate, regardless of how small the gate value already is.
        L2 norm, by contrast, has gradient 2x → as x→0, the gradient
        also→0, so L2 does NOT reliably drive values to exactly 0.
        L1 thus naturally creates a "winner-takes-all" effect, collapsing
        unimportant gates to exactly 0 while allowing important ones to
        remain large.

    Returns:
        Scalar tensor representing the total sparsity penalty.
    """
    total = torch.tensor(0.0, requires_grad=True)
    for layer in model.get_all_prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total = total + gates.sum()
    return total


def compute_sparsity_level(model: SelfPruningNetwork, threshold: float = 1e-2) -> float:
    """
    Calculate the percentage of weights whose gate value < threshold.
    A higher sparsity level means more effective self-pruning.

    Args:
        model: The SelfPruningNetwork instance.
        threshold: Gates below this value are considered 'pruned'.

    Returns:
        Sparsity level as a percentage (0–100).
    """
    total_weights = 0
    pruned_weights = 0

    with torch.no_grad():
        for layer in model.get_all_prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total_weights += gates.numel()
            pruned_weights += (gates < threshold).sum().item()

    return 100.0 * pruned_weights / total_weights


# ─────────────────────────────────────────────────────────────
# Part 3: Data Loading
# ─────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Download and return CIFAR-10 train/test DataLoaders."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=transform_train)
    test_dataset  = datasets.CIFAR10(root='./data', train=False,
                                     download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=256,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, lam):
    """Run one training epoch. Returns (avg_total_loss, avg_clf_loss, avg_sp_loss)."""
    model.train()
    total_loss_sum = clf_loss_sum = sp_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Classification loss
        clf = F.cross_entropy(logits, labels)

        # Sparsity regularization
        sp = sparsity_loss(model)

        # Total loss = CE + λ * L1_gate_norm
        loss = clf + lam * sp
        loss.backward()

        # Gradient clipping to stabilize training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_sum += loss.item()
        clf_loss_sum   += clf.item()
        sp_loss_sum    += sp.item()

    n = len(loader)
    return total_loss_sum / n, clf_loss_sum / n, sp_loss_sum / n


@torch.no_grad()
def evaluate(model, loader, device):
    """Return test accuracy (%) on the given loader."""
    model.eval()
    correct = total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return 100.0 * correct / total


def train_model(lam: float, epochs: int, device, train_loader, test_loader):
    """
    Full training run for a given lambda value.

    Args:
        lam: Sparsity regularization weight (λ).
        epochs: Number of training epochs.
        device: torch.device.
        train_loader / test_loader: DataLoaders.

    Returns:
        model, final_test_accuracy, final_sparsity_level
    """
    print(f"\n{'='*60}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*60}")

    model = SelfPruningNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        total_l, clf_l, sp_l = train_one_epoch(
            model, train_loader, optimizer, device, lam
        )
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sp  = compute_sparsity_level(model)
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss={total_l:.4f} (CE={clf_l:.4f}, SP={sp_l:.4f}) | "
                  f"Test Acc={acc:.2f}% | Sparsity={sp:.2f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sp  = compute_sparsity_level(model)
    print(f"\n  FINAL  → Test Accuracy: {final_acc:.2f}%  |  "
          f"Sparsity Level: {final_sp:.2f}%")

    return model, final_acc, final_sp


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_gate_distribution(model: SelfPruningNetwork, lam: float,
                           save_path: str = "gate_distribution.png"):
    """
    Plot histogram of all gate values across PrunableLinear layers.

    A successful pruning run will show:
      - A large spike near 0 (pruned/inactive weights)
      - A secondary cluster away from 0 (active/important weights)
    """
    all_gates = []

    with torch.no_grad():
        for layer in model.get_all_prunable_layers():
            gates = torch.sigmoid(layer.gate_scores).cpu().numpy().flatten()
            all_gates.extend(gates.tolist())

    all_gates = np.array(all_gates)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_gates, bins=100, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(x=0.01, color='crimson', linestyle='--', linewidth=1.5,
               label='Prune threshold (0.01)')
    ax.set_xlabel("Gate Value (σ(gate_score))", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(f"Gate Value Distribution  |  λ = {lam}", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.text(0.7, 0.85,
            f"Sparsity: {(all_gates < 0.01).mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Gate distribution plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    EPOCHS = 30
    BATCH_SIZE = 128
    LAMBDAS = [1e-4, 1e-3, 5e-3]   # Low, Medium, High sparsity pressure

    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)

    results = []
    best_model = None
    best_lam   = None

    for lam in LAMBDAS:
        model, acc, sp = train_model(lam, EPOCHS, device, train_loader, test_loader)
        results.append((lam, acc, sp))

        # Track best model (highest accuracy)
        if best_model is None or acc > results[0][1]:
            best_model = model
            best_lam   = lam

    # ── Summary Table ──────────────────────────────────────────
    print("\n\n" + "═"*55)
    print("  Results Summary")
    print("═"*55)
    print(f"  {'Lambda':>10}  {'Test Accuracy':>15}  {'Sparsity (%)':>14}")
    print("  " + "-"*50)
    for lam, acc, sp in results:
        print(f"  {lam:>10.4f}  {acc:>14.2f}%  {sp:>13.2f}%")
    print("═"*55)

    # ── Plot best model's gate distribution ───────────────────
    if best_model is not None:
        plot_gate_distribution(best_model, best_lam, "gate_distribution.png")


if __name__ == "__main__":
    main()
