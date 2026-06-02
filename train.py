import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import MLP


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    return 100.0 * correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform_train)
    test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model     = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    os.makedirs("models", exist_ok=True)
    best_acc = 0.0

    print(f"{'Época':>6}  {'Loss':>8}  {'Treino%':>8}  {'Teste%':>8}")
    print("-" * 40)

    for epoch in range(1, 21):
        model.train()
        total_loss = correct = total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out  = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(data)
            correct    += out.argmax(1).eq(target).sum().item()
            total      += len(data)

        scheduler.step()

        train_acc = 100.0 * correct / total
        test_acc  = evaluate(model, test_loader, device)
        avg_loss  = total_loss / total

        print(f"{epoch:>6}  {avg_loss:>8.4f}  {train_acc:>8.2f}  {test_acc:>8.2f}", end="")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "models/mlp_mnist.pth")
            print("  ✓ salvo", end="")

        print()

    print(f"\nMelhor acurácia no teste: {best_acc:.2f}%")
    print("Modelo salvo em models/mlp_mnist.pth")


if __name__ == "__main__":
    train()
