"""MNIST 手写数字分类：由 08手写分类.ipynb 整理而来。

默认依次训练“手写参数更新”和 nn.Module 两个版本，并把图表、模型保存到
项目根目录的 outputs/mnist。运行 ``python src/mnist_classifier.py --help``
查看可选参数。
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mnist"
MEAN, STD = 0.1307, 0.3081
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 784)
        return self.fc2(F.relu(self.fc1(x)))


def make_loaders(
    data_dir: Path,
    batch_size: int,
    limit_train: int | None = None,
    limit_test: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))]
    )
    train_set = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    if limit_train is not None:
        train_set = Subset(train_set, range(min(limit_train, len(train_set))))
    if limit_test is not None:
        test_set = Subset(test_set, range(min(limit_test, len(test_set))))

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=generator
    )
    test_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False)
    print(f"训练集：{len(train_set)} 张；测试集：{len(test_set)} 张")
    return train_loader, test_loader


def evaluate_logits(test_loader: DataLoader, predict) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            prediction = predict(xb).argmax(dim=1)
            correct += (prediction == yb).sum().item()
            total += yb.numel()
    return correct / total


def train_manual(
    train_loader: DataLoader, test_loader: DataLoader, epochs: int, lr: float
) -> tuple[list[float], list[float]]:
    torch.manual_seed(42)
    w1 = (torch.randn(784, 128) * (2.0 / 784) ** 0.5).requires_grad_()
    b1 = torch.zeros(128, requires_grad=True)
    w2 = (torch.randn(128, 10) * (2.0 / 128) ** 0.5).requires_grad_()
    b2 = torch.zeros(10, requires_grad=True)
    params = [w1, b1, w2, b2]

    def predict(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x.view(-1, 784) @ w1 + b1) @ w2 + b2

    losses, accuracies = [], []
    for epoch in range(epochs):
        running_loss = 0.0
        samples = 0
        for xb, yb in train_loader:
            loss = F.cross_entropy(predict(xb), yb)
            loss.backward()
            with torch.no_grad():
                for param in params:
                    param -= lr * param.grad
                    param.grad.zero_()
            running_loss += loss.item() * yb.size(0)
            samples += yb.size(0)
        losses.append(running_loss / samples)
        accuracies.append(evaluate_logits(test_loader, predict))
        print(
            f"[手写] epoch {epoch + 1}/{epochs}, "
            f"train_loss: {losses[-1]:.4f}, test_accuracy: {accuracies[-1]:.4f}"
        )
    return losses, accuracies


def train_module(
    train_loader: DataLoader, test_loader: DataLoader, epochs: int, lr: float
) -> tuple[SimpleMLP, list[float], list[float]]:
    torch.manual_seed(42)
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        samples = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * yb.size(0)
            samples += yb.size(0)
        model.eval()
        losses.append(running_loss / samples)
        accuracies.append(evaluate_logits(test_loader, model))
        print(
            f"[Module] epoch {epoch + 1}/{epochs}, "
            f"train_loss: {losses[-1]:.4f}, test_accuracy: {accuracies[-1]:.4f}"
        )
    return model, losses, accuracies


def save_examples(dataset, path: Path, model: SimpleMLP | None = None) -> None:
    indices = list(range(9)) if model is None else random.Random(7).sample(range(len(dataset)), 9)
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ax, index in zip(axes.flat, indices):
        image, label = dataset[index]
        ax.imshow(image.squeeze() * STD + MEAN, cmap="gray")
        if model is None:
            title, color = f"label: {label}", "black"
        else:
            with torch.no_grad():
                probabilities = F.softmax(model(image.unsqueeze(0)), dim=1)[0]
            prediction = int(probabilities.argmax())
            title = f"true {label}, predicted {prediction} ({float(probabilities[prediction]):.0%})"
            color = "green" if prediction == label else "red"
        ax.set_title(title, fontsize=10, color=color)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_history(histories: dict[str, tuple[list[float], list[float]]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, (losses, accuracies) in histories.items():
        epochs = range(1, len(losses) + 1)
        display_name = "manual" if name == "手写更新" else "nn.Module"
        axes[0].plot(epochs, losses, marker="o", label=display_name)
        axes[1].plot(epochs, accuracies, marker="o", label=display_name)
    axes[0].set(title="Training loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="Test accuracy", xlabel="epoch", ylabel="accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练并评估 MNIST 手写数字分类器")
    parser.add_argument("--implementation", choices=("manual", "module", "both"), default="both")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-train", type=int, help="仅用于快速验证：限制训练样本数")
    parser.add_argument("--limit-test", type=int, help="仅用于快速验证：限制测试样本数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs 和 --batch-size 必须大于 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.limit_train, args.limit_test
    )
    save_examples(train_loader.dataset, args.output_dir / "train_samples.png")
    histories: dict[str, tuple[list[float], list[float]]] = {}
    if args.implementation in ("manual", "both"):
        histories["手写更新"] = train_manual(
            train_loader, test_loader, args.epochs, args.learning_rate
        )
    if args.implementation in ("module", "both"):
        model, losses, accuracies = train_module(
            train_loader, test_loader, args.epochs, args.learning_rate
        )
        histories["nn.Module"] = (losses, accuracies)
        checkpoint = args.output_dir / "simple_mlp.pt"
        torch.save(model.state_dict(), checkpoint)
        save_examples(test_loader.dataset, args.output_dir / "predictions.png", model)
        print(f"模型已保存：{checkpoint}")
    save_history(histories, args.output_dir / "training_history.png")
    print(f"图表已保存：{args.output_dir}")


if __name__ == "__main__":
    main()
