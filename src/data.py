import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_mnist_loaders(
    batch_size=128, train_size=None, test_size=None, download=True, root="./data"
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.MNIST(
        root=root, train=True, download=download, transform=transform
    )
    test = datasets.MNIST(
        root=root, train=False, download=download, transform=transform
    )
    if train_size is not None and train_size < len(train):
        train = Subset(train, list(range(train_size)))
    if test_size is not None and test_size < len(test):
        test = Subset(test, list(range(test_size)))
    if batch_size is None:
        batch_size = len(train)
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        batch_size = len(test)
        test_loader = DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    return train_loader, test_loader


def get_cifar10_loaders(
    batch_size=128, train_size=None, test_size=None, download=True, root="./data"
):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train = datasets.CIFAR10(
        root=root, train=True, download=download, transform=transform
    )
    test = datasets.CIFAR10(
        root=root, train=False, download=download, transform=transform
    )
    if train_size is not None and train_size < len(train):
        train = Subset(train, list(range(train_size)))
    if test_size is not None and test_size < len(test):
        test = Subset(test, list(range(test_size)))
    if batch_size is None:
        batch_size = len(train)
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        batch_size = len(test)
        test_loader = DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        test_loader = DataLoader(
            test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
        )
    return train_loader, test_loader
