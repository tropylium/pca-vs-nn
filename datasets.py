from dataclasses import dataclass
import torch
import json
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


@dataclass
class Dataset:
    name: str
    train: torch.Tensor
    valid: torch.Tensor
    mean: torch.Tensor
    metadata: dict


def load_dataset_raw(
    name: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if name == "MNIST":
        # load data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                #  transforms.Normalize((0.5,), (0.5,))
            ]
        )
        # MNIST
        train_dataset = datasets.MNIST(
            root="datasets", train=True, transform=transform, download=True
        )
        test_dataset = datasets.MNIST(
            root="datasets", train=False, transform=transform, download=True
        )
    elif name == "SVHN":
        # load data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )
        # SVHN
        train_dataset = datasets.SVHN(
            root="datasets", split="train", transform=transform, download=True
        )
        test_dataset = datasets.SVHN(
            root="datasets", split="test", transform=transform, download=True
        )
    elif name == "CIFAR10":
        # load data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(
                #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # ),
            ]
        )
        # CIFAR10
        train_dataset = datasets.CIFAR10(
            root="datasets", train=True, transform=transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            root="datasets", train=False, transform=transform, download=True
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=128,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1024, shuffle=False, num_workers=128
    )

    all_train_data = []
    for data, _ in train_loader:
        all_train_data.append(data)
    all_train_data = torch.concatenate(all_train_data).to(device)

    all_valid_data = []
    for data, _ in test_loader:
        all_valid_data.append(data)
    all_valid_data = torch.concatenate(all_valid_data).to(device)

    return all_train_data, all_valid_data


def load_dataset(name: str, device: torch.device) -> Dataset:
    train_dataset, valid_dataset = load_dataset_raw(name, device)
    metadata_dir = Path("datasets-metadata")
    metadata_path = metadata_dir / f"{name}.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    mean = torch.load(metadata_dir / metadata["mean_file"], weights_only=True)
    assert isinstance(mean, torch.Tensor)
    mean = mean.unsqueeze(0).to(device)
    print(mean.shape, train_dataset.shape, valid_dataset.shape)

    train_dataset = train_dataset - mean
    valid_dataset = valid_dataset - mean
    return Dataset(name, train_dataset, valid_dataset, mean, metadata)
