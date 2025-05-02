import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pathlib import Path
import time
import json

from typing import *

from tqdm import tqdm

import itertools
import argparse

from datasets import load_dataset

class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        hidden_layers: int,
        residual_connection: bool = False,
    ):
        """
        Args:
            input_dim: int, input dimension
            hidden_dim: int, dimension of hidden layers
            latent_dim: int, dimension of latent space
            hidden_layers: int, number of hidden layers
            residual_connection: bool, whether to use residual connection in encoder/decoder
        """
        super(Autoencoder, self).__init__()
        assert hidden_layers > 0, "hidden_layers must be greater than 0"

        # observe this accounts for hidden_layers=1 properly
        layer_dims = [[hidden_dim, hidden_dim] for _ in range(hidden_layers)]
        layer_dims[0][0] = input_dim
        layer_dims[-1][1] = latent_dim

        encoder_layers = []
        for input_d, output_d in layer_dims:
            encoder_layers.append(nn.Linear(input_d, output_d))
            if output_d != latent_dim:
                encoder_layers.append(nn.ReLU())

        self.encoder_nn = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for output_d, input_d in reversed(layer_dims):
            if input_d != latent_dim:
                decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(input_d, output_d))

        self.decoder_nn = nn.Sequential(*decoder_layers)

        if residual_connection:
            self.encoder_residual = nn.Linear(input_dim, latent_dim)
            self.decoder_residual = nn.Linear(latent_dim, input_dim)
        else:
            self.encoder_residual = None
            self.decoder_residual = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_residual is not None:
            x = self.encoder_residual(x) + self.encoder_nn(x)
        else:
            x = self.encoder_nn(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder_residual is not None:
            z = self.decoder_residual(z) + self.decoder_nn(z)
        else:
            z = self.decoder_nn(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.view(x.size(0), -1)
        x = self.encode(x)
        x = self.decode(x)
        x = x.view(orig_shape)
        return x


def train_and_get_results(
    dataset: tuple[torch.Tensor, torch.Tensor],
    model_kwargs: dict[str, Any],
    device: torch.device,
    rng: torch.Generator,
    checkpoint_func: Optional[Callable] = None,
    batch_size: int = 100,
    epochs: int = 100,
):
    model = Autoencoder(**model_kwargs).to(device)
    all_train_data, all_valid_data = dataset
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    start_time = time.time()
    epoch_data = {}
    for epoch in range(epochs):
        # Training loop
        train_loss = 0
        with torch.enable_grad():
            # Simulate dataloader shuffling
            permuted_data = torch.randperm(len(all_train_data), generator=rng)
            iters = len(all_train_data) // batch_size
            for i in range(iters):
                data = all_train_data[
                    permuted_data[i * batch_size : (i + 1) * batch_size]
                ]
                data = data.to(device)
                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, data)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
        scheduler.step()
        train_loss /= iters

        # Validation loop
        with torch.no_grad():
            output = model(all_valid_data)
            valid_loss = criterion(output, all_valid_data).item()

        epoch_data[epoch] = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "time_taken": time.time() - start_time,
        }

        if checkpoint_func is not None:
            checkpoint_func(model, epoch)

    return model, epoch_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument("--device", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=18065)
    # parser.add_argument("--latent_dim", type=int)
    # parser.add_argument("--hidden_dim", type=int)
    args = parser.parse_args()

    # Setup
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("using device", device)
    rng = torch.Generator().manual_seed(args.seed)

    train_dataset, valid_dataset, metadata = load_dataset(args.dataset, device)

    arglist = itertools.product(
        [1, 2, 4, 8, 16, 32, 64],
        [True, False],
    )

    for hidden_layers, residual_connection in arglist:
        print(f"Training with hidden_layers={hidden_layers} and residual_connection={residual_connection}")
        exp_dir = Path("results") / f"{args.dataset} hidden_layers={hidden_layers} residual={residual_connection}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        model_kwargs = {
            "input_dim": train_dataset[0].numel(),
            "hidden_dim": metadata["hidden_dim"],
            "latent_dim": metadata["latent_dim"],
            "hidden_layers": hidden_layers,
            "residual_connection": residual_connection,
        }

        def checkpoint_func(model, epoch):
            if epoch % 10 == 0:
                torch.save(model.state_dict(), exp_dir / f"epoch={epoch}.pth")

        model, epoch_data = train_and_get_results(
            (train_dataset, valid_dataset),
            model_kwargs,
            device,
            rng,
            checkpoint_func=checkpoint_func,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

        with open(exp_dir / "epoch_data.json", "w") as f:
            json.dump(epoch_data, f, indent=4)

        torch.save(model.state_dict(), exp_dir / "model.pth")
