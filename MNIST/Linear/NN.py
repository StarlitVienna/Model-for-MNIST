import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten_layer = nn.Flatten()
        self.layer_1 = nn.Linear(in_features=28*28, out_features=8*8*8)
        self.layer_2 = nn.Linear(in_features=8*8*8, out_features=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(self.layer_1(self.flatten_layer(x)))


def train_model(model, optimizer, loss_fn, train_dataloader, epochs):
    model.train()
    for epoch in range(epochs):
        for batch, (batch_features, batch_labels) in enumerate(train_dataloader):
            logits = model(batch_features)
            loss = loss_fn(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()


train_data = datasets.MNIST(
        root="./data/",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=None
        )

test_data = datasets.MNIST(
        root="./data/",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=None
        )

train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=32
        )

test_dataloader = DataLoader(
        test_data,
        shuffle=True,
        batch_size=32
        )
