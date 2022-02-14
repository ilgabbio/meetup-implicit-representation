from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import Tensor, no_grad
from torch.nn import Module, Sequential, Linear, ReLU, MSELoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from PIL import Image


class ImplicitImage:
    def __init__(self, nn: Module) -> None:
        self._nn = nn
    
    def render(self, width: int, height: int) -> np.ndarray:
        with no_grad():
            self._nn.eval()
            xs = np.linspace(0, 1, width)
            return np.clip(
                np.array([
                    self._render_line(xs, y)
                    for y in np.linspace(0, 1, height)
                ]), 0, 1)
        
    def _render_line(self, xs: np.ndarray, y: float) -> np.ndarray:
        batch = np.transpose(np.broadcast_arrays(xs, y))
        return self._nn(Tensor(batch.astype(np.float32))).numpy()


class Mlp(Sequential):
    def __init__(
        self, 
        layer_sizes: List[int], 
        activation: Module=ReLU()
    ) -> None:
        super().__init__(*[
            Sequential(
                Linear(layer_sizes[i-1], layer_sizes[i]),
                activation
            )
            for i in range(1,len(layer_sizes))
        ])


class ImageDataset(Dataset[Tuple[Tensor,Tensor]]):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self._image = image

    def __len__(self):
        return self._image.shape[0]*self._image.shape[1]
    
    def __getitem__(self, index) -> Tuple[Tensor,Tensor]:
        x = index % self._image.shape[1]
        y = index // self._image.shape[0]

        value = np.array((x / self._image.shape[1], y / self._image.shape[0]))
        value = Tensor(value.astype(np.float32))
        label = Tensor(self._image[y,x,:].astype(np.float32) / 255)

        return value, label


def train_implicit_image(
    image: np.ndarray,
    layer_size: int = 32,
    num_layers: int = 5,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    reconstruction_path: Optional[str] = None
) -> ImplicitImage:
    data = DataLoader(
        dataset=ImageDataset(image),
        batch_size=batch_size,
        shuffle=True,
    )

    model = Mlp([2] + [layer_size]*num_layers + [3])

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss = MSELoss()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {num_layers} hidden layers with {layer_size} neurons each.")
    print(f"Total trainable parameters: {num_params}")
    for e in range(epochs):
        train_error = 0
        for coords, colors in data:
            model.train()
            optimizer.zero_grad()

            estimated_colors = model(coords)
            error = loss(estimated_colors, colors)
            train_error += error.item()

            error.backward()
            optimizer.step()

        train_error /= len(data)
        print(f"Epoch: {e+1}, error: {train_error}")

        if reconstruction_path is not None:
            image_data = ImplicitImage(model).render(480,640)
            image_data = (image_data * 255).astype(np.uint8)
            img = Image.fromarray(image_data, mode="RGB")
            img.save(f"{reconstruction_path}{e+1:04d}.png")


if __name__ == "__main__":
    im = train_implicit_image(
        np.asarray(Image.open("data/img01.jpg")),
        reconstruction_path="data/results/img01_epoch"
    )