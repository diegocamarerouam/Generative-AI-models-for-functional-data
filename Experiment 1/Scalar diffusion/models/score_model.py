import torch
from torch import nn
from torch import Tensor
from typing import Callable, Union
from abc import ABC, abstractmethod
import numpy as np

RandomStateLike = Union[int, np.random.RandomState, torch.Generator, None]


def validate_random_state(seed: RandomStateLike) -> Union[int, np.random.RandomState, None]:
    if seed is None or isinstance(seed, (int, np.random.RandomState, torch.Generator)):
        return seed
    raise TypeError(f"Invalid random state type: {type(seed)}")


def make_torch_generator(seed: RandomStateLike, device: str | torch.device = "cpu") -> torch.Generator:
    """ 
    Generates a torch.Generator on the appropriate device using NumPy or int seeds.
    """
    target_device = torch.device(device)
    dev_type = target_device.type
    
    generator = torch.Generator(device=dev_type)
    
    if seed is None:
        generator.seed()
    elif isinstance(seed, int):
        generator.manual_seed(seed)
    elif isinstance(seed, np.random.RandomState):
        uint32_seed = int(seed.randint(0, 2**31 - 1))
        generator.manual_seed(uint32_seed)
    elif isinstance(seed, torch.Generator):
        return seed
    else:
        raise TypeError(f"Cannot create torch.Generator from type: {type(seed)}")
        
    return generator


class Swish(nn.Module):
    """Module form of swish activation to keep the model serializable."""
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time steps."""

    def __init__(
        self, 
        n_points: int, 
        scale: float = 30.0, 
        random_state: RandomStateLike = None, 
        device: str | torch.device = "cpu"
    ):
        super().__init__()

        _gen = make_torch_generator(random_state, device=device)
        self.rff_weights = nn.Parameter(
            torch.randn(n_points // 2, generator=_gen, device=device) * scale,
            requires_grad=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * torch.pi
        gaus = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return gaus


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.dense(x)[..., None]


class ScoreModel(nn.Module, ABC):
    """Abstract base class for time-dependent score-based models."""

    @abstractmethod
    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass of the score-based model.

        Args:
          x: The input functional data as a tensor, shape (N, M) or (N, 1, M).
          t: The time steps as a tensor, shape (N,).
          y: Optional tensor with the class labels of the data, shape (N,). 
            Default is `None`.

        Returns:
          The output of the score-based model, shape (N, M).
        """
        ...

class UNetScoreModel(ScoreModel):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        multiply_inv_sigma: Callable[[Tensor, Tensor], Tensor] | None = None,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        kernel_sizes: tuple[int, ...] = (9, 9, 9, 9),
        n_groups: tuple[int, ...] = (4, 32, 32, 32),
        n_points: int = 100, 
        device: str | torch.device = "cpu",
        random_state: RandomStateLike = None,
    ):
        """Initialize a time-dependent score-based network.
        Args:
            multiply_inv_sigma: A callable ``(h, t) -> h_scaled`` that applies
                            the inverse square-root covariance operator to
                            ``h``. For diagonal/scalar processes this is
                            pointwise multiplication; for circulant processes
                            it applies ``Q diag(inv_sigma_Y(t)) Q^T`` via FFT.
                           If None, no scaling is applied. Default is None.
          channels: The number of channels for feature maps of each resolution.
          n_points: The dimensionality of Gaussian random Fourier feature
          embeddings.
          kernel_sizes: The kernel sizes for the convolutional layers.
          n_groups: The number of groups for group normalization in each layer.
          random_state: The random state to use for reproducibility
          of the Gaussian random Fourier features. Default is None.
        """
        
        super().__init__()
        random_state = validate_random_state(random_state)
        
        self.device = device
        self.n_points = n_points
        self.channels = channels
        self.n_groups = n_groups
        
        n_points_even = n_points + (n_points % 2)
        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(n_points=n_points_even, random_state=random_state, device=device),
            nn.Linear(n_points_even, n_points),
        )
        
        # Encoding layers
        self.conv1 = nn.Conv1d(1, channels[0], kernel_sizes[0], stride=1, padding=kernel_sizes[0]//2, bias=False)
        self.dense1 = Dense(n_points, channels[0])
        self.gnorm1 = nn.GroupNorm(n_groups[0], num_channels=channels[0])
        
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_sizes[1], stride=4, padding=kernel_sizes[1]//2, bias=False)
        self.dense2 = Dense(n_points, channels[1])
        self.gnorm2 = nn.GroupNorm(n_groups[1], num_channels=channels[1])
        
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_sizes[2], stride=4, padding=kernel_sizes[2]//2, bias=False)
        self.dense3 = Dense(n_points, channels[2])
        self.gnorm3 = nn.GroupNorm(n_groups[2], num_channels=channels[2])
        
        self.conv4 = nn.Conv1d(channels[2], channels[3], kernel_sizes[3], stride=4, padding=kernel_sizes[3]//2, bias=False)
        self.dense4 = Dense(n_points, channels[3])
        self.gnorm4 = nn.GroupNorm(n_groups[3], num_channels=channels[3])

        # Decoding layers
        self.tconv4 = nn.ConvTranspose1d(
            channels[3], channels[2], kernel_size=kernel_sizes[3], padding=kernel_sizes[3]//2,
            stride=kernel_sizes[3]//2, bias=False, output_padding=kernel_sizes[3]//2-1,
        )
        self.dense5 = Dense(n_points, channels[2])
        self.tgnorm4 = nn.GroupNorm(n_groups[3], num_channels=channels[2])
        
        self.tconv3 = nn.ConvTranspose1d(
            channels[2] + channels[2], channels[1], kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2,
            stride=kernel_sizes[2]//2, bias=False, output_padding=kernel_sizes[2]//2-1,
        )
        self.dense6 = Dense(n_points, channels[1])
        self.tgnorm3 = nn.GroupNorm(n_groups[2], num_channels=channels[1])
        
        self.tconv2 = nn.ConvTranspose1d(
            channels[1] + channels[1], channels[0], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2,
            stride=kernel_sizes[1]//2, bias=False, output_padding=kernel_sizes[1]//2-1,
        )
        self.dense7 = Dense(n_points, channels[0])
        self.tgnorm2 = nn.GroupNorm(n_groups[1], num_channels=channels[0])
        
        self.tconv1 = nn.ConvTranspose1d(channels[0] + channels[0], 1, kernel_sizes[0], padding=kernel_sizes[0]//2, stride=1, bias=False)

        self.act = Swish()
        self.multiply_inv_sigma = multiply_inv_sigma
        
        self.to(device)

    def get_config(self) -> dict[str, int | tuple[int, ...]]:
        """Return constructor arguments needed to rebuild the architecture."""
        return {
            "n_points": self.n_points,
            "channels": tuple(self.channels),
            "n_groups": tuple(self.n_groups),
        }

    def forward(self, x: Tensor, t: Tensor, y: Tensor | None = None) -> Tensor:
        """Forward pass of the score-based model.

        Args:
          x: The input functional data as a tensor, shape (N, M) or (N, 1, M).
          t: The time steps as a tensor, shape (N,).
          y: Optional tensor with the class labels of the data, shape (N,).
             Default is `None`.

        Returns:
          The output of the score-based model, shape (N, M).
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device, dtype=x.dtype)
            
        if t.dim() == 0:
            t = t.expand(x.shape[0])
            
        embed = self.act(self.embed(t))
        x_in = x.unsqueeze(1) if x.dim() == 2 else x

        # Encoder path
        h1 = self.conv1(x_in)
        h1 += self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))
        
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        # Decoding path
        h = self.tconv4(h4)
        h = self._match_size(h, h3)
        h += self.dense5(embed)
        h = self.act(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.act(self.tgnorm3(h))
        h = self._match_size(h, h2)
        
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.act(self.tgnorm2(h))
        h = self._match_size(h, h1)

        h = self.tconv1(torch.cat([h, h1], dim=1))
        
        h = h.squeeze(1)
        if self.multiply_inv_sigma is not None:
            h = self.multiply_inv_sigma(h, t)

        if h.isnan().any():
            raise ValueError("NaN values encountered in score network output.")
        return h.unsqueeze(1) if x.dim() != h.dim() else h

    def _match_size(self, x: Tensor, target: Tensor) -> Tensor:
        """Crop or pad x to match target's spatial dimensions."""
        if x.shape[-1] != target.shape[-1]:
            diff = x.shape[-1] - target.shape[-1]
            if diff > 0:
                start = diff // 2
                return x[..., start:start + target.shape[-1]]
            pad_total = -diff
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return nn.functional.pad(x, (pad_left, pad_right))
        return x