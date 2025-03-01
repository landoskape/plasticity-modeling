import torch
from torch import nn
import torch.nn.functional as F
from . import inference


class Oja(nn.Module):
    def __init__(self, num_units: int, num_inputs: int, method: str = "oja"):
        super().__init__()
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.method = method
        self.W = nn.Parameter(torch.randn(num_inputs, num_units))
        self.normalize_weights()

    def normalize_weights(self):
        with torch.no_grad():
            self.W.data = F.normalize(self.W.data, dim=0)
        if self.method == "gha":
            self.gram_schmidt()

    def gram_schmidt(self):
        """Implement Gram-Schmidt on the weights"""

        def proj(u, v):
            return (v @ u).view(1, -1) / (u * u).sum(dim=0).view(1, -1) * u

        with torch.no_grad():
            NC = self.W.size(1)
            for nc in range(1, NC):
                self.W[:, nc] -= torch.sum(proj(self.W[:, :nc], self.W[:, nc]), dim=1)

    def loss(self, x, y):
        if self.method == "oja":
            """In combination with weight normalization leads to Oja's rule"""
            return -0.5 * torch.sum(y**2)

        elif self.method == "gha":  # Compute lower triangular part of y.T @ y
            y_corr_lower = torch.tril(y.T @ y)

            # Compute the loss
            return -0.5 * torch.trace(self.W.T @ self.W @ y_corr_lower) - torch.trace(self.W.T @ x.T @ y)

    def update_weights(self, x, y, lr=0.1):
        if self.method == "oja":
            self.update_oja(x, y, lr)
        elif self.method == "gha":
            self.update_gha(x, y, lr)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def update_oja(self, x, y, lr=0.1):
        """Implement Oja's rule directly"""
        with torch.no_grad():
            dw = x.T @ y - torch.sum(self.W.unsqueeze(1) * (y**2).unsqueeze(0), dim=1)
            self.W += lr * dw

    def update_gha(self, x, y, lr=0.1):
        """Implement GHA rule directly"""
        with torch.no_grad():
            dw = x.T @ y - self.W @ torch.tril(y.T @ y)
            self.W += lr * dw

    def forward(self, x):
        return x @ self.W


class BCM(nn.Module):
    def __init__(self, num_units: int, num_inputs: int, y0: float = 1.0, eps: float = 0.001):
        super().__init__()
        self.num_units = num_units
        self.num_inputs = num_inputs
        self.y0 = y0
        self.eps = eps
        self.W = nn.Parameter(torch.randn(num_inputs, num_units))
        self.normalize_weights()

    def normalize_weights(self):
        with torch.no_grad():
            self.W.data = F.normalize(self.W.data, dim=0)

    def update_weights(self, x, y, lr=0.1):
        with torch.no_grad():
            yhat = torch.mean((y - self.y0) ** 2)
            phi = y * (y - yhat)
            dw = x.T @ phi - self.eps * self.W
            self.W += lr * dw

    def forward(self, x):
        return x @ self.W


class SparseNet(torch.nn.Module):
    """
    Sparse Coding model in PyTorch

    Attributes:
        shape (torch.Size): shape of the input data.
        num_basis (int): number of basis functions.
        dim_basis (int): dimension of a basis function.
        method (str): method used for sparse coding.
        phi (torch.nn.Parameter): Learnable parameter representing the basis functions.
            Initialized with normalized random values for encoding features.
        device (torch.device): device on which the tensors are placed.
    """

    def __init__(self, num_basis: int, shape: torch.Size, method: str = "FISTA", device: torch.device = None) -> None:
        """
        Constructor for SparseNet.

        Parameters:
            num_basis (int): number of basis functions components (phi).
            shape (torch.Size): shape of the input data.
            method (str, optional): algorithm used to infer the sparse coeffiecents (alpha). The available methods: {'FISTA', 'ISTA', 'log'} (default: 'FISTA').
            device (torch.device, optional): device to place tensors on (default: None).
        """
        if not isinstance(num_basis, int) or not isinstance(shape, torch.Size):
            raise TypeError("SparseNet.__init__() arguments (position 1 and 2) must be integers.")

        if method not in inference.Methods.__members__:
            raise TypeError(f"SparseNet.__init__() invalid value for 'method': allowed values are {tuple(inference.Methods.__members__.keys())}.")

        super().__init__()
        self.device = device

        self.method = method
        self.shape = shape
        self.num_basis = num_basis
        self.dim_basis = shape.numel()

        self.__phi = torch.nn.Parameter(data=F.normalize(torch.randn(self.num_basis, self.dim_basis, device=device), p=2, dim=1), requires_grad=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for SparseNet.

        Parameters:
            x (torch.Tensor): input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the sparse coefficients (alpha) and the reconstructed image.
        """
        if isinstance(x, torch.Tensor):
            x.to(self.device)
            x = torch.flatten(x, start_dim=1)
        else:
            raise TypeError("SparseNet(): argument 'x' (position 1) must be a Tensor.")

        if x.dim() != 2 or x.shape[1] != self.dim_basis:
            raise ValueError(f"SparseNet(): input tensor must have shape (batch_size, {self.dim_basis}): got shape {tuple(x.shape)}.")

        if self.method == "FISTA":
            alpha = inference.FISTA(x, phi=self.__phi.detach(), device=self.device)
        elif self.method == "ISTA":
            alpha = inference.ISTA(x, phi=self.__phi.detach(), device=self.device)
        elif self.method == "LOG":
            alpha = inference.LOG_REGU(x, phi=self.__phi.detach(), device=self.device)

        recon = torch.mm(alpha, self.__phi).view(-1, self.shape[0], self.shape[1])
        return alpha, recon

    @property
    def phi(self) -> torch.Tensor:
        """
        Get the basis functions elements (phi) of the Sparse Coding model.

        Returns:
            torch.Tensor: the set of basis functions.
        """
        return self.__phi.detach().view(-1, self.shape[0], self.shape[1])

    @property
    def W(self) -> torch.Tensor:
        """
        Get the basis functions elements (phi) of the Sparse Coding model.

        Returns:
            torch.Tensor: the set of basis functions.
        """
        return self.__phi.detach().T

    def loss(self, img_batch, pred):
        return ((img_batch - pred) ** 2).sum()

    def normalize_weights(self):
        with torch.no_grad():
            self.__phi.data = F.normalize(self.__phi.data, dim=1)
