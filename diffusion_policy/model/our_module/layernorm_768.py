import torch


class LayerNorm(torch.nn.Module):
    """Differentiable LayerNorm used by the 768-dimensional QAT path."""

    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.register_buffer(
            "delta",
            torch.tensor([1.0 / normalized_shape], dtype=torch.float16),
            persistent=False,
        )
        self.weight = None
        self.bias = None

    def forward(self, input: torch.Tensor):
        input = input.to(torch.float16)
        mean = input.sum(dim=-1, keepdim=True) * self.delta
        minus = input - mean
        var = torch.square(minus).sum(dim=-1, keepdim=True)
        std = torch.sqrt(1 / (var + self.eps))
        return (minus * std).to(torch.float16)
