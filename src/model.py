import torch.nn as nn

class SimpleReLUMLP(nn.Module):
    """Simple fully-connected ReLU MLP for MNIST (flattened 28x28 input -> hidden layers -> 10 logits)."""
    def __init__(self, hidden_sizes=[512, 256, 128], input_dim=28*28, n_classes=10):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU(inplace=True))
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # expect x shape (B,1,28,28) or (B, 784)
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        return self.net(x)
