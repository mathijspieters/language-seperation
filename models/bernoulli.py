import torch.nn as nn
import torch
from torch.nn import Linear, Sequential

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STE_bernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        unif = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        sample = unif.sample(sample_shape=input.size()).squeeze(-1).to(DEVICE)
        return torch.ceil(input - sample)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output#, None, None, None


class STE(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(STE, self).__init__()

        self.layer = Sequential(
            Linear(in_features, out_features, bias=True)
        )

        self.sigmoid = nn.Sigmoid()

        self.act = STE_bernoulli.apply

    def forward(self, x):
        logits = self.layer(x)  # [B, T, 1]
        logits = self.sigmoid(logits)
        # binary layer step:
        return self.act(logits)

