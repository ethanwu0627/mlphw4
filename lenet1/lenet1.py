import torch
import torch.nn as nn

class TanhScaled(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)

# FIX HERE LATER***
def load_digit_codes():
    mu = torch.zeros(10, 84)
    return mu

class LeNet5(nn.Module):
    def __init__(self, mu_codes):
        super().__init__()

        self.tanh = TanhScaled()

        # C1: Conv (1x32x32 to 6x28x28)
        self.C1 = nn.Conv2d(1, 6, kernel_size = 5)

        # S2: Subsampling (AvgPool + learnable a,b)
        self.S2_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.S2_a = nn.Parameter(torch.ones(6))
        self.S2_b = nn.Parameter(torch.zeros(6))

        # C3: Conv (6x14x14 to 16x10x10)
        self.C3 = nn.Conv2d(6, 16, kernel_size=5)

        # S4: Subsampling (AvgPool + learnable a,b)
        self.S4_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.S4_a = nn.Parameter(torch.ones(16))
        self.S4_b = nn.Parameter(torch.zeros(16))

        # C5: Conv (16x5x5 to 120x1x1)
        self.C5 = nn.Conv2d(16, 120, kernel_size=5)

        # F6: Fully connected (120 to 84)
        self.F6 = nn.Linear(120, 84)

        # RBF output layer using DIGIT bitmap codes
        # mu_codes: shape (10, 84)
        self.mu = nn.Parameter(mu_codes, requires_grad=False)



    #  Forward pass: returns 10 squared Euclidean distances
    def forward(self, x):

        # C1
        x = self.tanh(self.C1(x))

        # S2
        x = self.S2_pool(x)
        x = self.tanh(self.S2_a.view(1,6,1,1) * x + self.S2_b.view(1,6,1,1))

        # C3
        x = self.tanh(self.C3(x))

        # S4
        x = self.S4_pool(x)
        x = self.tanh(self.S4_a.view(1,16,1,1) * x + self.S4_b.view(1,16,1,1))

        # C5
        x = self.tanh(self.C5(x))
        x = torch.flatten(x, start_dim=1)

        # F6
        h = self.tanh(self.F6(x))

        # RBF output: squared Euclidean distances to 10 code vectors
        # h: (batch, 84)
        # mu: (10, 84)
        # output distances: (batch, 10)
        D = torch.sum((h.unsqueeze(1) - self.mu.unsqueeze(0)) ** 2, dim=2)

        return D

def build_lenet5():
    mu_codes = load_digit_codes()
    return LeNet5(mu_codes)
