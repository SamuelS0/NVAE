import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, λ):
        ctx.λ = λ
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        # multiply incoming gradient by -λ
        return grad_output.neg() * ctx.λ, None

def grad_reverse(x, λ=1.0):
    return GradReverse.apply(x, λ)

class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_r_classes, hidden_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_r_classes),  # Output num_r_classes instead of 2
        )

    def forward(self, features, λ):
        # reverse gradient here
        reversed_features = grad_reverse(features, λ)
        return self.net(reversed_features)


class DANN(nn.Module):
    def __init__(self, num_y_classes, num_r_classes, z_dim):
        super(DANN, self).__init__()
        self.num_y_classes = num_y_classes
        self.num_r_classes = num_r_classes
        self.z_dim = z_dim

        # Feature extractor matching VAE encoder architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),     #(N, 32, 28, 28) 
            nn.ReLU(),
            nn.MaxPool2d(2),                                                   #(N, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  #(N, 64, 14, 14) 
            nn.ReLU(),
            nn.MaxPool2d(2),                                                   #(N, 64, 7, 7)
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, z_dim)  # Project to z_dim directly
        )

        self.domain_discriminator = DomainDiscriminator(z_dim, num_r_classes)
        self.classifier = nn.Linear(z_dim, num_y_classes)

    def forward(self, x, y, r):
        features = self.feature_extractor(x)
        reversed_features = grad_reverse(features)
        domain_predictions = self.domain_discriminator(reversed_features, 1.0)
        y_predictions = self.classifier(features)

        return y_predictions, domain_predictions
