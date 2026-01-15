import torch
import torch.nn as nn
import torch.nn.functional as F



def make_fc(dims):
    """Fully-connected NN with BatchNorm + Softplus activations except last layer (no activation)."""
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.Softplus())
    # Exclude the last Softplus for final output layer
    return nn.Sequential(*layers[:-1])


def make_fc_wo_batch_norm(dims):
    """Fully-connected NN with Softplus activations, no BatchNorm, except last layer (no activation)."""
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Softplus())
    return nn.Sequential(*layers[:-1])

def make_f(dims, af):
    """Fully-connected NN with user-supplied activation after every Linear."""
    layers = []
    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(af)
    return nn.Sequential(*layers)  # Keep final nonlinearity for flexibility

def split_in_half(t):
    """
    Splits last dimension of tensor in half (e.g., for mean/scale outputs).
    """
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)



class StateDecoder(nn.Module):
    """Decoder: p(z|B)"""
    def __init__(self, hidden_dims, latent_dim=50, state_dim=2):
        super().__init__()
        dims = [state_dim] + hidden_dims + [2 * latent_dim]
        self.fc = make_fc_wo_batch_norm(dims)
        self.softplus = nn.Softplus()

    def forward(self, x):
        hidden = self.fc(x)
        hidden = hidden.reshape(x.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = self.softplus(scale)
        return loc, scale

class ZDecoder(nn.Module):
    """Decoder: p(x|z) — outputs gate logits and normalized mu for ZINB."""
    def __init__(self, hidden_dims, num_genes, latent_dim):
        super().__init__()
        dims = [latent_dim] + hidden_dims + [2 * num_genes]
        self.fc = make_fc(dims)

    def forward(self, z):
        gate, mu = split_in_half(self.fc(z))
        mu = F.softmax(mu, dim=-1)
        return gate, mu
    

class XEncoder(nn.Module):
    """Encoder: p(z|x) — outputs (loc, scale) for z."""
    def __init__(self, hidden_dims, latent_dim, num_genes):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2 * latent_dim]
        self.fc = make_fc(dims)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = x.float()
        hidden = self.fc(x)
        hidden = hidden.reshape(x.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = self.softplus(scale)
        return loc, scale
    
class ZEncoder(nn.Module):
    """Cell state encoder: p(B|z)."""
    def __init__(self, hidden_dims, latent_dim=50, state_dim=2):
        super().__init__()
        dims = [latent_dim] + hidden_dims + [2 * state_dim]
        self.fc = make_fc_wo_batch_norm(dims)
        self.softplus = nn.Softplus()

    def forward(self, z):
        hidden = self.fc(z)
        hidden = hidden.reshape(z.shape[:-1] + hidden.shape[-1:])
        loc, scale = split_in_half(hidden)
        scale = self.softplus(scale)
        return loc, scale
    
class LEncoder(nn.Module):
    """Library size encoder: q(xl | x)."""
    def __init__(self, hidden_dims, num_genes):
        super().__init__()
        dims = [num_genes] + hidden_dims + [2]
        self.fc = make_fc(dims)

    def forward(self, s):
        l_loc, l_scale = split_in_half(self.fc(s))
        l_scale = F.softplus(l_scale)
        return l_loc, l_scale
    
class PotentialNet(nn.Module):
    """Potential energy neural net."""
    def __init__(self, hidden_dims, latent_dim, af):
        super().__init__()
        dims = [latent_dim] + hidden_dims + [1]
        self.fc = make_f(dims, af)
        self.lin = nn.Linear(latent_dim, 1)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.sigmoid = nn.Sigmoid()
        self.latent_dim = latent_dim

    def forward(self, x):
        gate = self.sigmoid(self.gate)
        out = gate * self.fc(x)
        return out
    
    
class GradientNet(nn.Module):
    """Potential gradient (for ODE)."""
    def __init__(self, potential):
        super().__init__()
        self.potential = potential

    def forward(self, t, x):
        if not x.requires_grad:
            x = x.requires_grad_(True)
        potential = self.potential(x)
        grad = torch.autograd.grad(potential, x, grad_outputs=torch.ones_like(potential), create_graph=True)[0]
        return -grad
