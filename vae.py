"""Generative Core: Variational Autoencoder in PyTorch."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """Comprime il vettore dei dati in media e log-varianza dello spazio latente."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


def reparameterize(mu, logvar):
    """z = mu + sigma * epsilon, con epsilon ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Decoder(nn.Module):
    """Ricostruisce il vettore originale a partire da un punto dello spazio latente."""

    def __init__(self, latent_dim, hidden_dim, output_dim, n_numeric, cat_groups):
        super().__init__()
        self.n_numeric = n_numeric
        self.cat_groups = cat_groups
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z):
        out = self.net(z)
        parts = [torch.sigmoid(out[:, :self.n_numeric])]
        start = self.n_numeric
        for size in self.cat_groups:
            parts.append(F.softmax(out[:, start:start + size], dim=1))
            start += size
        return torch.cat(parts, dim=1)


class VAE(nn.Module):
    def __init__(self, input_dim, n_numeric, cat_groups, hidden_dim=128, latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_numeric, cat_groups)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(x_recon, x, mu, logvar, beta=0.005):
    """Loss = Reconstruction Loss (MSE) + beta * KL Divergence."""
    batch_size = x.size(0)
    recon_loss = F.mse_loss(x_recon, x, reduction="sum") / batch_size
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return recon_loss + beta * kl_div, recon_loss, kl_div


def train_vae(model, X, epochs=500, batch_size=64, lr=1e-3, beta=0.005, seed=42, callback=None):
    """Addestra il VAE. callback(epoca, history) viene chiamata a ogni epoca (per mostrare l'avanzamento)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"total": [], "recon": [], "kl": []}
    n_samples = len(loader.dataset)

    for epoch in range(1, epochs + 1):
        model.train()
        tot = rec = kl = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            x_recon, mu, logvar = model(batch)
            loss, r, k = vae_loss(x_recon, batch, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item() * batch.size(0)
            rec += r.item() * batch.size(0)
            kl += k.item() * batch.size(0)
        history["total"].append(tot / n_samples)
        history["recon"].append(rec / n_samples)
        history["kl"].append(kl / n_samples)
        if callback:
            callback(epoch, history)
    return history


def estimate_noise_std(model, X, n_numeric):
    """Deviazione standard dell'errore di ricostruzione per ogni variabile numerica."""
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        mu, _ = model.encoder(X_t)
        X_recon = model.decoder(mu).cpu().numpy()
    return (X[:, :n_numeric] - X_recon[:, :n_numeric]).std(axis=0)


def generate_synthetic(model, engine, n_samples, noise_std=None, seed=None):
    """Genera dati sintetici usando SOLO il decoder: z ~ N(0, I) -> decoder -> p(x|z)."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim, device=device)
        x_synth = model.decoder(z).cpu().numpy()
    if noise_std is not None:
        n_num = len(noise_std)
        rng = np.random.default_rng(seed)
        x_synth[:, :n_num] += rng.normal(0, 1, (n_samples, n_num)) * noise_std
    return engine.inverse_transform(x_synth)
