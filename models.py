import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Simplified and regularized VAE for small datasets
class SSVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=32):
        super(SSVAE, self).__init__()
        
        # Encoder
        self.encoder_fc1 = nn.Linear(input_dim, hidden_dim)
        self.encoder_ln1 = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = F.relu(self.encoder_ln1(self.encoder_fc1(x)))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar  # no prediction head
        

# VAE Loss function (no supervised term anymore)
def loss_function(x_recon, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss

# Training loop (unsupervised)
def train_ssvae(model, data_loader, optimizer, epochs=50, beta=1.0, device='cuda'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, in data_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x_batch)
            loss = loss_function(x_recon, x_batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss/len(data_loader)}')
    return model

# Entrypoint function (same signature as before)
def run_ssvae(X_train, X_no_label, epochs=50, batch_size=64, latent_dim=16, dev="cpu",hidden_dim = 128, lr=1e-3, use_only_unlabeled = True):
    print(f"Using device: {dev}")

    from sklearn.preprocessing import StandardScaler

    # Combine and scale all data
    scaler = StandardScaler()
    if use_only_unlabeled:
        X_all = X_no_label
    else:
        X_all = np.vstack([X_train, X_no_label])
    X_all = scaler.fit_transform(X_all)

    # Use only X_no_label for training
    X_tensor = torch.FloatTensor(X_all[len(X_train):])

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = X_tensor.shape[1]
    model = SSVAE(input_dim, latent_dim, hidden_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Train unsupervised VAE
    model = train_ssvae(model, loader, optimizer, epochs=epochs, beta=1.0, device=dev)
    return model
