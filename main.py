import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val


class Encoder(nn.Module):
    def __init__(self, in_ch = 3, hid = 128, out_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 4, 2, 1),
            nn.BatchNorm2d(hid),
            nn.ReLU(True),
            nn.Conv2d(hid, hid * 2, 4, 2, 1),
            nn.BatchNorm2d(hid * 2),
            nn.ReLU(True),
            nn.Conv2d(hid * 2, hid * 4, 4, 2, 1),
            nn.BatchNorm2d(hid * 4),
            nn.ReLU(True),
            nn.Conv2d(hid * 4, hid * 4, 4, 2, 1),
            nn.BatchNorm2d(hid * 4),
            nn.ReLU(True),
        )
        self.proj = nn.Conv2d(hid * 4, out_dim, 1)

    def forward(self, x):
        z = self.proj(self.net(x))
        return F.normalize(z, p = 2, dim = 1)


class Decoder(nn.Module):
    def __init__(self, out_ch = 3, hid = 128, in_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, hid * 4, 4, 2, 1),
            nn.BatchNorm2d(hid * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hid * 4, hid * 4, 4, 2, 1),
            nn.BatchNorm2d(hid * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hid * 4, hid * 2, 4, 2, 1),
            nn.BatchNorm2d(hid * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hid * 2, hid, 4, 2, 1),
            nn.BatchNorm2d(hid),
            nn.ReLU(True),
            nn.Conv2d(hid, out_ch, 3, 1, 1),
        )

    def forward(self, z):
        return self.net(z)


class DCVQQuantizer(nn.Module):
    def __init__(self, embed_dim = 128, num_codes = 512, num_subspaces = 16):
        super().__init__()
        self.D = embed_dim
        self.N = num_subspaces
        self.ds = embed_dim // num_subspaces
        self.M = num_codes
        self.codebooks = nn.Parameter(torch.randn(num_subspaces, num_codes, self.ds))

    def forward(self, z):
        B, D, H, W = z.shape

        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.N, self.ds).permute(1, 0, 2)
        dists = torch.cdist(z_flat, self.codebooks)
        indices = torch.argmin(dists, dim = 2)

        idx_exp = indices.unsqueeze(-1).expand(-1, -1, self.ds)
        z_q = torch.gather(self.codebooks, 1, idx_exp)

        loss_vq = F.mse_loss(z_q, z_flat.detach())
        loss_commit = F.mse_loss(z_q.detach(), z_flat)

        z_q = z_flat + (z_q - z_flat).detach()
        z_q = z_q.permute(1, 0, 2).reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        return z_q, loss_vq, loss_commit, indices.permute(1, 0)

    def revive_dead_codes(self, z_e):
        z_flat = z_e.permute(0, 2, 3, 1).reshape(-1, self.N, self.ds).permute(1, 0, 2)
        dists = torch.cdist(z_flat, self.codebooks)
        indices = torch.argmin(dists, dim = 2)

        for i in range(self.N):
            used = torch.unique(indices[i])
            all_idx = torch.arange(self.M, device = z_e.device)
            dead = all_idx[~torch.isin(all_idx, used)]

            if len(dead) > 0:
                rand_idx = torch.randint(0, z_flat.size(1), (len(dead),), device = z_e.device)
                with torch.no_grad():
                    self.codebooks[i, dead] = z_flat[i, rand_idx].detach()


class DCVQVAE(nn.Module):
    def __init__(self, embed_dim = 128, hidden_dim = 128, num_codes = 512, num_subspaces = 32):
        super().__init__()
        self.encoder = Encoder(hid = hidden_dim, out_dim = embed_dim)
        self.decoder = Decoder(hid = hidden_dim, in_dim = embed_dim)
        self.vq = DCVQQuantizer(embed_dim, num_codes, num_subspaces)
        self.num_subspaces = num_subspaces

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, l_vq, l_com, idx = self.vq(z_e)
        x_r = self.decoder(z_q)
        return x_r, l_vq, l_com, idx, z_e

    def compute_usage(self, indices):
        return [indices[..., i].unique().numel() for i in range(self.num_subspaces)]


def build_geodesic_matrix(dataset, k = 10, batch_size = 128):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 4)

    all_feats = []
    for x, _, _ in loader:
        all_feats.append(x.reshape(x.size(0), -1).numpy())

    all_feats = np.concatenate(all_feats, axis = 0)
    n = len(all_feats)

    nn_model = NearestNeighbors(n_neighbors = k + 1, n_jobs = -1).fit(all_feats)
    dists, indices = nn_model.kneighbors(all_feats)

    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = dists[i, j_idx]
            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([d, d])

    graph = csr_matrix((vals, (rows, cols)), shape = (n, n))
    geo_dist = shortest_path(graph, method = "D", directed = False)
    geo_dist = np.nan_to_num(geo_dist, nan = 0.0, posinf = 0.0)

    return torch.tensor(geo_dist / (geo_dist.max() + 1e-8), dtype = torch.float32)


def isomap_loss(z_e, data_indices, geo_matrix, device, max_pairs = 512):
    z_pool = z_e.mean(dim = [2, 3])
    n = z_pool.size(0)
    if n < 2:
        return torch.tensor(0.0, device = device)

    num_pairs = min(max_pairs, n * (n - 1) // 2)
    idx_i = torch.randint(0, n, (num_pairs,))
    idx_j = torch.randint(0, n, (num_pairs,))

    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]
    if len(idx_i) == 0:
        return torch.tensor(0.0, device = device)

    geo_d = geo_matrix[data_indices[idx_i], data_indices[idx_j]].to(device)
    enc_d = torch.norm(z_pool[idx_i.to(device)] - z_pool[idx_j.to(device)], dim = -1)

    enc_d = enc_d / (enc_d.mean().detach() + 1e-8)
    geo_d = geo_d / (geo_d.mean().detach() + 1e-8)
    return F.mse_loss(enc_d, geo_d)


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


def save_recon_grid(model, dl, epoch, device, N = 3, out_dir = "recon_vis_vanilla"):
    model.eval()

    x, _, _ = next(iter(dl))
    x = x[:N].to(device)

    with torch.no_grad():
        x_r, _, _, _, _ = model(x)

    x_np = denorm(x.cpu()).permute(0, 2, 3, 1).numpy()
    r_np = denorm(x_r.cpu()).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(3, N, figsize = (2.8 * N, 8))
    for i in range(N):
        axes[0, i].imshow(x_np[i])
        axes[0, i].axis("off")

        axes[1, i].imshow(r_np[i])
        axes[1, i].axis("off")

        axes[2, i].imshow(((x_np[i] - r_np[i]) ** 2).mean(2), cmap = "hot")
        axes[2, i].axis("off")

    os.makedirs(out_dir, exist_ok = True)
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}.png", dpi = 150, bbox_inches = "tight")
    plt.close(fig)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim = 128
    hidden_dim = 128
    num_codes = 512
    num_subspaces = 16

    tf = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )

    raw = torch.utils.data.ConcatDataset(
        [
            torchvision.datasets.Flowers102(
                root = "./data",
                split = s,
                download = True,
                transform = tf,
            )
            for s in ["train", "val", "test"]
        ]
    )

    dataset = IndexedDataset(raw)
    geo_matrix = build_geodesic_matrix(dataset)

    loader = DataLoader(
        dataset,
        batch_size = 64,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
    )

    test_dataset = IndexedDataset(
        torchvision.datasets.Flowers102(
            root = "./data",
            split = "test",
            download = True,
            transform = tf,
        )
    )
    test_loader = DataLoader(test_dataset, batch_size = 64, shuffle = False, num_workers = 4)

    model = DCVQVAE(
        embed_dim = embed_dim,
        hidden_dim = hidden_dim,
        num_codes = num_codes,
        num_subspaces = num_subspaces,
    ).to(device)

    opt_enc = torch.optim.Adam(model.encoder.parameters(), lr = 1e-4)
    opt_vq = torch.optim.Adam(model.vq.parameters(), lr = 1e-3)
    opt_dec = torch.optim.Adam(model.decoder.parameters(), lr = 5e-4)

    for epoch in range(50):
        model.train()

        tot_r = 0.0
        tot_iso = 0.0
        tot_vq = 0.0
        n = 0.0
        used_codes = [set() for _ in range(num_subspaces)]

        for x, _, idx in loader:
            x = x.to(device)

            set_requires_grad(model.decoder, False)
            set_requires_grad(model.vq, False)
            set_requires_grad(model.encoder, True)

            x_r, _, _, _, z_e = model(x)
            recon = F.mse_loss(x_r, x)
            iso = isomap_loss(z_e, idx, geo_matrix, device)
            loss_geo = recon + iso

            opt_enc.zero_grad()
            loss_geo.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt_enc.step()

            set_requires_grad(model.encoder, False)
            set_requires_grad(model.vq, True)

            _, l_vq, _, indices, _ = model(x)

            opt_vq.zero_grad()
            l_vq.backward()
            torch.nn.utils.clip_grad_norm_(model.vq.parameters(), 1.0)
            opt_vq.step()

            set_requires_grad(model.vq, False)
            set_requires_grad(model.decoder, True)

            x_r, _, _, _, _ = model(x)
            loss_dec = F.mse_loss(x_r, x)

            opt_dec.zero_grad()
            loss_dec.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), 1.0)
            opt_dec.step()

            with torch.no_grad():
                for s in range(num_subspaces):
                    used_codes[s].update(indices[:, s].cpu().tolist())

            tot_r += loss_dec.item()
            tot_iso += iso.item()
            tot_vq += l_vq.item()
            n += 1.0

        with torch.no_grad():
            x0 = next(iter(loader))[0].to(device)
            _, _, _, _, z_e = model(x0)
            model.vq.revive_dead_codes(z_e)

        usage = [len(s) for s in used_codes]
        avg_usage = sum(usage) / len(usage)
        print(
            f"Ep {epoch + 1:3d} | R: {tot_r / n:.4f} | I: {tot_iso / n:.4f} | VQ: {tot_vq / n:.4f} | "
            f"CB: {avg_usage:.0f}/{num_codes} avg  min {min(usage)}  max {max(usage)}"
        )

        if (epoch + 1) % 5 == 0:
            save_recon_grid(model, test_loader, epoch + 1, device)

    torch.save(model.state_dict(), "dcvq_vanilla.pt")


if __name__ == "__main__":
    train()