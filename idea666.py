import os, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class IndexedDataset(Dataset):
    def __init__(self, dataset): self.dataset = dataset
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx

class Encoder(nn.Module):
    def __init__(self, in_ch=3, hid=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, 4, 2, 1), nn.BatchNorm2d(hid), nn.ReLU(True),
            nn.Conv2d(hid, hid*2, 4, 2, 1), nn.BatchNorm2d(hid*2), nn.ReLU(True),
            nn.Conv2d(hid*2, hid*4, 4, 2, 1), nn.BatchNorm2d(hid*4), nn.ReLU(True),
            nn.Conv2d(hid*4, hid*4, 4, 2, 1), nn.BatchNorm2d(hid*4), nn.ReLU(True),
        )
        self.proj = nn.Conv2d(hid*4, out_dim, 1)
    def forward(self, x): return self.proj(self.net(x))

class Decoder(nn.Module):
    def __init__(self, out_ch=3, hid=128, in_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, hid*4, 4, 2, 1), nn.BatchNorm2d(hid*4), nn.ReLU(True),
            nn.ConvTranspose2d(hid*4, hid*4, 4, 2, 1), nn.BatchNorm2d(hid*4), nn.ReLU(True),
            nn.ConvTranspose2d(hid*4, hid*2, 4, 2, 1), nn.BatchNorm2d(hid*2), nn.ReLU(True),
            nn.ConvTranspose2d(hid*2, hid, 4, 2, 1), nn.BatchNorm2d(hid), nn.ReLU(True),
            nn.Conv2d(hid, out_ch, 3, 1, 1),
        )
    def forward(self, z): return self.net(z)

class DCVQQuantizerEMA(nn.Module):
    def __init__(self, embed_dim=128, num_codes=512, num_subspaces=16, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        assert embed_dim % num_subspaces == 0
        self.D, self.N, self.ds, self.M = embed_dim, num_subspaces, embed_dim // num_subspaces, num_codes
        self.beta, self.decay, self.eps = beta, decay, eps
        init_bound = 1.0 / num_codes
        embedding = torch.rand(self.N, num_codes, self.ds) * 2 * init_bound - init_bound
        self.register_buffer("codebooks", embedding)
        self.register_buffer("ema_cluster_size", torch.zeros(self.N, num_codes))
        self.register_buffer("ema_w", embedding.clone())

    def forward(self, z):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, self.N, self.ds)
        z_sq = torch.sum(z_flat ** 2, dim=2, keepdim=True)
        cb_sq = torch.sum(self.codebooks ** 2, dim=2).unsqueeze(0)
        interaction = torch.einsum('bnd,nmd->bnm', z_flat, self.codebooks)
        dists = z_sq + cb_sq - 2 * interaction
        indices = torch.argmin(dists, dim=2)

        idx_t = indices.permute(1, 0)
        z_q_gathered = torch.gather(self.codebooks, 1, idx_t.unsqueeze(-1).expand(-1, -1, self.ds))
        z_q_flat = z_q_gathered.permute(1, 0, 2)
        loss = self.beta * F.mse_loss(z_q_flat.detach(), z_flat)

        if self.training:
            encodings = F.one_hot(indices, self.M).float()
            self.ema_cluster_size.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
            n_sum = self.ema_cluster_size.sum(1, keepdim=True)
            cs = (self.ema_cluster_size + self.eps) / (n_sum + self.M * self.eps) * n_sum
            dw = torch.einsum('bnm,bnd->nmd', encodings, z_flat.detach())
            self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
            self.codebooks.copy_(self.ema_w / cs.unsqueeze(-1))

        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return z_q, loss, indices.reshape(B, H, W, self.N)

class DCVQVAE(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128, num_codes=512, num_subspaces=16):
        super().__init__()
        self.encoder = Encoder(hid=hidden_dim, out_dim=embed_dim)
        self.decoder = Decoder(hid=hidden_dim, in_dim=embed_dim)
        self.vq = DCVQQuantizerEMA(embed_dim, num_codes, num_subspaces)
        self.num_subspaces = num_subspaces
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_r = self.decoder(z_q)
        return x_r, vq_loss, indices, z_e
    def compute_usage(self, indices):
        return [indices[..., i].unique().numel() for i in range(self.num_subspaces)]

def build_geodesic_matrix(dataset, k=10, batch_size=128):
    print("Building geodesic distance matrix...")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_feats = []
    for x, _, _ in loader:
        all_feats.append(x.reshape(x.size(0), -1).numpy())
    all_feats = np.concatenate(all_feats, axis=0)
    n = len(all_feats)
    print(f"  {n} images, flattened to {all_feats.shape[1]}d")
    nn_model = NearestNeighbors(n_neighbors=k+1, algorithm='auto', n_jobs=-1).fit(all_feats)
    dists, indices = nn_model.kneighbors(all_feats)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx in range(1, k+1):
            j, d = indices[i, j_idx], dists[i, j_idx]
            rows.extend([i, j]); cols.extend([j, i]); vals.extend([d, d])
    graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
    print("  Computing shortest paths (Dijkstra)...")
    geo_dist = shortest_path(graph, method='D', directed=False)
    geo_dist = np.nan_to_num(geo_dist, nan=0.0, posinf=0.0)
    max_val = geo_dist.max() if geo_dist.max() > 0 else 1.0
    print(f"  Done. Matrix shape: {geo_dist.shape}")
    return torch.tensor(geo_dist / max_val, dtype=torch.float32)

def isomap_loss(z_e, data_indices, geo_matrix, device, max_pairs=512):
    B, D, H, W = z_e.shape
    z_pool = z_e.mean(dim=[2, 3])
    n = z_pool.size(0)
    if n < 2: return torch.tensor(0.0, device=device)
    num_pairs = min(max_pairs, n * (n - 1) // 2)
    idx_i = torch.randint(0, n, (num_pairs,))
    idx_j = torch.randint(0, n, (num_pairs,))
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if len(idx_i) == 0: return torch.tensor(0.0, device=device)
    geo_d = geo_matrix[data_indices[idx_i], data_indices[idx_j]].to(device)
    enc_d = torch.norm(z_pool[idx_i.to(device)] - z_pool[idx_j.to(device)], dim=-1)
    enc_d = enc_d / (enc_d.max().detach() + 1e-8)
    return F.mse_loss(enc_d, geo_d)

def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)
def save_recon_grid(model, dl, epoch, device, N=3, out_dir="recon_vis_dcvq_iso"):
    model.eval()
    x, y, _ = next(iter(dl))
    x = x[:N].to(device)
    with torch.no_grad(): x_r, _, indices, _ = model(x)
    x_np, r_np = denorm(x.cpu()).permute(0,2,3,1).numpy(), denorm(x_r.cpu()).permute(0,2,3,1).numpy()
    usages = model.compute_usage(indices)
    fig, axes = plt.subplots(3, N, figsize=(2.8*N, 8))
    fig.suptitle(f'Epoch {epoch} | Recon {F.mse_loss(x_r, x).item():.4f} | Usage {np.mean(usages):.0f}/{model.vq.M}')
    for i in range(N):
        axes[0,i].imshow(x_np[i]); axes[0,i].axis('off')
        axes[1,i].imshow(r_np[i]); axes[1,i].axis('off')
        axes[2,i].imshow(((x_np[i]-r_np[i])**2).mean(2), cmap='hot', vmin=0, vmax=0.15); axes[2,i].axis('off')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/epoch_{epoch:03d}.png', dpi=150, bbox_inches='tight'); plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim, hidden_dim, num_codes, num_subspaces = 128, 128, 512, 16
    iso_w = 0.01
    print(f"DCVQ+Iso (EMA) | 16x8d | 512 codes | iso_w={iso_w} | {device}")

    tf_train = transforms.Compose([transforms.Resize((64,64)), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])
    tf_test = transforms.Compose([transforms.Resize((64,64)),
                                  transforms.ToTensor(), transforms.Normalize((0.5,)*3,(0.5,)*3)])

    raw = torch.utils.data.ConcatDataset([
        torchvision.datasets.Flowers102(root="./data", split=s, download=True, transform=tf_train)
        for s in ["train","val","test"]])
    raw_test = torchvision.datasets.Flowers102(root="./data", split="test", download=True, transform=tf_test)

    dataset = IndexedDataset(raw)
    geo_matrix = build_geodesic_matrix(dataset)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(IndexedDataset(raw_test), batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = DCVQVAE(embed_dim, hidden_dim, num_codes, num_subspaces).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 70)

    for epoch in range(50):
        model.train()
        tot_r, tot_vq, tot_iso, n = 0., 0., 0., 0
        for x, _, idx in loader:
            x = x.to(device)
            x_r, vq_loss, last_idx, z_e = model(x)
            r_loss = F.mse_loss(x_r, x)
            iso_loss = isomap_loss(z_e, idx, geo_matrix, device)
            loss = r_loss + vq_loss + iso_w * iso_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_r += r_loss.item(); tot_vq += vq_loss.item(); tot_iso += iso_loss.item(); n += 1
        sch.step()

        usage = np.mean(model.compute_usage(last_idx))
        print(f"Epoch [{epoch+1:3d}/50]  Recon: {tot_r/n:.4f}  VQ: {tot_vq/n:.4f}  Iso: {tot_iso/n:.4f}  Usage: ~{usage:.0f}/512")

        if (epoch+1) % 5 == 0:
            model.eval()
            val_r, m = 0., 0
            with torch.no_grad():
                for x, _, _ in test_loader:
                    x = x.to(device)
                    x_r, _, _, _ = model(x)
                    val_r += F.mse_loss(x_r, x).item(); m += 1
            print(f"  >> Val Recon: {val_r/m:.4f}")
            save_recon_grid(model, test_loader, epoch+1, device)

    print("=" * 70)
    torch.save(model.state_dict(), "dcvq_iso_ema.pt")
    print("Saved dcvq_iso_ema.pt")

if __name__ == "__main__": train()