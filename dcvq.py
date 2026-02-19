import os, math, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, in_ch=3, hid=256, out_dim=256):
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
    def __init__(self, out_ch=3, hid=256, in_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_dim, hid*4, 4, 2, 1), nn.BatchNorm2d(hid*4), nn.ReLU(True),
            nn.ConvTranspose2d(hid*4, hid*4, 4, 2, 1), nn.BatchNorm2d(hid*4), nn.ReLU(True),
            nn.ConvTranspose2d(hid*4, hid*2, 4, 2, 1), nn.BatchNorm2d(hid*2), nn.ReLU(True),
            nn.ConvTranspose2d(hid*2, hid, 4, 2, 1), nn.BatchNorm2d(hid), nn.ReLU(True),
            nn.Conv2d(hid, out_ch, 3, 1, 1),
        )
    def forward(self, z): return self.net(z)

class DCVQQuantizer(nn.Module):
    def __init__(self, embed_dim=256, num_codes=512, num_subspaces=32, beta=0.25):
        super().__init__()
        assert embed_dim % num_subspaces == 0
        self.D, self.N, self.ds, self.M, self.beta = embed_dim, num_subspaces, embed_dim // num_subspaces, num_codes, beta
        self.codebooks = nn.ParameterList([
            nn.Parameter(torch.randn(num_codes, self.ds) * 0.02)
            for _ in range(self.N)
        ])

    def _nearest(self, x, cb):
        dist = torch.cdist(x.unsqueeze(0), cb.unsqueeze(0)).squeeze(0)
        idx = dist.argmin(-1)
        return idx, cb[idx]

    def forward(self, z):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        chunks = z_flat.chunk(self.N, dim=-1)
        q_parts, idx_parts, vq = [], [], 0.0
        for s, cb in zip(chunks, self.codebooks):
            idx, q = self._nearest(s, cb)
            vq = vq + F.mse_loss(s.detach(), q) + self.beta * F.mse_loss(s, q.detach())
            q_st = s + (q - s).detach()
            q_parts.append(q_st)
            idx_parts.append(idx)
        z_q = torch.cat(q_parts, dim=-1)
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        indices = torch.stack([t.view(B, H, W) for t in idx_parts], 1)
        return z_q, vq / self.N, indices

class DCVQVAE(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256, num_codes=512, num_subspaces=32):
        super().__init__()
        self.encoder = Encoder(hid=hidden_dim, out_dim=embed_dim)
        self.decoder = Decoder(hid=hidden_dim, in_dim=embed_dim)
        self.vq = DCVQQuantizer(embed_dim, num_codes, num_subspaces)
        self.num_subspaces = num_subspaces

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_r = self.decoder(z_q)
        return x_r, vq_loss, indices

    def compute_usage(self, indices):
        return [indices[:, i].unique().numel() for i in range(self.num_subspaces)]

def denorm(x): return (x * 0.5 + 0.5).clamp(0, 1)

def save_recon_grid(model, dl, epoch, device, N=3, out_dir="recon_vis_dcvq"):
    model.eval()
    x, y = next(iter(dl))
    x, y = x[:N].to(device), y[:N]
    with torch.no_grad(): x_r, _, indices = model(x)
    x_np = denorm(x.cpu()).permute(0, 2, 3, 1).numpy()
    r_np = denorm(x_r.cpu()).permute(0, 2, 3, 1).numpy()
    usages = model.compute_usage(indices)
    fig, axes = plt.subplots(3, N, figsize=(2.8*N, 8))
    fig.suptitle(f'DCVQ Epoch {epoch} | Recon {F.mse_loss(x_r, x).item():.4f} | Usage {np.mean(usages):.0f}/{model.vq.M}', fontsize=11, fontweight='bold')
    for i in range(N):
        axes[0, i].imshow(x_np[i]); axes[0, i].axis('off'); axes[0, i].set_title(str(int(y[i])), fontsize=9)
        axes[1, i].imshow(r_np[i]); axes[1, i].axis('off'); axes[1, i].set_title(f'MSE:{((x_np[i]-r_np[i])**2).mean():.4f}', fontsize=9)
        diff = ((x_np[i]-r_np[i])**2).mean(axis=2)
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.15); axes[2, i].axis('off')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim, hidden_dim, num_codes, num_subspaces = 256, 256, 512, 32
    batch_size, num_epochs, lr = 64, 50, 3e-4
    print(f"DCVQ-VAE | Flowers-102 @ 64x64 | {num_subspaces}x{embed_dim//num_subspaces}d | {num_codes} codes | {device}")

    tf_train = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
    tf_test  = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])

    train_dataset = torch.utils.data.ConcatDataset([
        torchvision.datasets.Flowers102(root="./data", split=s, download=True, transform=tf_train)
        for s in ["train", "val", "test"]
    ])
    test_dataset = torchvision.datasets.Flowers102(root="./data", split="test", download=True, transform=tf_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DCVQVAE(embed_dim=embed_dim, hidden_dim=hidden_dim, num_codes=num_codes, num_subspaces=num_subspaces).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print("="*70)

    for epoch in range(num_epochs):
        model.train()
        tot_r = tot_vq = 0.0
        n = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_r, vq_loss, _ = model(x)
            r_loss = F.mse_loss(x_r, x)
            loss = r_loss + vq_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_r += r_loss.item()
            tot_vq += vq_loss.item()
            n += 1
        sch.step()

        usages = []
        model.eval()
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                _, _, idx = model(x)
                usages.append(model.compute_usage(idx))
        avg_usage = np.mean([np.mean(u) for u in usages])

        print(f"Epoch [{epoch+1:3d}/{num_epochs}]  Recon: {tot_r/n:.4f}  VQ: {tot_vq/n:.4f}  Usage: {avg_usage:.0f}/{num_codes}")

        if (epoch + 1) % 5 == 0:
            val = 0.0
            m = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    x_r, _, _ = model(x)
                    val += F.mse_loss(x_r, x).item()
                    m += 1
            print(f"  >> Val Recon: {val/m:.4f}")
            save_recon_grid(model, test_loader, epoch+1, device)
            print()

    print("="*70)
    model.eval()
    test_r = 0.0
    m = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_r, _, _ = model(x)
            test_r += F.mse_loss(x_r, x).item()
            m += 1
    print(f"Final Test Recon: {test_r/m:.4f}")
    torch.save(model.state_dict(), "dcvq_flowers102.pt")
    print("Saved dcvq_flowers102.pt")

if __name__ == "__main__":
    train()