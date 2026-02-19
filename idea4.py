import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np


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

    def forward(self, x):
        return self.proj(self.net(x))


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

    def forward(self, z):
        return self.net(z)


class VQQuantizer(nn.Module):
    def __init__(self, num_codes=512, embed_dim=256, beta=0.25):
        super().__init__()
        self.M = num_codes
        self.D = embed_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_codes, embed_dim)
        self.codebook.weight.data.uniform_(-1.0/num_codes, 1.0/num_codes)

    def forward(self, z):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        dist = torch.cdist(z_flat.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = dist.argmin(dim=-1)
        q = self.codebook(indices)
        vq_loss = F.mse_loss(z_flat.detach(), q) + self.beta * F.mse_loss(z_flat, q.detach())
        q_st = z_flat + (q - z_flat).detach()
        z_q = q_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return z_q, vq_loss, indices.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256, num_codes=512):
        super().__init__()
        self.encoder = Encoder(hid=hidden_dim, out_dim=embed_dim)
        self.decoder = Decoder(hid=hidden_dim, in_dim=embed_dim)
        self.vq = VQQuantizer(num_codes, embed_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

FLOWERS102_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colts foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
    "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers",
    "stemless gentian", "artichoke", "sweet william", "carnation",
    "garden phlox", "love in the mist", "mexican aster", "alpine sea holly",
    "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip",
    "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy",
    "common dandelion", "petunia", "wild pansy", "primula", "sunflower",
    "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia",
    "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum",
    "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania",
    "azalea", "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium", "frangipani",
    "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]


def save_recon_grid(model, dl, epoch, device, N=3, out_dir="recon_vis_vqvae"):
    model.eval()
    x, labels = next(iter(dl))
    x = x[:N].to(device)
    labels = labels[:N]
    with torch.no_grad():
        x_r, _, indices = model(x)
        usage = indices.unique().numel()
    x_np = denorm(x.cpu()).permute(0, 2, 3, 1).numpy()
    r_np = denorm(x_r.cpu()).permute(0, 2, 3, 1).numpy()
    fig, axes = plt.subplots(3, N, figsize=(2.8*N, 8))
    fig.suptitle(
        f'VQVAE Epoch {epoch} | Recon: {F.mse_loss(x_r, x).item():.4f} | '
        f'Batch usage: {usage}/{model.vq.M}',
        fontsize=11, fontweight='bold'
    )
    for i in range(N):
        axes[0, i].imshow(x_np[i]); axes[0, i].axis('off')
        lbl = int(labels[i])
        name = FLOWERS102_NAMES[lbl] if lbl < len(FLOWERS102_NAMES) else str(lbl)
        axes[0, i].set_title(name[:20], fontsize=8)
        axes[1, i].imshow(r_np[i]); axes[1, i].axis('off')
        mse = ((x_np[i] - r_np[i])**2).mean()
        axes[1, i].set_title(f'MSE:{mse:.4f}', fontsize=8)
        diff = ((x_np[i] - r_np[i])**2).mean(axis=2)
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.15); axes[2, i].axis('off')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_dim, hidden_dim, num_codes = 256, 256, 512
    batch_size, num_epochs, lr = 64, 50, 3e-4

    print(f"VQ-VAE | Flowers-102 @ 64x64 | codebook: {num_codes} x {embed_dim}d")
    print(f"Device: {device}")

    tf_train = transforms.Compose([
        transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tf_test = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    train_dataset = torch.utils.data.ConcatDataset([
        torchvision.datasets.Flowers102(root="./data", split=s, download=True, transform=tf_train)
        for s in ["train", "val", "test"]
    ])
    test_dataset = torchvision.datasets.Flowers102(root="./data", split="test", download=True, transform=tf_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

    model = VQVAE(embed_dim=embed_dim, hidden_dim=hidden_dim, num_codes=num_codes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*70)

    for epoch in range(num_epochs):
        model.train()
        tot_r = tot_vq = 0.0
        n = 0
        for x, _ in train_loader:
            x = x.to(device)
            x_recon, vq_loss, indices = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot_r += recon_loss.item()
            tot_vq += vq_loss.item()
            n += 1
        scheduler.step()

        model.eval()
        all_idx = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                _, _, idx = model(x)
                all_idx.append(idx.cpu())
        usage = torch.cat(all_idx).unique().numel()

        print(f"Epoch [{epoch+1:3d}/{num_epochs}]  Recon: {tot_r/n:.4f}  VQ: {tot_vq/n:.4f}  Usage: {usage}/{num_codes}")

        if (epoch + 1) % 5 == 0:
            val = m = 0
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
    test_r = m = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_r, _, _ = model(x)
            test_r += F.mse_loss(x_r, x).item()
            m += 1
    print(f"Final Test Recon: {test_r/m:.4f}")
    torch.save(model.state_dict(), "vqvae_flowers102.pt")
    print("Saved vqvae_flowers102.pt")


if __name__ == "__main__":
    train()