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


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, num_features=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
        )
        self.proj = nn.Conv2d(hidden_dim * 4, num_features, 1)

    def forward(self, x):
        return self.proj(self.net(x))


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=128, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, hidden_dim * 4, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, out_channels, 3, 1, 1),
        )

    def forward(self, z):
        return self.net(z)


class BinaryVQVAE(nn.Module):
    def __init__(self, num_features=256, embed_dim=128, hidden_dim=128, in_channels=3, hierarchical_alpha=0.0):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.hierarchical_alpha = hierarchical_alpha

        self.encoder = Encoder(in_channels, hidden_dim, num_features)
        self.decoder = Decoder(out_channels=in_channels, hidden_dim=hidden_dim, embed_dim=embed_dim)

        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
        self.register_buffer(
            "dim_weights",
            1.0 + hierarchical_alpha * torch.arange(num_features).float() / num_features
        )

    def gumbel_sigmoid(self, logits, tau):
        if self.training:
            g1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            g2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g1 - g2) / tau)
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft  # STE
        else:
            return (torch.sigmoid(logits) > 0.5).float()

    def encode(self, x, tau=1.0):
        logits = self.encoder(x)
        binary = self.gumbel_sigmoid(logits, tau)
        return logits, binary

    def decode(self, binary):
        B, D, H, W = binary.shape
        b_flat = binary.permute(0, 2, 3, 1).reshape(-1, D)
        z_flat = b_flat @ self.feature_embeddings
        z = z_flat.view(B, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()
        x_recon = self.decoder(z)
        return x_recon, z

    def forward(self, x, tau = 1.0):
        logits, binary = self.encode(x, tau)
        x_recon, z = self.decode(binary)
        return x_recon, logits, binary, z

    def compute_loss(self, x, x_recon, binary, diversity_w = 0.01):
        recon_loss = F.mse_loss(x_recon, x)

        p = binary.mean(dim=[0, 2, 3])  # (D,)
        eps = 1e-7
        p = p.clamp(eps, 1 - eps)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        diversity_loss = -entropy

        if self.hierarchical_alpha > 0:
            soft = torch.sigmoid(self.encoder(x))
            dim_w = self.dim_weights.view(1, -1, 1, 1)
            hier_loss = (dim_w * (soft - binary).pow(2)).mean()
        else:
            hier_loss = torch.tensor(0.0, device=x.device)

        total_loss = recon_loss + diversity_w * diversity_loss + 0.1 * hier_loss
        return total_loss, recon_loss, diversity_loss, hier_loss

    def compute_perplexity(self, binary):
        B, D, H, W = binary.shape
        p = binary.permute(0, 2, 3, 1).reshape(-1, D).mean(dim=0)
        eps = 1e-7
        p = p.clamp(eps, 1 - eps)

        per_dim_entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        avg_entropy = per_dim_entropy.mean()
        perplexity = torch.exp(avg_entropy)

        active_dims = ((p > 0.1) & (p < 0.9)).sum().item()
        avg_active = binary.sum(dim=1).mean().item()
        return perplexity.item(), active_dims, avg_active


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


CLASSES = ['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck']


def save_recon_grid(model, dl, epoch, device, N=3, out_dir="recon_vis_v3"):
    model.eval()
    x, labels = next(iter(dl))
    x = x[:N].to(device)
    labels = labels[:N]

    with torch.no_grad():
        x_r, logits, binary, z = model(x)

        codes_per_img = binary.permute(0, 2, 3, 1).reshape(N, -1, model.num_features)
        per_img_uniq = [c.unique(dim=0).size(0) for c in codes_per_img]
        total_positions = codes_per_img.shape[1] * N

    x_np = denorm(x.cpu()).permute(0, 2, 3, 1).numpy()
    r_np = denorm(x_r.cpu()).permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(3, N, figsize=(2.8 * N, 8))
    fig.suptitle(
        f'Epoch {epoch} | Recon: {F.mse_loss(x_r, x).item():.4f} | '
        f'Perplexity: {model.compute_perplexity(binary)[0]:.3f} | '
        f'uniq codes: {sum(per_img_uniq)}/{total_positions}',
        fontsize=11, fontweight='bold'
    )

    for i in range(N):
        axes[0, i].imshow(x_np[i]); axes[0, i].axis('off')
        axes[0, i].set_title(CLASSES[int(labels[i])], fontsize = 9)

        axes[1, i].imshow(r_np[i]); axes[1, i].axis('off')
        mse = ((x_np[i] - r_np[i]) ** 2).mean()
        axes[1, i].set_title(f'MSE:{mse:.4f} | uniq:{per_img_uniq[i]}', fontsize = 9)

        diff = ((x_np[i] - r_np[i]) ** 2).mean(axis = 2)
        axes[2, i].imshow(diff, cmap='hot', vmin = 0, vmax = 0.15)
        axes[2, i].axis('off')
        axes[2, i].set_title('SqErr', fontsize = 9)

    axes[0, 0].set_ylabel('Original', fontsize = 10, fontweight='bold')
    axes[1, 0].set_ylabel('Recon', fontsize = 10, fontweight='bold')
    axes[2, 0].set_ylabel('Error', fontsize = 10, fontweight='bold')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out} | per-img uniq: {per_img_uniq}")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features, embed_dim, hidden_dim = 256, 128, 128
    hierarchical_alpha = 0.5
    diversity_w = 1.0
    tau_start, tau_end = 2.0, 0.2
    batch_size, num_epochs, lr = 128, 50, 3e-4

    print(f"Device: {device}")
    print(f"Num binary features (D): {num_features}")
    print(f"Embed dim: {embed_dim}")
    print(f"Hierarchical alpha: {hierarchical_alpha}")
    print(f"Diversity weight: {diversity_w}")
    print(f"Tau schedule: {tau_start} -> {tau_end}")
    print()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BinaryVQVAE(
        num_features=num_features,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        hierarchical_alpha=hierarchical_alpha,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        tau = tau_start + (tau_end - tau_start) * epoch / max(num_epochs - 1, 1)

        tot = rec = div = hrl = perp = act = avg1 = 0.0
        n = 0

        for x, _ in train_loader:
            x = x.to(device)
            x_recon, logits, binary, z = model(x, tau)
            total_loss, recon_loss, diversity_loss, hier_loss = model.compute_loss(
                x, x_recon, binary, diversity_w=diversity_w
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ppx, a, s = model.compute_perplexity(binary)

            tot += total_loss.item()
            rec += recon_loss.item()
            div += diversity_loss.item()
            hrl += hier_loss.item()
            perp += ppx
            act += a
            avg1 += s
            n += 1

        scheduler.step()

        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}]  "
            f"tau: {tau:.3f}  Total: {tot/n:.4f}  Recon: {rec/n:.4f}  "
            f"Diversity: {div/n:.4f}  Hier: {hrl/n:.4f}  "
            f"Perplexity: {perp/n:.3f}  "
            f"Active dims: {act/n:.0f}/{num_features}  Avg 1s: {avg1/n:.1f}"
        )

        if (epoch + 1) % 5 == 0:
            model.eval()
            val = 0.0
            m = 0
            with torch.no_grad():
                for x, _ in test_loader:
                    x = x.to(device)
                    x_recon, _, binary, _ = model(x)
                    val += F.mse_loss(x_recon, x).item()
                    m += 1
            print(f"  >> Val Recon Loss: {val/m:.4f}")
            save_recon_grid(model, test_loader, epoch + 1, device, N=3, out_dir="recon_vis_v3")
            print()

    print("=" * 70)
    print("Training complete!")

    model.eval()
    test_recon = test_perp = test_active = test_avg1 = 0.0
    m = 0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            x_recon, _, binary, _ = model(x)
            test_recon += F.mse_loss(x_recon, x).item()
            ppx, a, s = model.compute_perplexity(binary)
            test_perp += ppx
            test_active += a
            test_avg1 += s
            m += 1

    print(
        f"Final Test  |  Recon: {test_recon/m:.4f}  Perplexity: {test_perp/m:.3f}  "
        f"Active dims: {test_active/m:.0f}/{num_features}  Avg 1s: {test_avg1/m:.1f}"
    )


if __name__ == "__main__":
    train()