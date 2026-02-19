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


class PatchEmbed(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_channels=3, embed_dim=512):
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=512, num_features=512,
                 img_size=128, patch_size=8, depth=6, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, num_features)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.proj(x)
        B, N, D = x.shape
        x = x.transpose(1, 2).view(B, D, H, W)
        return x


class ViTDecoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=512, embed_dim=256,
                 img_size=128, patch_size=8, depth=6, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = out_channels * patch_size * patch_size

        self.proj_in = nn.Linear(embed_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, hidden_dim) * 0.02)
        self.blocks = nn.Sequential(*[
            TransformerBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, patch_dim)

    def forward(self, z):
        B, C, H, W = z.shape
        z = z.flatten(2).transpose(1, 2)
        z = self.proj_in(z)
        z = z + self.pos_embed
        z = self.blocks(z)
        z = self.norm(z)
        z = self.proj_out(z)
        p = self.patch_size
        h = w = self.img_size // p
        z = z.view(B, h, w, 3, p, p)
        z = z.permute(0, 3, 1, 4, 2, 5).contiguous()
        z = z.view(B, 3, self.img_size, self.img_size)
        return z


class BinaryVQVAE(nn.Module):
    def __init__(self, num_features=512, embed_dim=256, hidden_dim=512, in_channels=3,
                 img_size=128, patch_size=8, enc_depth=6, dec_depth=6, num_heads=8):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.encoder = ViTEncoder(in_channels, hidden_dim, num_features, img_size, patch_size, enc_depth, num_heads)
        self.decoder = ViTDecoder(out_channels=in_channels, hidden_dim=hidden_dim, embed_dim=embed_dim, img_size=img_size, patch_size=patch_size, depth=dec_depth, num_heads=num_heads)
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)

    def gumbel_sigmoid(self, logits, tau):
        if self.training:
            g1 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            g2 = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            y_soft = torch.sigmoid((logits + g1 - g2) / tau)
            y_hard = (y_soft > 0.5).float()
            return y_hard - y_soft.detach() + y_soft
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

    def forward(self, x, tau=1.0):
        logits, binary = self.encode(x, tau)
        x_recon, z = self.decode(binary)
        return x_recon, logits, binary, z

    def compute_loss(self, x, x_recon, logits, binary, diversity_w=0.0, binarization_w=0.0):
        recon_loss = F.mse_loss(x_recon, x)
        p = binary.mean(dim=[0, 2, 3])
        eps = 1e-7
        p = p.clamp(eps, 1 - eps)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        diversity_loss = -entropy
        sigma = torch.sigmoid(logits)
        binarization_loss = (sigma * (1 - sigma)).mean()
        total_loss = recon_loss + diversity_w * diversity_loss + binarization_w * binarization_loss
        return total_loss, recon_loss, diversity_loss, binarization_loss

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


CLASSES = None
ANALYSIS_CLASSES = [
    "apple_pie", "pizza", "sushi", "steak", "ice_cream",
    "french_fries", "hamburger", "hot_dog", "chocolate_cake", "caesar_salad"
]


def save_recon_grid(model, dl, epoch, device, N=3, out_dir="recon_vis"):
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
        f'uniq: {sum(per_img_uniq)}/{total_positions}',
        fontsize=10, fontweight='bold'
    )
    for i in range(N):
        axes[0, i].imshow(x_np[i]); axes[0, i].axis('off')
        lbl = int(labels[i])
        name = CLASSES[lbl] if CLASSES and lbl < len(CLASSES) else str(lbl)
        axes[0, i].set_title(name[:20], fontsize=8)
        axes[1, i].imshow(r_np[i]); axes[1, i].axis('off')
        mse = ((x_np[i] - r_np[i]) ** 2).mean()
        axes[1, i].set_title(f'MSE:{mse:.4f} | uniq:{per_img_uniq[i]}', fontsize=8)
        diff = ((x_np[i] - r_np[i]) ** 2).mean(axis=2)
        axes[2, i].imshow(diff, cmap='hot', vmin=0, vmax=0.15)
        axes[2, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Recon', fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel('Error', fontsize=10, fontweight='bold')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out}")


def save_bit_stats(model, dl, epoch, device, num_samples=100, out_dir="bit_stats"):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    for cls_name in ANALYSIS_CLASSES:
        if cls_name not in CLASSES:
            continue
        target_idx = CLASSES.index(cls_name)
        collected_x = []
        for x, labels in dl:
            mask = labels == target_idx
            if mask.any():
                collected_x.append(x[mask])
            if sum(c.shape[0] for c in collected_x) >= num_samples:
                break
        if not collected_x or sum(c.shape[0] for c in collected_x) == 0:
            continue
        collected_x = torch.cat(collected_x, dim=0)[:num_samples].to(device)
        with torch.no_grad():
            _, _, binary, _ = model(collected_x)
            mean_binary = binary.mean(dim=[2, 3])
            class_mean = mean_binary.mean(dim=0).cpu().numpy()
            class_std = mean_binary.std(dim=0).cpu().numpy()
        out_path = os.path.join(out_dir, f"epoch_{epoch:03d}_{cls_name}.csv")
        with open(out_path, "w") as f:
            f.write("bit_id,mean,std\n")
            for d in range(model.num_features):
                f.write(f"{d},{class_mean[d]:.6f},{class_std[d]:.6f}\n")
    print(f"  Saved bit stats -> {out_dir}/")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 128
    patch_size = 8
    num_features = 512
    embed_dim = 256
    vit_hidden_dim = 384
    enc_depth = 6
    dec_depth = 6
    num_heads = 8
    diversity_w = 0
    binarization_w = 0
    tau_start, tau_end = 1.0, 0.2
    batch_size = 32
    num_epochs = 50
    lr = 3e-4

    print(f"Dataset: Food-101 @ {img_size}x{img_size}")
    print(f"Device: {device}")
    print(f"Architecture: ViT (enc={enc_depth}L, dec={dec_depth}L, heads={num_heads})")
    print(f"Patch: {patch_size} -> {img_size//patch_size}x{img_size//patch_size} = {(img_size//patch_size)**2} tokens")
    print(f"ViT hidden dim: {vit_hidden_dim}")
    print(f"Num binary features: {num_features}")
    print(f"Embed dim: {embed_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Tau: {tau_start} -> {tau_end}")
    print()

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.Food101(root="./data", split="train", download=True, transform=transform_train)
    test_dataset = torchvision.datasets.Food101(root="./data", split="test", download=True, transform=transform_test)

    global CLASSES
    CLASSES = train_dataset.classes

    print(f"Train: {len(train_dataset)}")
    print(f"Test:  {len(test_dataset)}")
    print(f"Classes: {len(CLASSES)}")
    print()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = BinaryVQVAE(
        num_features=num_features,
        embed_dim=embed_dim,
        hidden_dim=vit_hidden_dim,
        img_size=img_size,
        patch_size=patch_size,
        enc_depth=enc_depth,
        dec_depth=dec_depth,
        num_heads=num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    print("=" * 70)

    for epoch in range(num_epochs):
        model.train()
        tau = tau_start + (tau_end - tau_start) * epoch / max(num_epochs - 1, 1)
        tot = rec = div = bnz = perp = act = avg1 = 0.0
        n = 0

        for x, _ in train_loader:
            x = x.to(device)
            x_recon, logits, binary, z = model(x, tau)
            total_loss, recon_loss, diversity_loss, binarization_loss = model.compute_loss(
                x, x_recon, logits, binary, diversity_w=diversity_w, binarization_w=binarization_w
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ppx, a, s = model.compute_perplexity(binary)
            tot += total_loss.item()
            rec += recon_loss.item()
            div += diversity_loss.item()
            bnz += binarization_loss.item()
            perp += ppx
            act += a
            avg1 += s
            n += 1

        scheduler.step()

        print(
            f"Epoch [{epoch+1:3d}/{num_epochs}]  "
            f"tau: {tau:.3f}  Total: {tot/n:.4f}  Recon: {rec/n:.4f}  "
            f"Div: {div/n:.4f}  Bin: {bnz/n:.4f}  "
            f"Perp: {perp/n:.3f}  "
            f"Active: {act/n:.0f}/{num_features}  Avg1s: {avg1/n:.1f}"
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
            save_recon_grid(model, test_loader, epoch + 1, device, N=3)
            save_bit_stats(model, test_loader, epoch + 1, device, num_samples=100)
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
        f"Final Test  |  Recon: {test_recon/m:.4f}  Perp: {test_perp/m:.3f}  "
        f"Active: {test_active/m:.0f}/{num_features}  Avg1s: {test_avg1/m:.1f}"
    )

    torch.save(model.state_dict(), "binary_vqvae_vit_food101.pt")
    print("Saved binary_vqvae_vit_food101.pt")


if __name__ == "__main__":
    train()