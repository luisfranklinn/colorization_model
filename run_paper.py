"""
run_paper.py — Pipeline completo para artigo científico SBPO
=============================================================
Executa em sequência:
  1. Treinamento com CIFAR-10 (download automático)
  2. Avaliação quantitativa (MSE, PSNR, SSIM)
  3. Ablação: CNN puro vs CNN + Min-Cost Flow
  4. Geração de todas as figuras e tabelas

Uso:
    python3 run_paper.py              # configuração padrão (rápida)
    python3 run_paper.py --full       # dataset completo (mais lento, melhores resultados)

Saída em: paper_output/
  figures/01_training_curves.png
  figures/02_sample_grid.png
  figures/03_cost_maps.png
  figures/04_ablation.png
  figures/05_lab_colorspace.png
  metrics.csv
  metrics_table.tex
  model_summary.txt
"""

import argparse
import csv
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")   # sem display — funciona em qualquer ambiente
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from skimage import color as skcolor
from skimage.metrics import (peak_signal_noise_ratio as psnr,
                              structural_similarity as ssim)

# importa o modelo do script principal
sys.path.insert(0, os.path.dirname(__file__))
from colorizer_mcf import FlowColorizer, CIFAR10LabDataset

warnings.filterwarnings("ignore")

# ── Estilo publicação ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":     150,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
})

OUT   = "paper_output"
FIGS  = os.path.join(OUT, "figures")
os.makedirs(FIGS, exist_ok=True)


# ===========================================================================
# TREINAMENTO
# ===========================================================================

def train(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        ab_opt, ab_init = model(L)
        loss = F.mse_loss(ab_opt, ab) + 0.5 * F.mse_loss(ab_init, ab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0.0
    for L, ab in loader:
        L, ab = L.to(device), ab.to(device)
        ab_opt, ab_init = model(L)
        loss = F.mse_loss(ab_opt, ab) + 0.5 * F.mse_loss(ab_init, ab)
        total += loss.item()
    return total / len(loader)


# ===========================================================================
# MÉTRICAS QUANTITATIVAS
# ===========================================================================

@torch.no_grad()
def compute_metrics(model, dataset, device, n_samples=200):
    """Calcula MSE, PSNR e SSIM médios em n_samples amostras."""
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []

    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in indices:
        L, ab_true = dataset[idx]
        ab_opt, _ = model(L.unsqueeze(0).to(device))

        ab_pred_np = ab_opt[0].permute(1, 2, 0).cpu().numpy() * 128.0
        ab_true_np = ab_true.permute(1, 2, 0).numpy() * 128.0

        L_np = (L.squeeze().numpy() + 1.0) * 50.0

        lab_pred = np.concatenate([L_np[:,:,None], ab_pred_np], axis=-1)
        lab_true = np.concatenate([L_np[:,:,None], ab_true_np], axis=-1)

        rgb_pred = np.clip(skcolor.lab2rgb(lab_pred), 0, 1).astype(np.float32)
        rgb_true = np.clip(skcolor.lab2rgb(lab_true), 0, 1).astype(np.float32)

        mse_val  = float(np.mean((rgb_pred - rgb_true) ** 2))
        psnr_val = float(psnr(rgb_true, rgb_pred, data_range=1.0))
        ssim_val = float(ssim(rgb_true, rgb_pred, data_range=1.0, channel_axis=-1))

        mse_list.append(mse_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return {
        "MSE":  (np.mean(mse_list),  np.std(mse_list)),
        "PSNR": (np.mean(psnr_list), np.std(psnr_list)),
        "SSIM": (np.mean(ssim_list), np.std(ssim_list)),
    }


# ===========================================================================
# FIGURA 1 — Curvas de treinamento
# ===========================================================================

def plot_training_curves(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(epochs, train_losses, "b-o", ms=4, lw=1.5, label="Treino")
    ax.plot(epochs, val_losses,   "r-s", ms=4, lw=1.5, label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Curvas de Aprendizado — CNN + Min-Cost Flow")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIGS, "01_training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 2 — Grade de amostras: P&B → CNN init → MCF → Original
# ===========================================================================

@torch.no_grad()
def plot_sample_grid(model, dataset, device, n=6):
    model.eval()
    indices = list(range(n))

    fig, axes = plt.subplots(n, 4, figsize=(9, n * 2.1))
    cols = ["Entrada P&B", "CNN (inicial)", "CNN + MCF (final)", "Original"]
    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=9, fontweight="bold")

    for row, idx in enumerate(indices):
        L, ab_true = dataset[idx]
        ab_opt, ab_init = model(L.unsqueeze(0).to(device))

        L_np = (L.squeeze().numpy() + 1.0) * 50.0

        def lab_to_rgb(ab_tensor):
            ab_np = ab_tensor[0].permute(1,2,0).cpu().numpy() * 128.0
            lab   = np.concatenate([L_np[:,:,None], ab_np], axis=-1)
            return np.clip(skcolor.lab2rgb(lab), 0, 1)

        rgb_init = lab_to_rgb(ab_init)
        rgb_opt  = lab_to_rgb(ab_opt)
        ab_t_np  = ab_true.permute(1,2,0).numpy() * 128.0
        lab_true = np.concatenate([L_np[:,:,None], ab_t_np], axis=-1)
        rgb_true = np.clip(skcolor.lab2rgb(lab_true), 0, 1)

        imgs = [L_np / 100.0, rgb_init, rgb_opt, rgb_true]
        cmaps = ["gray", None, None, None]

        for col, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest")
            ax.axis("off")

    fig.suptitle("Comparação Visual: Colorização CNN + Min-Cost Flow",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(FIGS, "02_sample_grid.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 3 — Visualização dos Mapas de Custo
# ===========================================================================

@torch.no_grad()
def plot_cost_maps(model, dataset, device, n=4):
    model.eval()
    fig, axes = plt.subplots(n, 4, figsize=(9, n * 2.1))
    cols = ["Entrada P&B", "Custo Horizontal $c_h$",
            "Custo Vertical $c_v$", "Resultado Final"]
    for ax, c in zip(axes[0], cols):
        ax.set_title(c, fontsize=9, fontweight="bold")

    for row in range(n):
        L, ab_true = dataset[row]
        feat       = model.cnn(L.unsqueeze(0).to(device))
        costs, _   = model.cost_predictor(feat)
        ab_opt, _  = model(L.unsqueeze(0).to(device))

        L_np  = (L.squeeze().numpy() + 1.0) * 50.0
        c_h   = costs[0, 0].cpu().numpy()
        c_v   = costs[0, 1].cpu().numpy()

        ab_np  = ab_opt[0].permute(1,2,0).cpu().numpy() * 128.0
        lab    = np.concatenate([L_np[:,:,None], ab_np], axis=-1)
        rgb    = np.clip(skcolor.lab2rgb(lab), 0, 1)

        data = [L_np / 100.0, c_h, c_v, rgb]
        cmps = ["gray", "hot", "hot", None]
        lbls = [None, "Baixo=fluxo livre\nAlto=borda preservada",
                "Baixo=fluxo livre\nAlto=borda preservada", None]

        for col, (img, cmp) in enumerate(zip(data, cmps)):
            ax = axes[row, col]
            im = ax.imshow(img, cmap=cmp, interpolation="nearest")
            ax.axis("off")
            if col in (1, 2) and row == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Mapas de Custo Aprendidos pela CNN\n"
                 "(controlam o fluxo de cor entre pixels adjacentes)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIGS, "03_cost_maps.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 4 — Ablação: CNN puro vs CNN + MCF
# ===========================================================================

@torch.no_grad()
def plot_ablation(model, dataset, device, metrics_cnn, metrics_mcf, n=4):
    """Compara visualmente CNN puro (ab_init) vs CNN+MCF (ab_opt) e plota métricas."""
    model.eval()

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(n + 1, 4, height_ratios=[0.4] + [1]*n,
                            hspace=0.35, wspace=0.08)

    # Linha de métricas (barra)
    ax_metrics = fig.add_subplot(gs[0, :])
    models_lb  = ["CNN puro (ab_init)", "CNN + MCF (ab_opt)"]
    psnr_vals  = [metrics_cnn["PSNR"][0], metrics_mcf["PSNR"][0]]
    ssim_vals  = [metrics_cnn["SSIM"][0], metrics_mcf["SSIM"][0]]

    x = np.array([0, 1])
    ax_metrics.bar(x - 0.2, psnr_vals, width=0.35, label="PSNR (dB)", color="#2196F3", alpha=0.85)
    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(models_lb)
    ax_metrics.set_ylabel("PSNR (dB)")
    ax_metrics.legend(loc="upper left", fontsize=8)
    ax_metrics.set_title("Ablação Quantitativa: CNN puro vs CNN + Min-Cost Flow", fontweight="bold")

    ax2 = ax_metrics.twinx()
    ax2.bar(x + 0.2, ssim_vals, width=0.35, label="SSIM", color="#FF5722", alpha=0.85)
    ax2.set_ylabel("SSIM")
    ax2.legend(loc="upper right", fontsize=8)

    # Anotações
    for i, (p, s) in enumerate(zip(psnr_vals, ssim_vals)):
        ax_metrics.text(x[i] - 0.2, p + 0.1, f"{p:.2f}", ha="center", fontsize=8)
        ax2.text(x[i] + 0.2, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)

    # Amostras visuais
    col_titles = ["P&B", "CNN puro", "CNN + MCF", "Original"]
    for col, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    for row in range(n):
        L, ab_true = dataset[row]
        ab_opt, ab_init = model(L.unsqueeze(0).to(device))
        L_np = (L.squeeze().numpy() + 1.0) * 50.0

        def to_rgb(ab_t):
            ab_np = ab_t[0].permute(1,2,0).cpu().numpy() * 128.0
            return np.clip(skcolor.lab2rgb(np.concatenate([L_np[:,:,None], ab_np], -1)), 0, 1)

        ab_t_np  = ab_true.permute(1,2,0).numpy() * 128.0
        rgb_true = np.clip(skcolor.lab2rgb(np.concatenate([L_np[:,:,None], ab_t_np], -1)), 0, 1)

        imgs = [L_np/100.0, to_rgb(ab_init), to_rgb(ab_opt), rgb_true]
        cmps = ["gray", None, None, None]

        for col, (img, cmp) in enumerate(zip(imgs, cmps)):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.imshow(img, cmap=cmp, vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")

    path = os.path.join(FIGS, "04_ablation.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 5 — Espaço de Cores CIE Lab (diagrama conceitual)
# ===========================================================================

def plot_lab_colorspace():
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    # Plano ab para L=50
    a_vals = np.linspace(-128, 127, 256)
    b_vals = np.linspace(-128, 127, 256)
    aa, bb = np.meshgrid(a_vals, b_vals)
    L50    = np.full_like(aa, 50)
    lab_grid = np.stack([L50, aa, bb], axis=-1)
    rgb_grid = np.clip(skcolor.lab2rgb(lab_grid), 0, 1)

    axes[0].imshow(rgb_grid, origin="lower",
                   extent=[-128, 127, -128, 127], aspect="auto")
    axes[0].set_xlabel("Canal $a$ (verde ← → vermelho)")
    axes[0].set_ylabel("Canal $b$ (azul ← → amarelo)")
    axes[0].set_title("Plano $ab$ do Espaço CIE Lab ($L=50$)")
    axes[0].axhline(0, color="white", lw=0.5, ls="--", alpha=0.5)
    axes[0].axvline(0, color="white", lw=0.5, ls="--", alpha=0.5)

    # Comparação escala de cinza vs cor
    np.random.seed(42)
    sample_L  = np.random.uniform(20, 80, (8, 8))
    sample_a  = np.random.uniform(-60, 60, (8, 8))
    sample_b  = np.random.uniform(-60, 60, (8, 8))
    lab_sample = np.stack([sample_L, sample_a, sample_b], axis=-1)
    rgb_sample = np.clip(skcolor.lab2rgb(lab_sample), 0, 1)
    gray_sample = sample_L[:, :, None] / 100.0 * np.ones((1, 1, 3))

    combined = np.concatenate([gray_sample, rgb_sample], axis=1)
    axes[1].imshow(combined, interpolation="nearest")
    axes[1].axvline(7.5, color="white", lw=2)
    axes[1].set_title("Espaço de cor: escala de cinza (esq.) vs Lab colorido (dir.)")
    axes[1].axis("off")
    axes[1].text(3.5, -0.8, "Canal $L$ (entrada)",
                 ha="center", fontsize=9, transform=axes[1].transData)
    axes[1].text(11.5, -0.8, "Canais $ab$ (alvo)",
                 ha="center", fontsize=9, transform=axes[1].transData)

    fig.suptitle("Espaço de Cores CIE Lab — Base do Modelo de Colorização",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(FIGS, "05_lab_colorspace.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# MÉTRICAS PARA CNN PURO (sem MCF)
# ===========================================================================

@torch.no_grad()
def compute_metrics_cnn_only(model, dataset, device, n_samples=200):
    """Métricas usando apenas a saída direta da CNN (ab_init), sem o otimizador."""
    model.eval()
    mse_list, psnr_list, ssim_list = [], [], []
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)

    for idx in indices:
        L, ab_true = dataset[idx]
        _, ab_init = model(L.unsqueeze(0).to(device))

        ab_pred_np = ab_init[0].permute(1,2,0).cpu().numpy() * 128.0
        ab_true_np = ab_true.permute(1,2,0).numpy() * 128.0
        L_np = (L.squeeze().numpy() + 1.0) * 50.0

        rgb_pred = np.clip(skcolor.lab2rgb(np.concatenate([L_np[:,:,None], ab_pred_np], -1)), 0, 1).astype(np.float32)
        rgb_true = np.clip(skcolor.lab2rgb(np.concatenate([L_np[:,:,None], ab_true_np], -1)), 0, 1).astype(np.float32)

        mse_list.append(float(np.mean((rgb_pred - rgb_true)**2)))
        psnr_list.append(float(psnr(rgb_true, rgb_pred, data_range=1.0)))
        ssim_list.append(float(ssim(rgb_true, rgb_pred, data_range=1.0, channel_axis=-1)))

    return {
        "MSE":  (np.mean(mse_list),  np.std(mse_list)),
        "PSNR": (np.mean(psnr_list), np.std(psnr_list)),
        "SSIM": (np.mean(ssim_list), np.std(ssim_list)),
    }


# ===========================================================================
# TABELA LaTeX
# ===========================================================================

def save_latex_table(metrics_cnn, metrics_mcf, path):
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Avaliação Quantitativa: CNN puro vs CNN + Min-Cost Flow (CIFAR-10)}",
        r"\label{tab:metrics}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Método & MSE $\downarrow$ & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ \\",
        r"\hline",
    ]
    for name, m in [("CNN puro", metrics_cnn), ("CNN + MCF (proposto)", metrics_mcf)]:
        row = (f"{name} & "
               f"${m['MSE'][0]:.4f} \\pm {m['MSE'][1]:.4f}$ & "
               f"${m['PSNR'][0]:.2f} \\pm {m['PSNR'][1]:.2f}$ & "
               f"${m['SSIM'][0]:.3f} \\pm {m['SSIM'][1]:.3f}$ \\\\")
        lines.append(row)
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Salvo: {path}")


# ===========================================================================
# SUMÁRIO DO MODELO
# ===========================================================================

def save_model_summary(model, path):
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_h    = model.H * (model.W - 1)
    n_v    = (model.H - 1) * model.W
    lines  = [
        "=" * 55,
        "  Arquitetura: CNN + Min-Cost Flow para Colorização",
        "=" * 55,
        f"  Parâmetros treináveis : {train_:>12,}",
        f"  Parâmetros totais     : {total:>12,}",
        f"  Tamanho do patch      : {model.H} × {model.W}",
        f"  Arestas horizontais   : {n_h:>12,}",
        f"  Arestas verticais     : {n_v:>12,}",
        f"  Variáveis no QP       : {model.H*model.W*2:>12,}",
        "-" * 55,
        "  Componentes:",
        "    UNetEncoder  — encoder/decoder com skip connections",
        "    CostPredictor — edge_head (Softplus) + color_head (Tanh)",
        "    FlowLayer    — QP diferenciável via cvxpylayers/ECOS",
        "-" * 55,
        "  Formulação do QP:",
        "    min  Σ c_h·‖Δab_h‖² + Σ c_v·‖Δab_v‖² + λ·‖ab−ab_init‖²",
        "    Backprop via diferenciação implícita das condições KKT",
        "=" * 55,
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Salvo: {path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",    action="store_true",
                        help="Usa dataset completo (60k imgs, mais lento)")
    parser.add_argument("--epochs",  type=int,   default=8)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--lam",     type=float, default=1.0)
    parser.add_argument("--data",    type=str,   default="./data")
    parser.add_argument("--metrics-samples", type=int, default=300)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("=" * 60)
    print("  Pipeline Artigo Científico — CNN + Min-Cost Flow")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Épocas  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  Saída   : {OUT}/")
    print()

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/7] Carregando CIFAR-10...")
    train_ds = CIFAR10LabDataset(root=args.data, train=True)
    val_ds   = CIFAR10LabDataset(root=args.data, train=False)

    if not args.full:
        n_train = 2000
        n_val   = 400
        train_ds = torch.utils.data.Subset(train_ds, range(n_train))
        val_ds   = torch.utils.data.Subset(val_ds,   range(n_val))
        print(f"  Subconjunto rápido: {n_train} treino / {n_val} val")
        print("  (use --full para 60k imagens)")
    else:
        print(f"  Dataset completo: {len(train_ds)} treino / {len(val_ds)} val")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Modelo ────────────────────────────────────────────────────────────
    print("\n[2/7] Construindo modelo...")
    model     = FlowColorizer(32, 32, base_ch=32, lam=args.lam).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    save_model_summary(model, os.path.join(OUT, "model_summary.txt"))

    # ── Treinamento ───────────────────────────────────────────────────────
    print(f"\n[3/7] Treinando por {args.epochs} épocas...")
    train_losses, val_losses = [], []
    best_val, best_path = float("inf"), os.path.join(OUT, "model_best.pth")

    for epoch in range(1, args.epochs + 1):
        tr = train(model, train_loader, optimizer, DEVICE)
        vl = evaluate(model, val_loader,  DEVICE)
        scheduler.step()
        train_losses.append(tr)
        val_losses.append(vl)
        marker = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), best_path)
            marker = " ← melhor"
        print(f"  Época {epoch:3d}/{args.epochs}  "
              f"treino={tr:.4f}  val={vl:.4f}{marker}")

    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    # ── Métricas ──────────────────────────────────────────────────────────
    print("\n[4/7] Calculando métricas quantitativas...")
    n_met = min(args.metrics_samples, len(val_ds))

    # Usa o val_ds real (não Subset) para compute_metrics que usa .cifar internamente
    raw_val = val_ds.dataset if isinstance(val_ds, torch.utils.data.Subset) else val_ds
    idx_arr = (val_ds.indices[:n_met]
               if isinstance(val_ds, torch.utils.data.Subset)
               else list(range(n_met)))

    class _Slice(torch.utils.data.Dataset):
        def __init__(self, ds, indices):
            self.ds, self.indices = ds, indices
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            return self.ds[self.indices[i]]

    eval_ds = _Slice(raw_val, idx_arr)

    metrics_mcf = compute_metrics(model, eval_ds, DEVICE, n_met)
    metrics_cnn = compute_metrics_cnn_only(model, eval_ds, DEVICE, n_met)

    print(f"\n  {'Método':<25} {'MSE':>8} {'PSNR':>8} {'SSIM':>8}")
    print(f"  {'-'*50}")
    for name, m in [("CNN puro", metrics_cnn), ("CNN + MCF (proposto)", metrics_mcf)]:
        print(f"  {name:<25} "
              f"{m['MSE'][0]:>8.4f} "
              f"{m['PSNR'][0]:>7.2f}dB "
              f"{m['SSIM'][0]:>8.3f}")

    # Salva CSV
    csv_path = os.path.join(OUT, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Método", "MSE_mean", "MSE_std", "PSNR_mean", "PSNR_std",
                    "SSIM_mean", "SSIM_std"])
        for name, m in [("CNN_puro", metrics_cnn), ("CNN_MCF", metrics_mcf)]:
            w.writerow([name,
                        f"{m['MSE'][0]:.6f}",  f"{m['MSE'][1]:.6f}",
                        f"{m['PSNR'][0]:.4f}", f"{m['PSNR'][1]:.4f}",
                        f"{m['SSIM'][0]:.4f}", f"{m['SSIM'][1]:.4f}"])
    print(f"  Salvo: {csv_path}")

    save_latex_table(metrics_cnn, metrics_mcf,
                     os.path.join(OUT, "metrics_table.tex"))

    # ── Figuras ───────────────────────────────────────────────────────────
    print("\n[5/7] Gerando figuras...")

    print("  → Fig 1: Curvas de treinamento")
    plot_training_curves(train_losses, val_losses)

    print("  → Fig 2: Grade de amostras")
    vis_ds = _Slice(raw_val, list(range(6)))
    plot_sample_grid(model, vis_ds, DEVICE, n=6)

    print("  → Fig 3: Mapas de custo")
    plot_cost_maps(model, _Slice(raw_val, list(range(4))), DEVICE, n=4)

    print("  → Fig 4: Ablação CNN vs MCF")
    plot_ablation(model, _Slice(raw_val, list(range(4))), DEVICE,
                  metrics_cnn, metrics_mcf, n=4)

    print("  → Fig 5: Espaço de cores CIE Lab")
    plot_lab_colorspace()

    # ── Resumo final ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("  [6/7] Pronto! Todos os artefatos salvos em:")
    print(f"         {os.path.abspath(OUT)}/")
    print()
    print("  Figuras:")
    for f in sorted(os.listdir(FIGS)):
        path = os.path.join(FIGS, f)
        print(f"    {f:<35}  {os.path.getsize(path)//1024:>5} KB")
    print()
    print("  Tabelas/Dados:")
    for fn in ["metrics.csv", "metrics_table.tex", "model_summary.txt"]:
        fp = os.path.join(OUT, fn)
        if os.path.exists(fp):
            print(f"    {fn:<35}  {os.path.getsize(fp):>6} bytes")
    print()
    print(f"  Tempo total: {elapsed/60:.1f} min")
    print()
    print("  [7/7] Cole no LaTeX do artigo:")
    print(r"    \input{paper_output/metrics_table.tex}")
    print(r"    \includegraphics[width=\linewidth]{paper_output/figures/02_sample_grid.png}")
    print("=" * 60)


if __name__ == "__main__":
    main()
