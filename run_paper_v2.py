"""
run_paper_v2.py — Pipeline V2 para artigo SBPO
===============================================
Comparação entre três arquiteturas de colorização:
  A) CNN puro   — U-Net → ab direto
  B) GNN        — U-Net → message passing em grade → ab
  C) MCF        — U-Net → Min-Cost Flow (QP diferenciável) → ab

Dataset : STL-10 (96×96, 13k rotuladas + 100k sem rótulo) → redimensiona para 64×64
          Fallback automático para CIFAR-10 se STL-10 falhar.

Saída em paper_output_v2/:
  figures/samples/  — PNG individual por amostra (300 DPI, pronto para artigo)
  figures/training_curves.png
  figures/metrics_radar.png
  figures/metrics_bars.png
  figures/color_distributions.png
  metrics_all.csv
  table_comparison.tex
  model_summary.txt

Uso:
  python3 run_paper_v2.py              # rápido (subconjunto)
  python3 run_paper_v2.py --full       # dataset completo
"""

import argparse, csv, os, sys, time, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from skimage import color as skcolor
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

sys.path.insert(0, os.path.dirname(__file__))
from models import CNNColorizer, GNNColorizer, MCFColorizer

warnings.filterwarnings("ignore")

# ── Estilo publicação ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.05,
})

OUT    = "paper_output_v2"
SFIGS  = os.path.join(OUT, "figures", "samples")
MFIGS  = os.path.join(OUT, "figures")
for d in [SFIGS, MFIGS]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = 64    # resolução de treino
DISPLAY  = 192   # resolução de exibição nas figuras (3×upscale)


# ===========================================================================
# DATASET — STL-10 / CIFAR-10 fallback
# ===========================================================================

class LabDataset(torch.utils.data.Dataset):
    """
    Wrapper genérico: imagem RGB → (L, ab) no espaço CIE Lab.
    Funciona com STL-10 e CIFAR-10.
    """
    def __init__(self, base_dataset, size=IMG_SIZE):
        self.ds   = base_dataset
        self.size = size
        self.tf   = T.Compose([T.Resize((size, size)), T.ToTensor()])

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        img = F.interpolate(img.unsqueeze(0), self.size, mode="bilinear",
                            align_corners=False).squeeze(0)
        rgb = img.permute(1, 2, 0).numpy()                 # [H,W,3] ∈ [0,1]
        lab = skcolor.rgb2lab(rgb)

        L  = torch.tensor(lab[:,:,0]/50.0-1.0, dtype=torch.float32).unsqueeze(0)
        ab = torch.tensor(lab[:,:,1:]/128.0,   dtype=torch.float32).permute(2,0,1)
        return L, ab, label


def load_datasets(data_dir, size=IMG_SIZE, subset=0):
    """Tenta STL-10 primeiro; fallback para CIFAR-10."""
    try:
        print("  Tentando STL-10...")
        train_base = torchvision.datasets.STL10(
            data_dir, split="unlabeled", download=True)
        val_base   = torchvision.datasets.STL10(
            data_dir, split="test",      download=True)
        name = "STL-10"
    except Exception as e:
        print(f"  STL-10 falhou ({e}). Usando CIFAR-10.")
        train_base = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True)
        val_base   = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
        name = "CIFAR-10"

    train_ds = LabDataset(train_base, size)
    val_ds   = LabDataset(val_base,   size)

    if subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, range(min(subset, len(train_ds))))
        val_ds   = torch.utils.data.Subset(val_ds,   range(min(subset//5, len(val_ds))))

    print(f"  Dataset: {name} | treino={len(train_ds)} | val={len(val_ds)}")
    return train_ds, val_ds, name


# ===========================================================================
# TREINAMENTO
# ===========================================================================

def train_epoch(model, loader, opt, device, is_mcf=False):
    model.train()
    total = 0.0
    for L, ab, _ in loader:
        L, ab = L.to(device), ab.to(device)
        opt.zero_grad()
        if is_mcf:
            ab_opt, ab_init = model(L)
            loss = F.mse_loss(ab_opt, ab) + 0.5 * F.mse_loss(ab_init, ab)
        else:
            ab_pred = model(L)
            loss = F.mse_loss(ab_pred, ab)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, is_mcf=False):
    model.eval()
    total = 0.0
    for L, ab, _ in loader:
        L, ab = L.to(device), ab.to(device)
        if is_mcf:
            ab_opt, ab_init = model(L)
            loss = F.mse_loss(ab_opt, ab) + 0.5 * F.mse_loss(ab_init, ab)
        else:
            ab_pred = model(L)
            loss = F.mse_loss(ab_pred, ab)
        total += loss.item()
    return total / len(loader)


def train_model(model, name, train_loader, val_loader,
                device, epochs, lr, is_mcf=False):
    print(f"\n  [{name}]")
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, path = float("inf"), os.path.join(OUT, f"best_{name}.pth")
    tr_hist, vl_hist = [], []

    for ep in range(1, epochs + 1):
        tr = train_epoch(model, train_loader, opt, device, is_mcf)
        vl = eval_epoch(model,  val_loader,   device, is_mcf)
        sched.step()
        tr_hist.append(tr); vl_hist.append(vl)
        mark = ""
        if vl < best:
            best = vl
            torch.save(model.state_dict(), path)
            mark = " ✓"
        print(f"    ep {ep:3d}/{epochs}  tr={tr:.4f}  vl={vl:.4f}{mark}")

    model.load_state_dict(torch.load(path, map_location=device))
    return tr_hist, vl_hist


# ===========================================================================
# MÉTRICAS
# ===========================================================================

def lab_to_rgb(L_np, ab_np):
    lab = np.concatenate([L_np[:,:,None], ab_np], axis=-1)
    return np.clip(skcolor.lab2rgb(lab), 0, 1).astype(np.float32)


@torch.no_grad()
def compute_metrics(model, dataset, device, n=300, is_mcf=False):
    model.eval()
    mse_l, psnr_l, ssim_l = [], [], []
    idxs = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)

    for i in idxs:
        L, ab_true, _ = dataset[i]
        if is_mcf:
            ab_pred, _ = model(L.unsqueeze(0).to(device))
        else:
            ab_pred = model(L.unsqueeze(0).to(device))

        ab_pred_np = ab_pred[0].permute(1,2,0).cpu().numpy() * 128.0
        ab_true_np = ab_true.permute(1,2,0).numpy() * 128.0
        L_np = (L.squeeze().numpy() + 1.0) * 50.0

        rgb_p = lab_to_rgb(L_np, ab_pred_np)
        rgb_t = lab_to_rgb(L_np, ab_true_np)

        mse_l.append(float(np.mean((rgb_p - rgb_t)**2)))
        psnr_l.append(float(calc_psnr(rgb_t, rgb_p, data_range=1.0)))
        ssim_l.append(float(calc_ssim(rgb_t, rgb_p, data_range=1.0, channel_axis=-1)))

    return {
        "MSE":  (np.mean(mse_l),  np.std(mse_l)),
        "PSNR": (np.mean(psnr_l), np.std(psnr_l)),
        "SSIM": (np.mean(ssim_l), np.std(ssim_l)),
    }


# ===========================================================================
# HELPERS DE VISUALIZAÇÃO
# ===========================================================================

def to_display(arr):
    """Escala array [H,W] ou [H,W,3] para DISPLAY×DISPLAY com interpolação bicúbica."""
    from PIL import Image
    if arr.ndim == 2:
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    else:
        img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    return np.array(img.resize((DISPLAY, DISPLAY), Image.BICUBIC)) / 255.0


@torch.no_grad()
def get_predictions(models_dict, L_t, device):
    """Retorna dict {nome: rgb_pred} para uma amostra."""
    out = {}
    for name, (model, is_mcf) in models_dict.items():
        model.eval()
        L_in = L_t.unsqueeze(0).to(device)
        if is_mcf:
            ab_pred, _ = model(L_in)
        else:
            ab_pred = model(L_in)
        out[name] = ab_pred[0].permute(1,2,0).cpu().numpy() * 128.0
    return out


# ===========================================================================
# FIGURA 1 — Amostra individual (5 colunas: P&B | CNN | GNN | MCF | Original)
# ===========================================================================

def plot_single_sample(models_dict, dataset, idx, device, save_path):
    L, ab_true, label = dataset[idx]
    L_np    = (L.squeeze().numpy() + 1.0) * 50.0
    ab_t_np = ab_true.permute(1,2,0).numpy() * 128.0

    preds = get_predictions(models_dict, L, device)

    cols  = ["Entrada\nP&B"] + list(preds.keys()) + ["Ground\nTruth"]
    n_col = len(cols)
    fig, axes = plt.subplots(1, n_col, figsize=(n_col * 2.8, 3.2))

    # P&B
    axes[0].imshow(to_display(L_np / 100.0), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(cols[0], fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Predições
    for ax, (name, ab_np) in zip(axes[1:-1], preds.items()):
        rgb = lab_to_rgb(L_np, ab_np)
        ax.imshow(to_display(rgb), vmin=0, vmax=1)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    # Ground Truth
    rgb_gt = lab_to_rgb(L_np, ab_t_np)
    axes[-1].imshow(to_display(rgb_gt), vmin=0, vmax=1)
    axes[-1].set_title("Ground\nTruth", fontsize=10, fontweight="bold")
    axes[-1].axis("off")

    fig.suptitle(f"Amostra #{idx} — Comparação CNN vs GNN vs MCF",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


# ===========================================================================
# FIGURA 2 — Grid resumo: N amostras × 5 colunas (compacto para artigo)
# ===========================================================================

def plot_summary_grid(models_dict, dataset, indices, device):
    n  = len(indices)
    cols_labels = ["P&B"] + list(models_dict.keys()) + ["Original"]
    n_col = len(cols_labels)

    fig, axes = plt.subplots(n, n_col, figsize=(n_col * 2.0, n * 2.0),
                             gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    # Cabeçalhos
    for j, lbl in enumerate(cols_labels):
        axes[0, j].set_title(lbl, fontsize=9, fontweight="bold", pad=4)

    for row, idx in enumerate(indices):
        L, ab_true, _ = dataset[idx]
        L_np    = (L.squeeze().numpy() + 1.0) * 50.0
        ab_t_np = ab_true.permute(1,2,0).numpy() * 128.0
        preds   = get_predictions(models_dict, L, device)

        imgs = ([L_np / 100.0]
                + [lab_to_rgb(L_np, ab) for ab in preds.values()]
                + [lab_to_rgb(L_np, ab_t_np)])
        cmps = ["gray"] + [None] * (n_col - 1)

        for col, (img, cmp) in enumerate(zip(imgs, cmps)):
            ax = axes[row, col]
            ax.imshow(to_display(img), cmap=cmp, vmin=0, vmax=1,
                      interpolation="lanczos")
            ax.axis("off")

    fig.suptitle("Resultados Qualitativos: CNN | GNN | MCF vs Ground Truth",
                 fontsize=12, y=1.005)
    path = os.path.join(MFIGS, "qualitative_grid.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 3 — Curvas de treinamento
# ===========================================================================

def plot_training_curves(histories):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = {"CNN": "#1976D2", "GNN": "#388E3C", "MCF": "#E64A19"}
    styles_tr = {"CNN": "o-",  "GNN": "s-",  "MCF": "^-"}
    styles_vl = {"CNN": "o--", "GNN": "s--", "MCF": "^--"}

    for name, (tr, vl) in histories.items():
        ep = range(1, len(tr)+1)
        ax1.plot(ep, tr, styles_tr[name], color=colors[name], ms=4,
                 lw=1.5, label=name)
        ax2.plot(ep, vl, styles_vl[name], color=colors[name], ms=4,
                 lw=1.5, label=name)

    for ax, title in zip([ax1, ax2], ["Loss de Treino", "Loss de Validação"]):
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Curvas de Aprendizado — CNN vs GNN vs MCF", fontsize=12)
    fig.tight_layout()
    path = os.path.join(MFIGS, "training_curves.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 4 — Gráfico de barras com erro (MSE, PSNR, SSIM)
# ===========================================================================

def plot_metrics_bars(all_metrics):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    names  = list(all_metrics.keys())
    colors = ["#1976D2", "#388E3C", "#E64A19"]
    keys   = ["MSE", "PSNR", "SSIM"]
    titles = ["MSE $\\downarrow$", "PSNR (dB) $\\uparrow$", "SSIM $\\uparrow$"]

    for ax, key, title in zip(axes, keys, titles):
        means = [all_metrics[n][key][0] for n in names]
        stds  = [all_metrics[n][key][1] for n in names]
        bars  = ax.bar(names, means, yerr=stds, capsize=5,
                       color=colors, alpha=0.85, edgecolor="black", linewidth=0.6,
                       error_kw={"elinewidth": 1.2, "capthick": 1.2})
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(key)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        # Anota valores
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + s + (max(means)*0.01),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold")

    # Destaca melhor
    for ax, key in zip(axes, keys):
        means = [all_metrics[n][key][0] for n in names]
        best  = np.argmin(means) if key == "MSE" else np.argmax(means)
        ax.patches[best].set_edgecolor("gold")
        ax.patches[best].set_linewidth(2.5)

    fig.suptitle("Comparação Quantitativa — CNN vs GNN vs MCF (CIFAR-10/STL-10)",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(MFIGS, "metrics_bars.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 5 — Radar chart
# ===========================================================================

def plot_radar(all_metrics):
    cats   = ["PSNR\n(norm.)", "SSIM", "1-MSE\n(norm.)"]
    names  = list(all_metrics.keys())
    colors = ["#1976D2", "#388E3C", "#E64A19"]
    N      = len(cats)

    # Normaliza cada métrica para [0,1] entre os modelos
    psnr_v = np.array([all_metrics[n]["PSNR"][0] for n in names])
    ssim_v = np.array([all_metrics[n]["SSIM"][0] for n in names])
    mse_v  = np.array([all_metrics[n]["MSE"][0]  for n in names])

    def norm(v, invert=False):
        mn, mx = v.min(), v.max()
        if mx == mn: return np.ones_like(v) * 0.5
        n = (v - mn) / (mx - mn)
        return 1 - n if invert else n

    data = np.stack([norm(psnr_v), norm(ssim_v), norm(mse_v, invert=True)], axis=1)

    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})

    for i, (name, color) in enumerate(zip(names, colors)):
        vals = data[i].tolist() + [data[i][0]]
        ax.plot(angles, vals, "o-", color=color, lw=2, label=name, ms=6)
        ax.fill(angles, vals, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title("Comparação Multidimensional\n(normalizado entre modelos)",
                 fontsize=11, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    ax.grid(True, alpha=0.3)

    path = os.path.join(MFIGS, "metrics_radar.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# FIGURA 6 — Distribuição de cores previstas vs real
# ===========================================================================

@torch.no_grad()
def plot_color_distributions(models_dict, dataset, device, n=200):
    fig, axes = plt.subplots(2, len(models_dict) + 1, figsize=(12, 5))
    names = ["Ground Truth"] + list(models_dict.keys())
    channels = ["Canal $a$ (verde ↔ vermelho)", "Canal $b$ (azul ↔ amarelo)"]

    idxs = np.random.choice(len(dataset), min(n, len(dataset)), replace=False)
    all_ab = {name: [] for name in names}

    for i in idxs:
        L, ab_true, _ = dataset[i]
        all_ab["Ground Truth"].append(ab_true.numpy().reshape(2, -1))
        for name, (model, is_mcf) in models_dict.items():
            model.eval()
            if is_mcf:
                ab_pred, _ = model(L.unsqueeze(0).to(device))
            else:
                ab_pred = model(L.unsqueeze(0).to(device))
            all_ab[name].append(
                ab_pred[0].permute(1,2,0).cpu().numpy().reshape(-1, 2).T * 128.0
            )

    colors_map = {
        "Ground Truth": "#555555",
        "CNN":          "#1976D2",
        "GNN":          "#388E3C",
        "MCF":          "#E64A19",
    }

    for col, name in enumerate(names):
        data = np.concatenate(all_ab[name], axis=-1) * (1 if name == "Ground Truth" else 1)
        for row in range(2):
            ax = axes[row, col]
            ch = data[row] if name == "Ground Truth" else data[row]
            ax.hist(ch, bins=60, color=colors_map.get(name, "gray"),
                    alpha=0.75, density=True, edgecolor="none")
            if row == 0:
                ax.set_title(name, fontweight="bold", fontsize=9)
            ax.set_xlabel(channels[row], fontsize=8)
            ax.set_ylabel("Densidade" if col == 0 else "", fontsize=8)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.set_axisbelow(True)

    fig.suptitle("Distribuição dos Canais de Cor Previstos vs Ground Truth",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(MFIGS, "color_distributions.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Salvo: {path}")


# ===========================================================================
# TABELA LaTeX + CSV
# ===========================================================================

def save_outputs(all_metrics, histories, dataset_name):
    # CSV
    csv_path = os.path.join(OUT, "metrics_all.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model", "MSE_mean", "MSE_std",
                    "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std"])
        for name, m in all_metrics.items():
            w.writerow([name,
                        f"{m['MSE'][0]:.6f}",  f"{m['MSE'][1]:.6f}",
                        f"{m['PSNR'][0]:.4f}", f"{m['PSNR'][1]:.4f}",
                        f"{m['SSIM'][0]:.4f}", f"{m['SSIM'][1]:.4f}"])
    print(f"  Salvo: {csv_path}")

    # LaTeX
    tex_path = os.path.join(OUT, "table_comparison.tex")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\caption{{Avaliação Quantitativa: CNN vs GNN vs MCF ({dataset_name})}}",
        r"\label{tab:comparison_v2}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Método & MSE $\downarrow$ & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ \\",
        r"\hline",
    ]
    for name, m in all_metrics.items():
        bold = name == "MCF"
        fmt  = lambda v, s: f"${v:.4f} \\pm {s:.4f}$"
        row  = (f"{'\\textbf{' if bold else ''}{name}{'}' if bold else ''} & "
                f"{fmt(m['MSE'][0], m['MSE'][1])} & "
                f"{fmt(m['PSNR'][0], m['PSNR'][1])} & "
                f"{fmt(m['SSIM'][0], m['SSIM'][1])} \\\\")
        lines.append(row)
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Salvo: {tex_path}")

    # Sumário
    sum_path = os.path.join(OUT, "model_summary.txt")
    with open(sum_path, "w") as f:
        f.write("=" * 58 + "\n")
        f.write("  Comparação de Arquiteturas — SBPO V2\n")
        f.write("=" * 58 + "\n")
        f.write(f"  Dataset       : {dataset_name} ({IMG_SIZE}×{IMG_SIZE})\n")
        f.write(f"  Display       : {DISPLAY}×{DISPLAY} (interpolação bicúbica)\n\n")
        f.write(f"  {'Modelo':<10} {'Parâm':>10} {'Descrição'}\n")
        f.write("  " + "-"*54 + "\n")
        descs = {
            "CNN": ("U-Net → Conv 1×1 → ab direto",),
            "GNN": ("U-Net → 3× WeightedGraphConv → ab",),
            "MCF": ("U-Net → QP Min-Cost Flow (32×32) → ab",),
        }
        # (parâmetros serão preenchidos no main)
        for name, desc in descs.items():
            f.write(f"  {name:<10} {'?':>10}   {desc[0]}\n")
        f.write("=" * 58 + "\n")
    print(f"  Salvo: {sum_path}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full",    action="store_true")
    p.add_argument("--epochs",  type=int,   default=5)
    p.add_argument("--batch",   type=int,   default=8)
    p.add_argument("--lr",      type=float, default=1e-4)
    p.add_argument("--data",    type=str,   default="../data")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Nº de amostras individuais salvas como PNG")
    p.add_argument("--subset",  type=int,   default=1500)
    args = p.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    print("=" * 60)
    print("  Pipeline V2 — CNN vs GNN vs MCF (SBPO)")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Épocas  : {args.epochs}  |  Batch: {args.batch}")
    print(f"  Saída   : {OUT}/\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[1/7] Carregando dataset...")
    subset = 0 if args.full else args.subset
    train_ds, val_ds, ds_name = load_datasets(args.data, IMG_SIZE, subset)

    train_ld = torch.utils.data.DataLoader(
        train_ds, args.batch, shuffle=True,  num_workers=0)
    val_ld   = torch.utils.data.DataLoader(
        val_ds,   args.batch, shuffle=False, num_workers=0)

    # ── Modelos ───────────────────────────────────────────────────────────
    print("\n[2/7] Inicializando modelos...")
    cnn = CNNColorizer(base_ch=32).to(DEVICE)
    gnn = GNNColorizer(base_ch=32, gnn_layers=3).to(DEVICE)
    mcf = MCFColorizer(base_ch=32, patch=32, lam=1.0).to(DEVICE)

    for name, m in [("CNN", cnn), ("GNN", gnn), ("MCF", mcf)]:
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name}: {n:,} parâmetros")

    # ── Treinamento ───────────────────────────────────────────────────────
    print("\n[3/7] Treinando...")
    histories = {}
    configs = [("CNN", cnn, False), ("GNN", gnn, False), ("MCF", mcf, True)]
    for name, model, is_mcf in configs:
        tr, vl = train_model(model, name, train_ld, val_ld,
                             DEVICE, args.epochs, args.lr, is_mcf)
        histories[name] = (tr, vl)

    # Dict para visualização
    models_dict = {
        "CNN": (cnn, False),
        "GNN": (gnn, False),
        "MCF": (mcf, True),
    }

    # ── Métricas ──────────────────────────────────────────────────────────
    print("\n[4/7] Calculando métricas (300 amostras)...")
    all_metrics = {}
    for name, (model, is_mcf) in models_dict.items():
        m = compute_metrics(model, val_ds, DEVICE, n=300, is_mcf=is_mcf)
        all_metrics[name] = m
        print(f"  {name}: MSE={m['MSE'][0]:.4f}  "
              f"PSNR={m['PSNR'][0]:.2f}dB  SSIM={m['SSIM'][0]:.3f}")

    # ── Amostras individuais ───────────────────────────────────────────────
    print(f"\n[5/7] Salvando {args.n_samples} amostras individuais (300 DPI)...")
    sample_indices = list(range(args.n_samples))
    for idx in sample_indices:
        path = os.path.join(SFIGS, f"sample_{idx:03d}.png")
        plot_single_sample(models_dict, val_ds, idx, DEVICE, path)
        print(f"  Salvo: {path}")

    # Grade resumo
    print("\n[6/7] Gerando figuras...")
    print("  → Grade qualitativa")
    plot_summary_grid(models_dict, val_ds, list(range(6)), DEVICE)

    print("  → Curvas de treinamento")
    plot_training_curves(histories)

    print("  → Barras de métricas")
    plot_metrics_bars(all_metrics)

    print("  → Radar chart")
    plot_radar(all_metrics)

    print("  → Distribuição de cores")
    plot_color_distributions(models_dict, val_ds, DEVICE)

    # ── Tabelas e sumário ─────────────────────────────────────────────────
    print("\n[7/7] Salvando tabelas LaTeX e CSV...")
    save_outputs(all_metrics, histories, ds_name)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  Concluído em {elapsed/60:.1f} min")
    print(f"\n  Figuras individuais : {OUT}/figures/samples/")
    print(f"  Figuras gerais      : {OUT}/figures/")
    print(f"  Tabela LaTeX        : {OUT}/table_comparison.tex")
    print(f"  Métricas CSV        : {OUT}/metrics_all.csv")
    print()
    print("  Cole no artigo:")
    print(r"    \input{paper_output_v2/table_comparison.tex}")
    print(r"    \includegraphics[width=\textwidth]{paper_output_v2/figures/qualitative_grid.png}")
    print("=" * 60)


if __name__ == "__main__":
    main()
