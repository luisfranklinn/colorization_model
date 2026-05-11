"""
Colorização de Imagens P&B com CNN + Camada de Min-Cost Flow
=============================================================
Dataset  : CIFAR-10 (baixado automaticamente via torchvision — 60 mil imagens 32×32)
Arquitetura:
  - Encoder U-Net (PyTorch): recebe canal L do espaço CIE Lab
  - Preditor de Custos: gera custos por aresta (H/V) e cor inicial ab
  - Camada de Otimização (cvxpylayers): QP diferenciável de Min-Cost Flow
  - Treinamento SSL: MSE entre cor otimizada e cor real

Uso:
  python colorizer_mcf.py            # baixa CIFAR-10 e treina
  python colorizer_mcf.py --demo     # só testa forward+backward sem dados reais

Dependências:
  pip install torch torchvision cvxpy cvxpylayers scikit-image Pillow numpy
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from skimage import color as skcolor


# ===========================================================================
# A. ARQUITETURA DO MODELO
# ===========================================================================

class DoubleConv(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetEncoder(nn.Module):
    """
    U-Net simplificada.
    Entrada : canal L normalizado em [-1, 1]  →  [B, 1, H, W]
    Saída   : mapa de features               →  [B, base_ch, H, W]
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        c = base_ch
        self.enc1     = DoubleConv(1,    c)
        self.enc2     = DoubleConv(c,    c*2)
        self.enc3     = DoubleConv(c*2,  c*4)
        self.bottleneck = DoubleConv(c*4, c*8)
        self.up3      = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3     = DoubleConv(c*8,  c*4)
        self.up2      = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2     = DoubleConv(c*4,  c*2)
        self.up1      = nn.ConvTranspose2d(c*2, c,   2, stride=2)
        self.dec1     = DoubleConv(c*2,  c)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1   # [B, base_ch, H, W]


class CostPredictor(nn.Module):
    """
    Latent Cost Predictor — dois heads a partir do mesmo mapa de features:
      edge_head  → custos c_h (horizontal) e c_v (vertical)  via Softplus
      color_head → estimativa inicial ab                      via Tanh
    """
    def __init__(self, in_ch: int = 32):
        super().__init__()
        mid = in_ch // 2
        self.edge_head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1),    nn.Softplus(),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 2, 1),    nn.Tanh(),
        )

    def forward(self, feat):
        return self.edge_head(feat), self.color_head(feat)


# ===========================================================================
# B. CAMADA DE MIN-COST FLOW DIFERENCIÁVEL
# ===========================================================================

def build_flow_layer(H: int, W: int, lam: float = 1.0):
    """
    Constrói um QP diferenciável via cvxpylayers.

    Formulação:
        min  Σ_e c_h[e]·‖ab[dst]−ab[src]‖²   (arestas horizontais)
           + Σ_e c_v[e]·‖ab[dst]−ab[src]‖²   (arestas verticais)
           + λ·‖ab_out − ab_init‖²_F          (fidelidade à CNN)

    Semântica de fluxo:
        c_ij alto  →  borda de textura  →  cor não atravessa (descontinuidade)
        c_ij baixo →  região homogênea  →  cor se propaga livremente

    Diferenciabilidade:
        cvxpylayers usa diferenciação implícita nas condições KKT,
        propagando ∂loss/∂c_h, ∂loss/∂c_v → pesos da U-Net.
    """
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    n   = H * W
    n_h = H * (W - 1)
    n_v = (H - 1) * W

    c_h        = cp.Parameter((n_h,), nonneg=True)
    c_v        = cp.Parameter((n_v,), nonneg=True)
    color_init = cp.Parameter((n, 2))
    color_out  = cp.Variable((n, 2))

    h_src = [i * W + j     for i in range(H) for j in range(W - 1)]
    h_dst = [i * W + j + 1 for i in range(H) for j in range(W - 1)]
    v_src = [ i      * W + j for i in range(H - 1) for j in range(W)]
    v_dst = [(i + 1) * W + j for i in range(H - 1) for j in range(W)]

    diff_h = color_out[h_dst, :] - color_out[h_src, :]
    diff_v = color_out[v_dst, :] - color_out[v_src, :]

    cost = (cp.sum(cp.multiply(c_h[:, None], cp.square(diff_h)))
          + cp.sum(cp.multiply(c_v[:, None], cp.square(diff_v)))
          + lam * cp.sum_squares(color_out - color_init))

    layer = CvxpyLayer(
        cp.Problem(cp.Minimize(cost)),
        parameters=[c_h, c_v, color_init],
        variables=[color_out],
    )
    return layer, (h_src, h_dst, v_src, v_dst)


class FlowColorizer(nn.Module):
    """
    Pipeline completo:
        L (grayscale) → U-Net → Preditor de Custos → Min-Cost Flow → ab otimizado
    """
    def __init__(self, height: int = 32, width: int = 32,
                 base_ch: int = 32, lam: float = 1.0):
        super().__init__()
        self.H, self.W = height, width
        self.cnn            = UNetEncoder(base_ch)
        self.cost_predictor = CostPredictor(base_ch)
        self.flow_layer, self._eidx = build_flow_layer(height, width, lam)

    def forward(self, x):
        """
        x        : [B, 1, H, W]  canal L normalizado em [-1, 1]
        retorna  : ab_opt  [B, 2, H, W]  cores refinadas pelo QP
                   ab_init [B, 2, H, W]  estimativa direta da CNN
        """
        B, _, H, W = x.shape
        feat          = self.cnn(x)
        costs, ab_init = self.cost_predictor(feat)

        c_h_px = costs[:, 0].reshape(B, -1)
        c_v_px = costs[:, 1].reshape(B, -1)
        h_src, h_dst, v_src, v_dst = self._eidx

        c_h    = (c_h_px[:, h_src] + c_h_px[:, h_dst]) / 2
        c_v    = (c_v_px[:, v_src] + c_v_px[:, v_dst]) / 2
        ab_flat = ab_init.permute(0, 2, 3, 1).reshape(B, H * W, 2)

        # cvxpylayers/ECOS é CPU-only; convertemos para float64 + CPU antes e
        # devolvemos os resultados para o device original depois
        device = x.device
        c_h64     = c_h.double().cpu()
        c_v64     = c_v.double().cpu()
        ab_flat64 = ab_flat.double().cpu()

        results = []
        for b in range(B):
            (ab_b,) = self.flow_layer(
                c_h64[b], c_v64[b], ab_flat64[b],
                solver_args={"solve_method": "SCS"},
            )
            results.append(ab_b.float().to(device))  # volta para float32 e device original

        ab_opt = (torch.stack(results)           # [B, n, 2]
                      .reshape(B, H, W, 2)
                      .permute(0, 3, 1, 2))       # [B, 2, H, W]
        return ab_opt, ab_init


# ===========================================================================
# C. DATASET — CIFAR-10 (download automático)
# ===========================================================================

class CIFAR10LabDataset(torch.utils.data.Dataset):
    """
    Wrapper do CIFAR-10 que devolve (L, ab) no espaço CIE Lab.

    CIFAR-10: 60 000 imagens coloridas 32×32 — nenhuma anotação necessária.
    O modelo aprende a colorir de forma auto-supervisionada.
    """
    def __init__(self, root: str = "./data", train: bool = True):
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True,
            transform=transforms.ToTensor(),   # [3, 32, 32] ∈ [0,1]
        )

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        img_t, _ = self.cifar[idx]                       # ignoramos o label
        rgb = img_t.permute(1, 2, 0).numpy()             # [32, 32, 3]
        lab = skcolor.rgb2lab(rgb)                        # L∈[0,100] ab∈[-128,127]

        L  = torch.tensor(lab[:, :, 0] / 50.0 - 1.0,
                          dtype=torch.float32).unsqueeze(0)       # [1, 32, 32]
        ab = torch.tensor(lab[:, :, 1:] / 128.0,
                          dtype=torch.float32).permute(2, 0, 1)   # [2, 32, 32]
        return L, ab


# ===========================================================================
# TREINAMENTO
# ===========================================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for i, (L, ab_true) in enumerate(loader):
        L, ab_true = L.to(device), ab_true.to(device)
        optimizer.zero_grad()

        ab_opt, ab_init = model(L)

        loss = (F.mse_loss(ab_opt,  ab_true)
              + 0.5 * F.mse_loss(ab_init, ab_true))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total += loss.item()
        if i % 50 == 0:
            print(f"  [{i:5d}/{len(loader)}] loss={loss.item():.4f}")
    return total / len(loader)


# ===========================================================================
# INFERÊNCIA
# ===========================================================================

@torch.no_grad()
def colorize_cifar_sample(model, dataset, idx, device):
    """Coloriza a amostra `idx` do dataset e retorna (original, colorizada) como arrays RGB."""
    model.eval()
    L, _ = dataset[idx]
    L_in = L.unsqueeze(0).to(device)

    ab_opt, _ = model(L_in)
    ab_np = ab_opt[0].permute(1, 2, 0).cpu().numpy() * 128.0

    L_np = (L.squeeze().numpy() + 1.0) * 50.0
    lab  = np.concatenate([L_np[:, :, None], ab_np], axis=-1)
    rgb  = np.clip(skcolor.lab2rgb(lab), 0, 1)

    # Desempacota Subset se necessário
    base_ds   = dataset.dataset if isinstance(dataset, torch.utils.data.Subset) else dataset
    real_idx  = dataset.indices[idx] if isinstance(dataset, torch.utils.data.Subset) else idx
    rgb_orig_t, _ = base_ds.cifar[real_idx]
    rgb_orig  = rgb_orig_t.permute(1, 2, 0).numpy()

    return rgb_orig, rgb


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",       action="store_true",
                        help="Executa apenas um forward+backward sintético")
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--batch",      type=int, default=8)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--lam",        type=float, default=1.0,
                        help="Peso λ da fidelidade no QP")
    parser.add_argument("--data",       type=str, default="./data")
    parser.add_argument("--save",       type=str, default="colorizer_mcf_best.pth")
    parser.add_argument("--subset",     type=int, default=0,
                        help="Limitar a N amostras para testes rápidos (0 = tudo)")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {DEVICE}")

    model = FlowColorizer(height=32, width=32, base_ch=32, lam=args.lam).to(DEVICE)
    print(f"Parâmetros treináveis : {sum(p.numel() for p in model.parameters()):,}")

    # ── Demo sintético ──────────────────────────────────────────────────────
    if args.demo:
        print("\nModo demo: forward + backward sintético...")
        L  = torch.randn(2, 1, 32, 32).to(DEVICE)
        ab = torch.randn(2, 2, 32, 32).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        model.train()
        ab_opt, ab_init = model(L)
        loss = F.mse_loss(ab_opt, ab) + 0.5 * F.mse_loss(ab_init, ab)
        loss.backward()
        opt.step()
        grad_ok = model.cnn.enc1.net[0].weight.grad is not None
        print(f"Forward OK  | ab_opt: {tuple(ab_opt.shape)}")
        print(f"Loss       : {loss.item():.4f}")
        print(f"Grad CNN   : {'SIM ✓  (backprop atravessa o QP)' if grad_ok else 'NÃO ✗'}")
        return

    # ── Treinamento com CIFAR-10 ─────────────────────────────────────────────
    print(f"\nCarregando CIFAR-10 em '{args.data}' (download automático se necessário)...")
    train_ds = CIFAR10LabDataset(root=args.data, train=True)
    val_ds   = CIFAR10LabDataset(root=args.data, train=False)

    if args.subset > 0:
        train_ds = torch.utils.data.Subset(train_ds, range(args.subset))
        val_ds   = torch.utils.data.Subset(val_ds,   range(min(args.subset // 5, len(val_ds))))
        print(f"Subconjunto: {args.subset} treino / {len(val_ds)} validação")
    else:
        print(f"Dataset    : {len(train_ds)} treino / {len(val_ds)} validação")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\n── Época {epoch}/{args.epochs} ─────────────────────────────")
        tr_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)

        # Validação rápida
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for L, ab_true in val_loader:
                L, ab_true = L.to(DEVICE), ab_true.to(DEVICE)
                ab_opt, ab_init = model(L)
                val_loss += (F.mse_loss(ab_opt,  ab_true)
                           + 0.5 * F.mse_loss(ab_init, ab_true)).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"  treino={tr_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.save)
            print(f"  → melhor modelo salvo em '{args.save}'")

    print(f"\nTreinamento concluído. Melhor val loss: {best_val:.4f}")

    # Amostras visuais salvas como NPY para inspeção
    print("\nSalvando 4 amostras de colorização em './samples/'...")
    os.makedirs("./samples", exist_ok=True)
    model.load_state_dict(torch.load(args.save, map_location=DEVICE))
    for i in range(4):
        orig, colorized = colorize_cifar_sample(model, val_ds, i, DEVICE)
        np.save(f"./samples/orig_{i}.npy",      orig)
        np.save(f"./samples/colorized_{i}.npy", colorized)
    print("Amostras salvas. Carregue com np.load() para visualizar.")


if __name__ == "__main__":
    main()
