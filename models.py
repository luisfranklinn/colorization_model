"""
models.py — Três arquiteturas para comparação no artigo SBPO V2
===============================================================
  A) CNNColorizer   — U-Net → previsão direta de ab
  B) GNNColorizer   — U-Net → message passing em grade de pixels → ab
  C) MCFColorizer   — U-Net → Min-Cost Flow (QP diferenciável) → ab

Imagens: 64×64 (STL-10 redimensionado)
MCF     : opera em patches 32×32 (downscale/upscale)
GNN     : opera em grade completa 64×64 via convolução ponderada
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# BLOCO BASE
# ===========================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


# ===========================================================================
# BACKBONE COMPARTILHADO — U-Net
# ===========================================================================

class SharedEncoder(nn.Module):
    """
    U-Net com 3 níveis de downscaling.
    Entrada : [B, 1, H, W] — canal L normalizado em [-1, 1]
    Saída   : [B, base_ch, H, W] — features em resolução original
    """
    def __init__(self, base_ch=32):
        super().__init__()
        c = base_ch
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(1,   c)
        self.enc2 = DoubleConv(c,   c*2)
        self.enc3 = DoubleConv(c*2, c*4)
        self.bot  = DoubleConv(c*4, c*8)
        self.up3  = nn.ConvTranspose2d(c*8, c*4, 2, stride=2)
        self.dec3 = DoubleConv(c*8, c*4)
        self.up2  = nn.ConvTranspose2d(c*4, c*2, 2, stride=2)
        self.dec2 = DoubleConv(c*4, c*2)
        self.up1  = nn.ConvTranspose2d(c*2, c,   2, stride=2)
        self.dec1 = DoubleConv(c*2, c)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b),  e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return d1


# ===========================================================================
# MODELO A — CNN BASELINE
# ===========================================================================

class CNNColorizer(nn.Module):
    """
    Baseline: U-Net → previsão direta dos canais a e b.
    Sem nenhuma camada de otimização.
    """
    def __init__(self, base_ch=32):
        super().__init__()
        self.encoder = SharedEncoder(base_ch)
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 2, 1),
            nn.Tanh(),
        )

    def forward(self, L):
        feat = self.encoder(L)
        return self.head(feat)   # [B, 2, H, W]


# ===========================================================================
# MODELO B — GNN COLORIZER
# ===========================================================================

class WeightedGraphConv(nn.Module):
    """
    Uma camada de Graph Convolution em grade 4-conexa.

    Formulação:
        Para cada pixel i com vizinhos N(i) = {cima, baixo, esq, dir}:
            w_ij = sigmoid(MLP([f_i ‖ f_j]))      ← peso de aresta aprendido
            m_i  = Σ_{j∈N(i)} w_ij · f_j / Z_i   ← mensagem agregada
            h_i  = ReLU(W · [f_i ‖ m_i])          ← atualização do nó

    Implementação via convolução agrupada (sem sparse ops — apenas PyTorch puro).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # MLP de peso de aresta (compartilhado entre as 4 direções)
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
            nn.Sigmoid(),
        )
        # Transformação de atualização do nó
        self.update = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _shift(self, x, direction):
        """Desloca o tensor para obter vizinhos sem padding circular."""
        if direction == "right":
            return F.pad(x[:, :, :, :-1], (1, 0))
        if direction == "left":
            return F.pad(x[:, :, :, 1:],  (0, 1))
        if direction == "down":
            return F.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        if direction == "up":
            return F.pad(x[:, :, 1:,  :], (0, 0, 0, 1))

    def forward(self, x):
        dirs = ["right", "left", "down", "up"]
        weighted = []
        total_w  = torch.zeros_like(x[:, :1])  # acumula pesos para normalização

        for d in dirs:
            neighbor = self._shift(x, d)
            w = self.edge_mlp(torch.cat([x, neighbor], 1))  # [B, C, H, W]
            weighted.append(w * neighbor)
            total_w = total_w + w.mean(dim=1, keepdim=True)

        # Mensagem agregada (normalizada)
        msg = sum(weighted) / (total_w + 1e-8)

        return self.update(torch.cat([x, msg], 1))  # [B, out_ch, H, W]


class GNNColorizer(nn.Module):
    """
    U-Net → 3 camadas de GNN em grade de pixels → ab.

    A GNN aprende quais pixels devem "trocar" cor com seus vizinhos,
    de forma análoga ao fluxo de cor no MCF — mas via aprendizado de fim a fim
    sem solver externo.
    """
    def __init__(self, base_ch=32, gnn_layers=3):
        super().__init__()
        self.encoder = SharedEncoder(base_ch)
        c = base_ch
        self.gnn = nn.ModuleList(
            [WeightedGraphConv(c, c) for _ in range(gnn_layers)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(c, c // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, 2, 1),
            nn.Tanh(),
        )

    def forward(self, L):
        x = self.encoder(L)
        for layer in self.gnn:
            x = layer(x)
        return self.head(x)  # [B, 2, H, W]


# ===========================================================================
# MODELO C — MCF COLORIZER (patch 32×32)
# ===========================================================================

def _build_flow_layer(H, W, lam=1.0):
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer

    n   = H * W
    n_h = H * (W - 1)
    n_v = (H - 1) * W

    c_h   = cp.Parameter((n_h,), nonneg=True)
    c_v   = cp.Parameter((n_v,), nonneg=True)
    c_init = cp.Parameter((n, 2))
    c_out  = cp.Variable((n, 2))

    h_src = [i*W+j   for i in range(H) for j in range(W-1)]
    h_dst = [i*W+j+1 for i in range(H) for j in range(W-1)]
    v_src = [ i*W+j  for i in range(H-1) for j in range(W)]
    v_dst = [(i+1)*W+j for i in range(H-1) for j in range(W)]

    dh = c_out[h_dst, :] - c_out[h_src, :]
    dv = c_out[v_dst, :] - c_out[v_src, :]

    obj = (cp.sum(cp.multiply(c_h[:, None], cp.square(dh)))
         + cp.sum(cp.multiply(c_v[:, None], cp.square(dv)))
         + lam * cp.sum_squares(c_out - c_init))

    layer = CvxpyLayer(
        cp.Problem(cp.Minimize(obj)),
        parameters=[c_h, c_v, c_init],
        variables=[c_out],
    )
    return layer, (h_src, h_dst, v_src, v_dst)


class MCFColorizer(nn.Module):
    """
    U-Net → Min-Cost Flow (QP diferenciável via cvxpylayers) → ab.

    Para imagens 64×64: downsample para 32×32 → MCF → upsample para 64×64.
    O QP garante que o fluxo de cor respeite as restrições de suavidade
    ponderadas pelos custos de aresta previstos pela CNN.
    """
    def __init__(self, base_ch=32, patch=32, lam=1.0):
        super().__init__()
        self.patch = patch
        self.encoder      = SharedEncoder(base_ch)
        self.cost_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 2, 1),       nn.Softplus(),
        )
        self.color_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 2, 1),       nn.Tanh(),
        )
        self.flow_layer, self._eidx = _build_flow_layer(patch, patch, lam)

    def forward(self, L):
        B, _, H, W = L.shape
        feat  = self.encoder(L)                         # [B, C, H, W]
        costs = self.cost_head(feat)                    # [B, 2, H, W]
        ab_init = self.color_head(feat)                 # [B, 2, H, W]

        # Downsample para 32×32
        P = self.patch
        costs_p   = F.interpolate(costs,   (P, P), mode="bilinear", align_corners=False)
        ab_init_p = F.interpolate(ab_init, (P, P), mode="bilinear", align_corners=False)

        c_h_px = costs_p[:, 0].reshape(B, -1)
        c_v_px = costs_p[:, 1].reshape(B, -1)
        h_src, h_dst, v_src, v_dst = self._eidx

        c_h = ((c_h_px[:, h_src] + c_h_px[:, h_dst]) / 2).double()
        c_v = ((c_v_px[:, v_src] + c_v_px[:, v_dst]) / 2).double()
        ab_flat = ab_init_p.permute(0, 2, 3, 1).reshape(B, P*P, 2).double()

        results = []
        for b in range(B):
            (ab_b,) = self.flow_layer(
                c_h[b], c_v[b], ab_flat[b],
                solver_args={"solve_method": "ECOS"},
            )
            results.append(ab_b.float())

        ab_opt_p = (torch.stack(results)
                        .reshape(B, P, P, 2)
                        .permute(0, 3, 1, 2))            # [B, 2, P, P]

        # Upsample de volta para H×W
        ab_opt = F.interpolate(ab_opt_p, (H, W), mode="bilinear", align_corners=False)
        return ab_opt, ab_init
