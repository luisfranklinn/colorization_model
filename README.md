# Colorização de Imagens P&B com CNN + Min-Cost Flow

Implementação de um modelo de colorização de imagens em escala de cinza que combina uma **U-Net** (PyTorch) com uma **camada de otimização de Min-Cost Flow diferenciável** (cvxpylayers), proposto como artigo para o **SBPO**.

## Arquitetura

```
Canal L (P&B) → U-Net → Preditor de Custos → QP Min-Cost Flow → Canais ab (cor)
```

| Componente | Descrição |
|---|---|
| `UNetEncoder` | Encoder/decoder com skip connections, recebe canal L normalizado |
| `CostPredictor` | Prevê custos por aresta (horizontal/vertical) + estimativa inicial de cor |
| `FlowLayer` | QP diferenciável via cvxpylayers — minimiza fluxo de custo entre pixels |

### Formulação do Problema de Otimização

```
min  Σ c_h[e]·‖ab[dst] − ab[src]‖²   (arestas horizontais)
   + Σ c_v[e]·‖ab[dst] − ab[src]‖²   (arestas verticais)
   + λ·‖ab_out − ab_init‖²_F          (fidelidade à CNN)
```

O gradiente retropropaga da loss final → `ab_out` → custos `c_h`, `c_v` → pesos da U-Net via diferenciação implícita das condições KKT.

## Resultados (CIFAR-10)

| Método | MSE ↓ | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|--------|
| CNN puro | 0.0061 | 24.13 dB | 0.915 |
| **CNN + MCF (proposto)** | **0.0058** | **24.65 dB** | **0.931** |

Figuras, tabelas LaTeX e métricas completas em [`paper_output/`](paper_output/).

## Instalação

```bash
pip install torch torchvision cvxpy cvxpylayers scikit-image matplotlib Pillow numpy ecos
```

## Uso

```bash
# Pipeline completo para artigo (download automático do CIFAR-10)
python3 run_paper.py

# Dataset completo (60k imagens, resultados melhores)
python3 run_paper.py --full --epochs 20 --batch 16

# Testar pipeline sem dados reais
python3 colorizer_mcf.py --demo
```

## Estrutura

```
├── colorizer_mcf.py        # Modelo: U-Net + Min-Cost Flow
├── run_paper.py            # Pipeline completo: treino + figuras + métricas
├── requirements.txt
├── colorizer_mcf_best.pth  # Pesos treinados
└── paper_output/
    ├── figures/
    │   ├── 01_training_curves.png   # Curvas de aprendizado
    │   ├── 02_sample_grid.png       # Comparação visual P&B → CNN → MCF → Original
    │   ├── 03_cost_maps.png         # Mapas de custo aprendidos pela CNN
    │   ├── 04_ablation.png          # Ablação CNN puro vs CNN + MCF
    │   └── 05_lab_colorspace.png    # Diagrama do espaço CIE Lab
    ├── metrics.csv
    ├── metrics_table.tex            # Tabela pronta para LaTeX
    └── model_summary.txt
```

## Por que Min-Cost Flow? (argumento SBPO)

- **Hibridização PO + DL**: estrutura de Pesquisa Operacional embutida como camada neural diferenciável
- **Garantia de restrições**: o solver ECOS garante factibilidade a cada forward pass
- **Diferenciabilidade**: cvxpylayers diferencia implicitamente nas condições KKT do QP
- **Interpretabilidade**: custos de aresta altos = bordas de textura preservadas; baixos = regiões homogêneas coloridas livremente
