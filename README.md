# Colorização P&B — V2: CNN vs GNN vs MCF

Extensão da V1 com **três arquiteturas comparativas** e **visualizações de alta qualidade** para publicação no SBPO.

## Novidades em relação à V1

| | V1 | V2 |
|---|---|---|
| Modelos | MCF apenas | CNN + GNN + MCF |
| Dataset | CIFAR-10 (32×32) | **STL-10 (96×96 → 64×64)** |
| Figuras | Grades pequenas | **PNG individual por amostra (192×192, bicúbico)** |
| Comparação | Ablação simples | **Radar chart + barras com desvio padrão** |

## Arquiteturas

### A) CNN Baseline
```
L (64×64) → U-Net → Conv 1×1 → ab (64×64)
```
Previsão direta sem nenhuma restrição estrutural.

### B) GNN Colorizer
```
L (64×64) → U-Net → 3× WeightedGraphConv → ab (64×64)
```
Cada camada de GNN aprende **pesos de aresta** entre pixels adjacentes e propaga cor via message passing — análogo ao MCF, mas sem solver externo.

**WeightedGraphConv** (sem dependências externas — PyTorch puro):
```
w_ij = sigmoid(MLP([f_i ‖ f_j]))           ← peso de aresta aprendido
m_i  = Σ_{j∈N(i)} w_ij · f_j / Z_i        ← mensagem agregada
h_i  = ReLU(W · [f_i ‖ m_i])              ← atualização do nó
```

### C) MCF Colorizer (proposto)
```
L (64×64) → U-Net → ↓32×32 → QP Min-Cost Flow → ↑64×64 → ab
```
Formulação:
```
min  Σ c_h[e]·‖Δab_h‖²  +  Σ c_v[e]·‖Δab_v‖²  +  λ·‖ab − ab_init‖²
```
Backprop via diferenciação implícita das condições KKT (cvxpylayers/ECOS).

## Resultados (STL-10, 2 épocas de demonstração)

| Método | MSE ↓ | PSNR ↑ | SSIM ↑ |
|--------|--------|--------|--------|
| CNN    | 0.0062 | 23.02 dB | 0.875 |
| GNN    | 0.0066 | 22.97 dB | 0.872 |
| **MCF (proposto)** | **0.0055** | **24.17 dB** | **0.922** |

## Instalação

```bash
pip install torch torchvision cvxpy cvxpylayers scikit-image matplotlib Pillow numpy ecos
```

## Uso

```bash
# Rápido (subconjunto de 1500 imagens)
python3 run_paper_v2.py

# Dataset completo STL-10 (100k imagens)
python3 run_paper_v2.py --full --epochs 20 --batch 16

# Controle de amostras individuais salvas
python3 run_paper_v2.py --n-samples 20
```

## Saída

```
paper_output_v2/
├── figures/
│   ├── samples/
│   │   ├── sample_000.png      ← P&B | CNN | GNN | MCF | Original (300 DPI)
│   │   ├── sample_001.png
│   │   └── ...
│   ├── qualitative_grid.png    ← grade resumo para o artigo
│   ├── training_curves.png     ← curvas treino/val dos 3 modelos
│   ├── metrics_bars.png        ← barras MSE/PSNR/SSIM com desvio padrão
│   ├── metrics_radar.png       ← radar chart comparativo
│   └── color_distributions.png ← histograma canais a e b
├── metrics_all.csv
├── table_comparison.tex        ← tabela LaTeX pronta
└── model_summary.txt
```

## No LaTeX do artigo

```latex
\input{paper_output_v2/table_comparison.tex}
\includegraphics[width=\textwidth]{paper_output_v2/figures/qualitative_grid.png}
\includegraphics[width=0.48\textwidth]{paper_output_v2/figures/metrics_bars.png}
\includegraphics[width=0.48\textwidth]{paper_output_v2/figures/metrics_radar.png}
```
