"""
Análise comparativa de um modelo Cool-Chic treinado:
  - Imagem original
  - Imagem decodificada (após encoder → decoder)
  - Todos os latentes (grades hierárquicas ŷ^0 … ŷ^N)
  - Distribuições de μ e scale produzidas pelo ARM

Uso:
    python coolchic_analysis.py \\
        --model  workdir/frame_encoder.pt \\
        --input  samples/images/othim01.png \\
        --save   analysis.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Cool-Chic imports
from coolchic.component.frame import load_frame_encoder
from coolchic.component.core.arm import compute_rate
from coolchic.io.io import load_frame_data_from_file


# ─────────────────────────────────────────────────
# Helpers e Paleta
# ─────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
ACCENT  = "#58a6ff"
GREEN   = "#3fb950"
ORANGE  = "#f78166"
PURPLE  = "#bc8cff"
YELLOW  = "#e3b341"
MUTED   = "#8b949e"
WHITE   = "#e6edf3"
COLORS  = [ACCENT, GREEN, ORANGE, PURPLE, YELLOW, "#79c0ff", "#ffa657", "#ff7b72"]

# Zonas ERP: (nome, cor, fracao_inicio, fracao_fim)
ERP_ZONES = [
    ("Polo Norte", "#74b9ff", 0.00, 0.25),
    ("Equador",    "#fdcb6e", 0.25, 0.75),
    ("Polo Sul",   "#fd79a8", 0.75, 1.00),
]

def split_erp_zones(tensor: torch.Tensor) -> list:
    """Divide um tensor [..., H, W] em 3 zonas ERP."""
    H = tensor.shape[-2]
    resultado = []
    for nome, cor, f0, f1 in ERP_ZONES:
        r0 = int(H * f0)
        r1 = int(H * f1)
        zona = tensor[..., r0:r1, :]
        resultado.append((nome, cor, zona))
    return resultado

def load_image(path: str) -> torch.Tensor:
    """Imagem do disco → [1, C, H, W] float32 em [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def tensor_stats(t: torch.Tensor) -> dict:
    """Estatísticas descritivas de um tensor qualquer."""
    f = t.flatten().float()
    return {
        "min"   : f.min().item(),
        "max"   : f.max().item(),
        "mean"  : f.mean().item(),
        "std"   : f.std().item(),
        "median": f.median().item(),
        "q25"   : f.quantile(0.25).item(),
        "q75"   : f.quantile(0.75).item(),
    }


def laplace_pdf(x_np: np.ndarray, mu: float, b: float, n_pts: int = 300) -> tuple:
    """Densidade da distribuição de Laplace para overlay no histograma."""
    xs = np.linspace(x_np.min(), x_np.max(), n_pts)
    ys = (1 / (2 * b)) * np.exp(-np.abs(xs - mu) / b)
    # Normalizar para a mesma escala do histograma
    return xs, ys


def style_ax(ax, xlabel="", ylabel="Frequência"):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=7)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=7)


def hist_on(ax, data_np: np.ndarray, color: str, label: str,
            n_bins: int = 128, density: bool = False) -> tuple:
    counts, edges = np.histogram(data_np, bins=n_bins, density=density)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, counts, alpha=0.35, color=color)
    ax.plot(centers, counts, color=color, linewidth=1.0, label=label)
    return counts, centers


def vlines_stats(ax, data_np: np.ndarray, color: str):
    mu  = data_np.mean()
    std = data_np.std()
    ax.axvline(mu,         color=WHITE,  linewidth=1.2, linestyle="--",
               label=f"μ={mu:.3f}")
    ax.axvline(mu - std, color=color, linewidth=0.8, linestyle=":",
               alpha=0.85, label=f"σ={std:.3f}")
    ax.axvline(mu + std, color=color, linewidth=0.8, linestyle=":", alpha=0.85)


# ─────────────────────────────────────────────────
# Extração de dados do modelo
# ─────────────────────────────────────────────────

@torch.no_grad()
def extract_model_data(frame_encoder, original_img: torch.Tensor) -> dict:
    """
    Executa um forward pass com hardround (sem ruído de treino)
    e retorna um dicionário com todos os tensores de interesse.
    """
    frame_encoder.eval()
    cc_enc = frame_encoder.coolchic_enc["residue"]

    # 1. Latentes quantizados (decoder-side)
    latents = cc_enc.get_quantize_latent(
        quantizer_noise_type="none",
        quantizer_type="hardround",
    )

    # 2. Contexto e saída do ARM (μ e scale)
    flat_latent, flat_context = cc_enc.get_latent_context(latents)
    raw_arm_out = cc_enc.arm(flat_context)                       # [B, 2]
    flat_mu, flat_scale = cc_enc.arm.reparameterize_output(raw_arm_out)
    flat_rate = compute_rate(flat_latent, flat_mu, flat_scale)

    # 3. Imagem decodificada
    result = frame_encoder(
        quantizer_noise_type="none",
        quantizer_type="hardround",
        flag_additional_outputs=False,
    )
    decoded_img = result.decoded_image  # [1, C, H, W] em [0,1]

    return {
        "latents"    : latents,         # List[Tensor [1,1,Hi,Wi]]
        "flat_latent": flat_latent,     # [B]
        "flat_mu"    : flat_mu,         # [B]
        "flat_scale" : flat_scale,      # [B]  (já exp-reparametrizado)
        "flat_rate"  : flat_rate,       # [B]  bits por símbolo
        "decoded_img": decoded_img,     # [1,C,H,W]
        "original_img": original_img,   # [1,C,H,W]
    }


# ─────────────────────────────────────────────────
# Impressão das estatísticas
# ─────────────────────────────────────────────────

def print_report(data: dict) -> None:
    W = 70
    sep = "─" * W

    def row(label, stats):
        print(f"  {label:<28}"
              f"  min={stats['min']:+.4f}"
              f"  max={stats['max']:+.4f}"
              f"  μ={stats['mean']:+.4f}"
              f"  σ={stats['std']:.4f}"
              f"  median={stats['median']:+.4f}")

    print(f"\n{'═'*W}")
    print(f"  ANÁLISE COMPARATIVA — Cool-Chic")
    print(f"{'═'*W}")

    # Imagens
    print(f"\n  {'[ IMAGENS ]'}")
    print(sep)
    orig = data["original_img"].squeeze(0)
    dec  = data["decoded_img"].squeeze(0)

    row("Original (global)",   tensor_stats(orig))
    row("Decodificada (global)", tensor_stats(dec))

    psnr = -10 * torch.log10(F.mse_loss(orig, dec.clamp(0,1))).item()
    print(f"\n  PSNR original ↔ decodificada:  {psnr:.2f} dB")

    for ch, ch_name in enumerate(["R", "G", "B"]):
        if ch < orig.shape[0]:
            row(f"  Original canal {ch_name}", tensor_stats(orig[ch]))
            row(f"  Decodificada canal {ch_name}", tensor_stats(dec[ch]))

    # Latentes
    print(f"\n  {'[ LATENTES QUANTIZADOS ]'}")
    print(sep)
    total_latent_pixels = sum(lat.numel() for lat in data["latents"])
    for i, lat in enumerate(data["latents"]):
        h, w = lat.shape[-2:]
        s = tensor_stats(lat)
        sparsity = (lat == 0).float().mean().item() * 100
        print(f"  Latente [{i}]  {h:4d}×{w:<4d}"
              f"  min={s['min']:+6.1f}  max={s['max']:+6.1f}"
              f"  μ={s['mean']:+.4f}  σ={s['std']:.4f}"
              f"  zeros={sparsity:.1f}%")

    # ARM
    print(f"\n  {'[ SAÍDA DO ARM ]'}")
    print(sep)
    row("μ (expectation)",     tensor_stats(data["flat_mu"]))
    row("scale b (Laplace)",   tensor_stats(data["flat_scale"]))
    row("taxa (bits/símbolo)", tensor_stats(data["flat_rate"]))
    total_bits = data["flat_rate"].sum().item()
    print(f"\n  Taxa total estimada: {total_bits/8/1024:.2f} KB  "
          f"({total_bits:.0f} bits  |  "
          f"{total_bits / total_latent_pixels:.2f} bits/símbolo médio)")
    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────────
# Visualização
# ─────────────────────────────────────────────────

def plot_analysis(data: dict, save_path=None, n_bins: int = 128):
    latents     = data["latents"]
    n_latents   = len(latents)
    orig        = data["original_img"].squeeze(0)   # [C,H,W]
    dec         = data["decoded_img"].squeeze(0).clamp(0, 1)
    flat_mu     = data["flat_mu"].cpu().numpy()
    flat_scale  = data["flat_scale"].cpu().numpy()
    flat_rate   = data["flat_rate"].cpu().numpy()
    flat_latent = data["flat_latent"].cpu().numpy()

    # Layout:
    #  Linha 0: imagem original | imagem decodificada | erro |  [gap]
    #  Linha 1: histograma orig vs decoded (global)
    #  Linha 2: histogramas dos latentes (um por coluna, até 4)
    #  Linha 3: μ, scale (b), taxa bits/símbolo
    n_cols = max(4, n_latents)
    fig = plt.figure(figsize=(n_cols * 3.5, 14), facecolor=BG)
    gs  = gridspec.GridSpec(4, n_cols, figure=fig, hspace=0.55, wspace=0.35)

    TITLE_KW  = dict(color=WHITE, fontsize=9,  fontweight="bold", pad=5)
    LEGEND_KW = dict(fontsize=6, facecolor=PANEL, labelcolor=WHITE,
                     framealpha=0.85, edgecolor="#30363d")

    # ── Linha 0: imagens ────────────────────────────────────────────
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(orig.permute(1, 2, 0).clamp(0,1).numpy())
    ax_orig.set_title("Original", **TITLE_KW)
    ax_orig.axis("off")

    ax_dec = fig.add_subplot(gs[0, 1])
    ax_dec.imshow(dec.permute(1, 2, 0).numpy())
    ax_dec.set_title("Decodificada", **TITLE_KW)
    ax_dec.axis("off")

    ax_err = fig.add_subplot(gs[0, 2])
    err = (orig - dec.clamp(0,1)).abs().mean(0).numpy()
    im = ax_err.imshow(err, cmap="hot", vmin=0, vmax=err.max())
    ax_err.set_title("Mapa de erro |orig−dec|", **TITLE_KW)
    ax_err.axis("off")
    plt.colorbar(im, ax=ax_err, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color=MUTED)

    psnr = -10 * np.log10(((orig.numpy() - dec.numpy())**2).mean())
    ax_dec.set_xlabel(f"PSNR = {psnr:.2f} dB", color=GREEN, fontsize=8)

    # Texto: resumo de taxa no 4º slot
    ax_txt = fig.add_subplot(gs[0, 3] if n_cols > 3 else gs[0, 2])
    ax_txt.axis("off")
    ax_txt.set_facecolor(PANEL)
    total_bits = flat_rate.sum()
    summary = (
        f"Taxa total\n"
        f"{total_bits:.0f} bits\n"
        f"{total_bits/8/1024:.2f} KB\n\n"
        f"Bits/símbolo\n"
        f"μ = {flat_rate.mean():.3f}\n"
        f"σ = {flat_rate.std():.3f}\n\n"
        f"Latentes\n"
        f"N = {n_latents} grids\n"
        f"{sum(lat.numel() for lat in latents):,} símbolos"
    )
    ax_txt.text(0.05, 0.95, summary, color=WHITE, fontsize=8.5,
                va="top", ha="left", transform=ax_txt.transAxes,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=PANEL, edgecolor="#30363d"))

    # ── Linha 1: histograma original vs decodificada ──────────────
    ax_img_hist = fig.add_subplot(gs[1, :])
    ax_img_hist.set_facecolor(PANEL)
    ch_names  = ["R", "G", "B"]
    ch_colors = ["#e74c3c", "#2ecc71", "#3498db"]
    for ch in range(orig.shape[0]):
        o_np = orig[ch].flatten().numpy()
        d_np = dec[ch].flatten().numpy()
        cnt_o, edges = np.histogram(o_np, bins=n_bins, range=(0, 1))
        cnt_d, _     = np.histogram(d_np, bins=n_bins, range=(0, 1))
        centers      = (edges[:-1] + edges[1:]) / 2
        c = ch_colors[ch]
        ax_img_hist.plot(centers, cnt_o, color=c, linewidth=1.2,
                         label=f"Original {ch_names[ch]}")
        ax_img_hist.plot(centers, cnt_d, color=c, linewidth=1.0,
                         linestyle="--", alpha=0.7,
                         label=f"Decodificada {ch_names[ch]}")

    style_ax(ax_img_hist, xlabel="Valor do pixel [0,1]")
    ax_img_hist.set_title("Distribuição de pixels: Original (─) vs Decodificada (- -)",
                           **TITLE_KW)
    ax_img_hist.legend(**LEGEND_KW, ncol=3)

    # ── Linha 2: latentes ────────────────────────────────────────
    # Calcula range global de todos os latentes para escala comparável
    all_lat_vals = np.concatenate([lat.flatten().numpy() for lat in latents])
    lat_min, lat_max = all_lat_vals.min(), all_lat_vals.max()

    for i, lat in enumerate(latents):
        col = i % n_cols
        ax = fig.add_subplot(gs[2, col])
        ax.set_facecolor(PANEL)
        lat_np = lat.flatten().numpy()
        color  = COLORS[i % len(COLORS)]
        h, w   = lat.shape[-2:]

        counts, edges = np.histogram(lat_np, bins=n_bins,
                                     range=(lat_min, lat_max))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.fill_between(centers, counts, alpha=0.4, color=color)
        ax.plot(centers, counts, color=color, linewidth=1.0)

        # Overlay curva de Laplace(0, std) – hipótese do ARM
        std_lat = lat_np.std()
        xs = np.linspace(lat_min, lat_max, 300)
        ys_laplace = (1 / (2 * std_lat + 1e-8)) * np.exp(-np.abs(xs) / (std_lat + 1e-8))
        ys_laplace *= counts.max() / (ys_laplace.max() + 1e-12)
        ax.plot(xs, ys_laplace, color=WHITE, linewidth=0.8,
                linestyle=":", alpha=0.7, label="Laplace(0,σ)")

        mu_l = lat_np.mean()
        ax.axvline(mu_l, color=WHITE, linewidth=1.0, linestyle="--")

        sparsity = (lat_np == 0).mean() * 100
        ax.set_title(f"Latente [{i}]  {h}×{w}\n"
                     f"μ={mu_l:.2f}  σ={std_lat:.2f}  0s={sparsity:.0f}%",
                     color=WHITE, fontsize=7.5, pad=3)
        style_ax(ax, xlabel="ŷ (quantizado)")
        ax.legend(**LEGEND_KW)

    # ── Linha 3: ARM outputs ──────────────────────────────────────
    arm_panels = [
        (flat_mu,     "μ (expectation ARM)",      ACCENT,  "Valor de μ"),
        (flat_scale,  "scale b (Laplace)",        GREEN,   "b = exp(raw − 4)"),
        (flat_rate,   "Taxa (bits / símbolo)",    ORANGE,  "bits"),
    ]
    for j, (arr, title, color, xlabel) in enumerate(arm_panels):
        ax = fig.add_subplot(gs[3, j])
        ax.set_facecolor(PANEL)
        counts, edges = np.histogram(arr, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.fill_between(centers, counts, alpha=0.4, color=color)
        ax.plot(centers, counts, color=color, linewidth=1.0)
        mu_a, std_a = arr.mean(), arr.std()
        ax.axvline(mu_a, color=WHITE, linewidth=1.2, linestyle="--",
                   label=f"μ={mu_a:.3f}")
        ax.axvline(mu_a - std_a, color=color, linewidth=0.8,
                   linestyle=":", alpha=0.8, label=f"σ={std_a:.3f}")
        ax.axvline(mu_a + std_a, color=color, linewidth=0.8,
                   linestyle=":", alpha=0.8)
        ax.set_title(title, **TITLE_KW)
        style_ax(ax, xlabel=xlabel)
        ax.legend(**LEGEND_KW)

    # Painel extra: μ vs ŷ (scatter amostrado)
    ax_sc = fig.add_subplot(gs[3, 3] if n_cols > 3 else gs[3, 2])
    ax_sc.set_facecolor(PANEL)
    sample_size = min(5000, len(flat_mu))
    idx = np.random.choice(len(flat_mu), sample_size, replace=False)
    ax_sc.scatter(flat_latent[idx], flat_mu[idx],
                  s=1.5, alpha=0.25, color=PURPLE, rasterized=True)
    lim = max(abs(flat_latent).max(), abs(flat_mu).max()) * 1.05
    ax_sc.plot([-lim, lim], [-lim, lim], color=WHITE, linewidth=0.8,
               linestyle="--", alpha=0.6, label="ŷ = μ (ideal)")
    ax_sc.set_title("ŷ vs μ (amostrado)", **TITLE_KW)
    style_ax(ax_sc, xlabel="ŷ (latente quantizado)", ylabel="μ (ARM)")
    ax_sc.legend(**LEGEND_KW)

    fig.suptitle("Cool-Chic — Análise comparativa: imagens, latentes e ARM",
                 color=WHITE, fontsize=13, fontweight="bold", y=1.005)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"✅  Figura salva em: {save_path}")
    else:
        plt.show()

    plt.close(fig)


    plt.close(fig)


# ─────────────────────────────────────────────────
# ERP Analysis (Tabela e Plot)
# ─────────────────────────────────────────────────

def print_erp_report(data: dict) -> None:
    W = 74
    sep = "─" * W
    print(f"\n{'═'*W}")
    print(f"  ANÁLISE ERP (ZONAS) — Cool-Chic")
    print(f"{'═'*W}")

    print(f"\n  {'[ LATENTES QUANTIZADOS ]'}")
    print(sep)
    
    for i, lat in enumerate(data["latents"]):
        h, w = lat.shape[-2:]
        print(f"\n  Latente [{i}]  {h:4d}×{w:<4d}")
        print(f"  {'Zona':<14}  {'min':>7}  {'max':>7}  {'μ':>8}  {'σ':>8}  {'zeros':>8}")
        print(f"  " + "─"*58)
        
        zonas = split_erp_zones(lat)
        for nome_zona, _, zona in zonas:
            s = tensor_stats(zona)
            sparsity = (zona == 0).float().mean().item() * 100
            print(f"  {nome_zona:<14}  {s['min']:>7.2f}  {s['max']:>7.2f}"
                  f"  {s['mean']:>+8.4f}  {s['std']:>8.4f}  {sparsity:>7.1f}%")
    print()


def plot_erp_analysis(data: dict, save_path=None, n_bins: int = 128):
    latents = data["latents"]
    n_latents = len(latents)
    
    n_cols = max(3, min(n_latents, 4))
    n_rows = (n_latents - 1) // n_cols + 1
    
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.5), facecolor=BG)
    gs  = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.3)
    
    TITLE_KW  = dict(color=WHITE, fontsize=10, fontweight="bold", pad=5)
    LEGEND_KW = dict(fontsize=7.5, facecolor=PANEL, labelcolor=WHITE,
                     framealpha=0.85, edgecolor="#30363d")
                     
    all_lat_vals = np.concatenate([lat.flatten().numpy() for lat in latents])
    lat_min, lat_max = all_lat_vals.min(), all_lat_vals.max()
                     
    for i, lat in enumerate(latents):
        r, c = divmod(i, n_cols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_facecolor(PANEL)
        
        zonas = split_erp_zones(lat)
        
        for nome_zona, cor, zona in zonas:
            zona_np = zona.flatten().numpy()
            counts, edges = np.histogram(zona_np, bins=n_bins, range=(lat_min, lat_max))
            centers = (edges[:-1] + edges[1:]) / 2
            
            mu = zona_np.mean()
            std = zona_np.std()
            
            ax.fill_between(centers, counts, alpha=0.25, color=cor)
            ax.plot(centers, counts, color=cor, linewidth=1.2,
                    label=f"{nome_zona} (σ={std:.2f})")
            ax.axvline(mu, color=cor, linewidth=1.0, linestyle="--")
            
        h, w = lat.shape[-2:]
        ax.set_title(f"Latente [{i}]  {h}×{w}", **TITLE_KW)
        style_ax(ax, xlabel="ŷ (quantizado)")
        ax.set_xlim(lat_min, lat_max)
        ax.legend(**LEGEND_KW)
        
    fig.suptitle("Distribuição dos Latentes por Zona ERP\nPolo Norte (Azul) | Equador (Amarelo) | Polo Sul (Rosa)", 
                 color=WHITE, fontsize=12, fontweight="bold", y=1.04)
                 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
        print(f"✅  Figura salva em: {save_path}")
    else:
        plt.show()
        
    plt.close(fig)


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Análise comparativa Cool-Chic: imagem original vs decodificada + latentes + ARM."
    )
    parser.add_argument(
        "--model", required=True,
        help="Caminho para o checkpoint do FrameEncoder (.pt)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Caminho para a imagem original (.png)."
    )
    parser.add_argument(
        "--save", default=None,
        help="Salva a figura no caminho indicado (opcional)."
    )
    parser.add_argument(
        "--bins", type=int, default=128,
        help="Número de bins dos histogramas (padrão: 128)."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed do numpy para o scatter amostrado (padrão: 0)."
    )
    parser.add_argument(
        "--erp-zones", action="store_true",
        help="Ativa a visualização separada por zonas ERP (Polos vs Equador)."
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"[1/3] Carregando modelo: {args.model}")
    frame_encoder = load_frame_encoder(args.model)

    print(f"[2/3] Carregando imagem: {args.input}")
    original_img = load_image(args.input)

    print(f"[3/3] Executando forward pass e extraindo dados...")
    data = extract_model_data(frame_encoder, original_img)

    if args.erp_zones:
        print_erp_report(data)
        plot_erp_analysis(data, save_path=args.save, n_bins=args.bins)
    else:
        print_report(data)
        plot_analysis(data, save_path=args.save, n_bins=args.bins)


if __name__ == "__main__":
    main()
