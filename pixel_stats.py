"""
Histograma e estatísticas descritivas de pixels.

Uso básico:
    python pixel_stats.py --image caminho/para/imagem.png
    python pixel_stats.py --image img1.png img2.png  (compara múltiplas)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────
# Carregamento
# ──────────────────────────────────────────────────────────────────

def load_image_as_tensor(path: str) -> torch.Tensor:
    """Carrega imagem do disco → tensor float32 em [0, 1], shape [C, H, W]."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # [H, W, C] em [0,1]
    return torch.from_numpy(arr).permute(2, 0, 1)   # → [C, H, W]


# ──────────────────────────────────────────────────────────────────
# Estatísticas descritivas
# ──────────────────────────────────────────────────────────────────

def compute_stats(tensor: torch.Tensor, name: str = "") -> dict:
    """
    Calcula estatísticas descritivas de um tensor qualquer.

    Parâmetros
    ----------
    tensor : torch.Tensor  — valores em qualquer range/shape
    name   : str           — rótulo para exibição

    Retorna
    -------
    dict com as métricas calculadas
    """
    flat = tensor.flatten().float()

    stats = {
        "name"    : name,
        "shape"   : tuple(tensor.shape),
        "n_pixels": flat.numel(),
        "min"     : flat.min().item(),
        "max"     : flat.max().item(),
        "mean"    : flat.mean().item(),
        "std"     : flat.std().item(),
        "median"  : flat.median().item(),
        "q25"     : flat.quantile(0.25).item(),
        "q75"     : flat.quantile(0.75).item(),
        "iqr"     : (flat.quantile(0.75) - flat.quantile(0.25)).item(),
        "skewness": _skewness(flat),
        "kurtosis": _kurtosis(flat),
    }
    return stats


def _skewness(x: torch.Tensor) -> float:
    """Assimetria (skewness) amostral."""
    mu  = x.mean()
    std = x.std()
    if std == 0:
        return 0.0
    return (((x - mu) / std) ** 3).mean().item()


def _kurtosis(x: torch.Tensor) -> float:
    """Curtose (excess kurtosis) amostral — 0 para distribuição normal."""
    mu  = x.mean()
    std = x.std()
    if std == 0:
        return 0.0
    return ((((x - mu) / std) ** 4).mean() - 3).item()


def print_stats(stats: dict) -> None:
    """Imprime tabela formatada de estatísticas."""
    W = 42
    print("\n" + "─" * W)
    print(f"  📊  {stats['name']}")
    print(f"  Shape : {stats['shape']}    N = {stats['n_pixels']:,}")
    print("─" * W)
    rows = [
        ("Mínimo",            stats["min"],      ""),
        ("Máximo",            stats["max"],      ""),
        ("Média (μ)",         stats["mean"],     ""),
        ("Desvio-padrão (σ)", stats["std"],      ""),
        ("Mediana",           stats["median"],   ""),
        ("1º Quartil (Q1)",   stats["q25"],      ""),
        ("3º Quartil (Q3)",   stats["q75"],      ""),
        ("IQR (Q3 – Q1)",     stats["iqr"],      ""),
        ("Assimetria",        stats["skewness"], " (0 = simétrico)"),
        ("Curtose (excess)",  stats["kurtosis"], " (0 = normal)"),
    ]
    for label, value, note in rows:
        print(f"  {label:<22} {value:>+10.5f}{note}")
    print("─" * W)


# ──────────────────────────────────────────────────────────────────
# Visualização
# ──────────────────────────────────────────────────────────────────

CHANNEL_COLORS = {0: "#e74c3c", 1: "#2ecc71", 2: "#3498db"}  # R G B
CHANNEL_NAMES  = {0: "R", 1: "G", 2: "B"}

# Zonas ERP: (nome, cor, fracao_inicio, fracao_fim)
ERP_ZONES = [
    ("Polo Norte", "#74b9ff", 0.00, 0.25),
    ("Equador",    "#fdcb6e", 0.25, 0.75),
    ("Polo Sul",   "#fd79a8", 0.75, 1.00),
]


def plot_histogram(
    tensors,
    names,
    n_bins: int = 256,
    save_path=None,
    per_channel: bool = True,
):
    """
    Plota histograma(s) completo(s) com estatísticas.

    tensors     : lista de tensores [C, H, W] em [0, 1]
    names       : rótulo de cada tensor
    n_bins      : número de bins do histograma
    per_channel : True → 3 subplots separados (R, G, B)
                  False → 1 subplot com os 3 canais sobrepostos
    """
    n_images  = len(tensors)
    n_rows    = 4 if per_channel else 2          # imagem + (3 canais ou 1 combinado)
    fig_h     = 10 if per_channel else 5
    fig = plt.figure(figsize=(7 * n_images, fig_h), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")

    outer = gridspec.GridSpec(1, n_images, figure=fig, hspace=0.4)

    for img_idx, (tensor, name) in enumerate(zip(tensors, names)):
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, 1, subplot_spec=outer[img_idx], hspace=0.55
        )

        # ── linha 0: imagem ───────────────────────────────────────
        ax_img = fig.add_subplot(inner[0])
        img_np = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        ax_img.imshow(img_np)
        ax_img.set_title(name, color="white", fontsize=13, fontweight="bold", pad=8)
        ax_img.axis("off")

        all_pixels = tensor.flatten().float()
        vmin, vmax = all_pixels.min().item(), all_pixels.max().item()

        if per_channel:
            # ── linhas 1–3: um subplot por canal (R, G, B) ───────
            for ch in range(tensor.shape[0]):
                ax = fig.add_subplot(inner[ch + 1])
                ax.set_facecolor("#16213e")

                pixels = tensor[ch].flatten().float().numpy()
                color  = CHANNEL_COLORS.get(ch, "#ffffff")
                label  = CHANNEL_NAMES.get(ch, f"Ch{ch}")

                counts, bin_edges = np.histogram(pixels, bins=n_bins, range=(vmin, vmax))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.fill_between(bin_centers, counts, alpha=0.45, color=color)
                ax.plot(bin_centers, counts, color=color, linewidth=1.0, label=label)

                mu, sigma = pixels.mean(), pixels.std()
                ax.axvline(mu,         color="white", linewidth=1.2, linestyle="--",
                           label=f"μ={mu:.3f}")
                ax.axvline(mu - sigma, color=color,   linewidth=0.8, linestyle=":",
                           alpha=0.8, label=f"σ={sigma:.3f}")
                ax.axvline(mu + sigma, color=color,   linewidth=0.8, linestyle=":",
                           alpha=0.8)

                ax.set_xlim(vmin, vmax)
                ax.tick_params(colors="white", labelsize=7)
                for spine in ax.spines.values():
                    spine.set_color("#444")
                ax.set_facecolor("#16213e")
                ax.set_ylabel("Frequência", color="#aaa", fontsize=7)
                ax.set_xlabel("Valor do pixel", color="#aaa", fontsize=7)
                ax.legend(fontsize=7, loc="upper right",
                          facecolor="#0f3460", labelcolor="white",
                          framealpha=0.8, edgecolor="#444")

        else:
            # ── linha 1: canais R, G, B sobrepostos no mesmo eixo ─
            ax = fig.add_subplot(inner[1])
            ax.set_facecolor("#16213e")

            for ch in range(tensor.shape[0]):
                pixels = tensor[ch].flatten().float().numpy()
                color  = CHANNEL_COLORS.get(ch, "#ffffff")
                label  = CHANNEL_NAMES.get(ch, f"Ch{ch}")

                counts, bin_edges = np.histogram(pixels, bins=n_bins, range=(vmin, vmax))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax.fill_between(bin_centers, counts, alpha=0.25, color=color)
                ax.plot(bin_centers, counts, color=color, linewidth=1.2,
                        label=f"{label}  μ={pixels.mean():.3f}  σ={pixels.std():.3f}")

            # Linha de média global
            mu_all   = all_pixels.mean().item()
            std_all  = all_pixels.std().item()
            ax.axvline(mu_all,           color="white", linewidth=1.2,
                       linestyle="--", label=f"μ global={mu_all:.3f}")
            ax.axvline(mu_all - std_all, color="white", linewidth=0.8,
                       linestyle=":", alpha=0.7, label=f"σ global={std_all:.3f}")
            ax.axvline(mu_all + std_all, color="white", linewidth=0.8,
                       linestyle=":", alpha=0.7)

            ax.set_xlim(vmin, vmax)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#444")
            ax.set_ylabel("Frequência", color="#aaa", fontsize=8)
            ax.set_xlabel("Valor do pixel", color="#aaa", fontsize=8)
            ax.legend(fontsize=8, loc="upper right",
                      facecolor="#0f3460", labelcolor="white",
                      framealpha=0.8, edgecolor="#444")

        fig.text(
            0.5 / n_images + img_idx / n_images,
            -0.01,
            f"min={vmin:.3f}  max={vmax:.3f}  "
            f"μ={all_pixels.mean():.4f}  σ={all_pixels.std():.4f}",
            ha="center", va="top", color="#aaa", fontsize=8,
            transform=fig.transFigure,
        )

    title = "Histograma de pixels por canal" if per_channel else "Histograma de pixels (canais sobrepostos)"
    fig.suptitle(title, color="white", fontsize=15, fontweight="bold", y=1.01)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n✅  Figura salva em: {save_path}")
    else:
        plt.show()


# ──────────────────────────────────────────────────────────────────
# ERP: fatiamento por zona de latitude
# ──────────────────────────────────────────────────────────────────

def split_erp_zones(tensor: torch.Tensor) -> list:
    """
    Divide um tensor [C, H, W] em 3 faixas de latitude ERP.

    Retorna lista de (nome, cor, sub_tensor [C, h_i, W]).
    """
    C, H, W = tensor.shape
    resultado = []
    for nome, cor, f0, f1 in ERP_ZONES:
        r0 = int(H * f0)
        r1 = int(H * f1)
        zona = tensor[:, r0:r1, :]   # [C, h_zona, W]
        resultado.append((nome, cor, zona))
    return resultado


def plot_erp_zones(
    tensor: torch.Tensor,
    image_name: str,
    n_bins: int = 256,
    save_path=None,
):
    """
    Plota histogramas das 3 zonas ERP sobrepostos, um subplot por canal.

    tensor     : [C, H, W] float32 em [0, 1]
    image_name : nome para o título
    """
    zonas   = split_erp_zones(tensor)
    C       = tensor.shape[0]
    vmin    = tensor.min().item()
    vmax    = tensor.max().item()

    fig, axes = plt.subplots(1, C, figsize=(6 * C, 5), constrained_layout=True)
    fig.patch.set_facecolor("#1a1a2e")
    if C == 1:
        axes = [axes]

    LEGEND_KW = dict(fontsize=7.5, facecolor="#0f3460", labelcolor="white",
                     framealpha=0.85, edgecolor="#444")

    for ch_idx, ax in enumerate(axes):
        ax.set_facecolor("#16213e")
        ch_name = CHANNEL_NAMES.get(ch_idx, f"Ch{ch_idx}")

        for nome, cor, zona in zonas:
            pixels = zona[ch_idx].flatten().float().numpy()
            counts, edges = np.histogram(pixels, bins=n_bins, range=(vmin, vmax))
            centers = (edges[:-1] + edges[1:]) / 2

            mu  = pixels.mean()
            std = pixels.std()

            ax.fill_between(centers, counts, alpha=0.20, color=cor)
            ax.plot(centers, counts, color=cor, linewidth=1.4,
                    label=f"{nome}  μ={mu:.3f}  σ={std:.3f}")
            ax.axvline(mu, color=cor, linewidth=1.2, linestyle="--", alpha=0.9)

        ax.set_title(f"Canal {ch_name}", color="white", fontsize=10,
                     fontweight="bold", pad=6)
        ax.set_xlabel("Valor do pixel", color="#aaa", fontsize=8)
        ax.set_ylabel("Frequência",     color="#aaa", fontsize=8)
        ax.set_xlim(vmin, vmax)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.legend(**LEGEND_KW)

    fig.suptitle(
        f"Distribuição por zona ERP — {image_name}\n"
        f"Polo Norte 0–25% | Equador 25–75% | Polo Sul 75–100%",
        color="white", fontsize=11, fontweight="bold",
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n✅  Figura salva em: {save_path}")
    else:
        plt.show()

    plt.close(fig)


def analyze_erp_zones(
    image_paths: list,
    n_bins: int = 256,
    save_path=None,
) -> None:
    """
    Carrega imagem(ns), imprime tabela comparativa entre zonas ERP
    e plota histogramas sobrepostos.
    """
    W_LINE = 78
    SEP    = "─" * W_LINE

    for path in image_paths:
        t    = load_image_as_tensor(path)
        nome = Path(path).name
        C, H, _ = t.shape

        print(f"\n{'═' * W_LINE}")
        print(f"  🌐  Análise ERP por zona — {nome}  ({H}×{t.shape[2]}px)")
        print(f"{'═' * W_LINE}")

        # Cabeçalho da tabela
        print(f"\n  {'Zona':<14}"
              f"  {'Canal':<6}"
              f"  {'min':>7}"
              f"  {'max':>7}"
              f"  {'μ':>8}"
              f"  {'σ':>8}"
              f"  {'mediana':>8}"
              f"  {'IQR':>8}"
              f"  {'assimetria':>10}")
        print(f"  {SEP}")

        zonas = split_erp_zones(t)
        for nome_zona, _, zona in zonas:
            # Linha global da zona (todos os canais juntos)
            s_global = compute_stats(zona, name=nome_zona)
            print(f"  {nome_zona:<14}  {'global':<6}"
                  f"  {s_global['min']:>7.4f}"
                  f"  {s_global['max']:>7.4f}"
                  f"  {s_global['mean']:>+8.4f}"
                  f"  {s_global['std']:>8.4f}"
                  f"  {s_global['median']:>+8.4f}"
                  f"  {s_global['iqr']:>8.4f}"
                  f"  {s_global['skewness']:>+10.4f}")

            # Uma linha por canal R, G, B
            for ch_idx in range(C):
                s = compute_stats(zona[ch_idx],
                                  name=f"{nome_zona}/{CHANNEL_NAMES[ch_idx]}")
                ch_name = CHANNEL_NAMES.get(ch_idx, f"Ch{ch_idx}")
                print(f"  {'':14}  {ch_name:<6}"
                      f"  {s['min']:>7.4f}"
                      f"  {s['max']:>7.4f}"
                      f"  {s['mean']:>+8.4f}"
                      f"  {s['std']:>8.4f}"
                      f"  {s['median']:>+8.4f}"
                      f"  {s['iqr']:>8.4f}"
                      f"  {s['skewness']:>+10.4f}")
            print(f"  {SEP}")

        plot_erp_zones(t, Path(path).name, n_bins=n_bins, save_path=save_path)


# ──────────────────────────────────────────────────────────────────
# Função de alto nível — análise de imagens no disco
# ──────────────────────────────────────────────────────────────────

def analyze_images(
    image_paths,
    n_bins: int = 256,
    save_path=None,
    per_channel: bool = True,
) -> None:
    """
    Carrega imagens, imprime estatísticas e plota histograma.

    image_paths : lista de caminhos para imagens
    n_bins      : número de bins do histograma
    save_path   : se fornecido, salva a figura em vez de exibir
    per_channel : True para estatísticas separadas por canal R/G/B
    """
    tensors, names = [], []

    for path in image_paths:
        t = load_image_as_tensor(path)
        n = Path(path).name
        tensors.append(t)
        names.append(n)

        # Estatísticas globais
        print_stats(compute_stats(t, name=f"{n} — todos os canais"))

        # Estatísticas por canal
        if per_channel:
            for ch, ch_name in CHANNEL_NAMES.items():
                if ch < t.shape[0]:
                    print_stats(compute_stats(t[ch], name=f"  Canal {ch_name}"))

    plot_histogram(tensors, names, n_bins=n_bins, save_path=save_path, per_channel=per_channel)


# ──────────────────────────────────────────────────────────────────
# Função de alto nível — análise de tensor arbitrário
# (útil para latentes do Cool-Chic, saída do ARM, etc.)
# ──────────────────────────────────────────────────────────────────

def analyze_tensor(
    tensor: torch.Tensor,
    name: str = "Tensor",
    n_bins: int = 256,
    save_path=None,
) -> dict:
    """
    Analisa qualquer tensor (ex: latentes do Cool-Chic, saída do ARM, etc.).

    tensor     : shape qualquer
    name       : rótulo para exibição
    n_bins     : bins do histograma
    save_path  : path para salvar figura (opcional)

    Retorna dict com as estatísticas.
    """
    stats = compute_stats(tensor, name=name)
    print_stats(stats)

    flat = tensor.flatten().float().numpy()

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    counts, bin_edges = np.histogram(flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax.fill_between(bin_centers, counts, alpha=0.4, color="#a29bfe")
    ax.plot(bin_centers, counts, color="#a29bfe", linewidth=1.2)

    mu, sigma = flat.mean(), flat.std()
    ax.axvline(mu, color="white", linestyle="--", linewidth=1.5,
               label=f"μ = {mu:.4f}")
    ax.axvline(mu - sigma, color="#fd79a8", linestyle=":", linewidth=1.0,
               label=f"μ ± σ  (σ = {sigma:.4f})")
    ax.axvline(mu + sigma, color="#fd79a8", linestyle=":", linewidth=1.0)

    ax.set_title(name, color="white", fontsize=13, fontweight="bold")
    ax.set_xlabel("Valor", color="#aaa")
    ax.set_ylabel("Frequência", color="#aaa")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.legend(facecolor="#0f3460", labelcolor="white", framealpha=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"✅  Figura salva em: {save_path}")
    else:
        plt.show()

    return stats


# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Histograma e estatísticas descritivas de imagens."
    )
    parser.add_argument(
        "--image", nargs="+", required=True,
        help="Caminho(s) para imagem(ns) PNG/JPEG."
    )
    parser.add_argument(
        "--bins", type=int, default=256,
        help="Número de bins do histograma (padrão: 256)."
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Salva a figura no caminho indicado em vez de exibir."
    )
    parser.add_argument(
        "--no-per-channel", action="store_true",
        help="Omite estatísticas separadas por canal R/G/B."
    )
    parser.add_argument(
        "--erp-zones", action="store_true",
        help="Modo ERP: divide em Polo Norte (25%%), Equador (50%%), Polo Sul (25%%) e compara."
    )
    args = parser.parse_args()

    if args.erp_zones:
        analyze_erp_zones(
            image_paths=args.image,
            n_bins=args.bins,
            save_path=args.save,
        )
    else:
        analyze_images(
            image_paths=args.image,
            n_bins=args.bins,
            save_path=args.save,
            per_channel=not args.no_per_channel,
        )
