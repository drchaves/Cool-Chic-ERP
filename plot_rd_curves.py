"""
plot_rd_curves.py

Lê o arquivo benchmark_all.tsv e gera uma curva Rate-Distortion (RD)
comparando os modos 'standard' e 'erp'.

- Eixo X: PSNR (dB)
- Eixo Y: Rate (bpp)

A curva é a MÉDIA sobre todas as imagens para cada (modo, lambda).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── configuração ──────────────────────────────────────────────────────────────
TSV_PATH = os.path.join(os.path.dirname(__file__),
                        "benchmark_results_ctc_final_final", "benchmark_all.tsv")
OUT_PATH  = os.path.join(os.path.dirname(__file__),
                        "benchmark_results_ctc_final_final", "rd_curve.png")

# ── estilo ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#161b27",
    "axes.edgecolor":    "#2e3650",
    "axes.labelcolor":   "#c8d0e8",
    "axes.titlecolor":   "#e8eaf6",
    "xtick.color":       "#8892b0",
    "ytick.color":       "#8892b0",
    "grid.color":        "#1e2640",
    "grid.linewidth":    0.8,
    "text.color":        "#c8d0e8",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

PALETTE = {
    "standard": {"color": "#64b5f6", "marker": "o", "ls": "-",  "lw": 2.2},
    "erp":      {"color": "#f06292", "marker": "s", "ls": "--", "lw": 2.2},
}
LABEL = {"standard": "Standard (rectangular ARM)", "erp": "ERP (geodesic ARM)"}

# ── leitura ───────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep=r"\s+", engine="python")
df.columns = df.columns.str.strip()
df["ws_psnr_db"]  = df["ws_psnr_db"].astype(float)
df["rate_bpp"] = df["rate_bpp"].astype(float)
df["lmbda"]    = df["lmbda"].astype(float)
df["mode"]     = df["mode"].str.strip()

# ── média sobre imagens para cada (modo, lambda) ──────────────────────────────
avg = (df
       .groupby(["mode", "lmbda"], as_index=False)
       .agg(psnr_mean=("ws_psnr_db",  "mean"),
            bpp_mean =("rate_bpp", "mean"))
       .sort_values("psnr_mean"))

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))

for mode, grp in avg.groupby("mode"):
    grp = grp.sort_values("psnr_mean")
    s = PALETTE[mode]
    ax.plot(grp["psnr_mean"], grp["bpp_mean"],
            color=s["color"], marker=s["marker"], ls=s["ls"], lw=s["lw"],
            markersize=8, markeredgewidth=1.4, markeredgecolor="#0f1117",
            label=LABEL[mode], zorder=3)

    # anotar cada ponto com o valor de λ
    for _, row in grp.iterrows():
        lbl = f"λ={row['lmbda']:.4g}"
        offset_x = 0.08
        offset_y = 0.012 if mode == "standard" else -0.022
        ax.annotate(lbl,
                    xy=(row["psnr_mean"], row["bpp_mean"]),
                    xytext=(row["psnr_mean"] + offset_x, row["bpp_mean"] + offset_y),
                    fontsize=7.5, color=s["color"], alpha=0.85,
                    arrowprops=None)

ax.set_xlabel("PSNR (dB)", fontsize=13, labelpad=8)
ax.set_ylabel("Rate (bpp)", fontsize=13, labelpad=8)
ax.set_title("Rate-Distortion: Standard vs ERP\n(média sobre todas as imagens)",
             fontsize=14, pad=14, fontweight="bold")

ax.grid(True, which="both", alpha=0.4)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# legenda com fundo semi-transparente
leg = ax.legend(fontsize=11, framealpha=0.25, facecolor="#1e2640",
                edgecolor="#2e3650", loc="upper left")
for text in leg.get_texts():
    text.set_color("#e8eaf6")

# referência visual: região onde ERP < standard (melhor compressão)
std = avg[avg["mode"] == "standard"].sort_values("psnr_mean")
erp = avg[avg["mode"] == "erp"].sort_values("psnr_mean")

# interp bpp do standard nos pontos do erp para preencher a área entre curvas
if len(std) > 1 and len(erp) > 1:
    psnr_common = np.linspace(
        max(std["psnr_mean"].min(), erp["psnr_mean"].min()),
        min(std["psnr_mean"].max(), erp["psnr_mean"].max()),
        200)
    bpp_std_i = np.interp(psnr_common, std["psnr_mean"], std["bpp_mean"])
    bpp_erp_i = np.interp(psnr_common, erp["psnr_mean"], erp["bpp_mean"])
    ax.fill_between(psnr_common, bpp_std_i, bpp_erp_i,
                    where=(bpp_erp_i < bpp_std_i),
                    color="#f06292", alpha=0.08, label="_erp_better")
    ax.fill_between(psnr_common, bpp_std_i, bpp_erp_i,
                    where=(bpp_erp_i >= bpp_std_i),
                    color="#64b5f6", alpha=0.08, label="_std_better")

# bordas do plot
for spine in ax.spines.values():
    spine.set_edgecolor("#2e3650")

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Plot salvo em: {OUT_PATH}")
plt.show()
